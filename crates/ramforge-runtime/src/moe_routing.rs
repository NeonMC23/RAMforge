//! MoE expert layout, routing aggregation, and residency simulation.
//!
//! Routing observations come from ForgeCore's read-only `ffn_moe_topk` capture.
//! They describe which experts the native router selected; they do not prove
//! which expert pages were resident or which tensors the backend read.
//!
//! Cache simulations are theoretical. They never move data and must not be
//! reported as measured inference improvements.

use std::collections::{BTreeMap, HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use ramforge_core::GgufModel;

use crate::ExpertRoutingObservation;

pub const MOE_ROUTING_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertLayerLayout {
    pub layer: u32,
    pub expert_count: u64,
    pub expert_used_count: Option<u64>,
    /// Sum of `*_exps` tensors in this layer.
    pub routed_tensor_bytes: u64,
    /// `routed_tensor_bytes / expert_count`, rounded up. Metadata arithmetic,
    /// not a native allocation.
    pub per_expert_bytes: Option<u64>,
    pub shared_expert_bytes: u64,
    pub other_tensor_bytes: u64,
    pub tensor_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertLayout {
    pub architecture: Option<String>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub layers: Vec<ExpertLayerLayout>,
    pub total_routed_expert_bytes: u64,
    pub total_shared_expert_bytes: u64,
    pub total_other_bytes: u64,
    pub note: String,
}

/// Build per-layer expert layout from GGUF tensor names and byte lengths.
pub fn expert_layout(model: &GgufModel) -> ExpertLayout {
    let info = model.info();
    let expert_count = info.expert_count;
    let mut layers: BTreeMap<u32, ExpertLayerLayout> = BTreeMap::new();
    for tensor in &model.tensors {
        let Some(layer) = layer_from_tensor_name(&tensor.name) else {
            continue;
        };
        let bytes = tensor.byte_length.unwrap_or(0);
        let entry = layers.entry(layer).or_insert_with(|| ExpertLayerLayout {
            layer,
            expert_count: expert_count.unwrap_or(0),
            expert_used_count: info.expert_used_count,
            routed_tensor_bytes: 0,
            per_expert_bytes: None,
            shared_expert_bytes: 0,
            other_tensor_bytes: 0,
            tensor_names: Vec::new(),
        });
        entry.tensor_names.push(tensor.name.clone());
        if tensor.name.contains("_shexp") || tensor.name.contains("shared_expert") {
            entry.shared_expert_bytes = entry.shared_expert_bytes.saturating_add(bytes);
        } else if tensor.name.contains("_exps") || tensor.name.contains(".expert") {
            entry.routed_tensor_bytes = entry.routed_tensor_bytes.saturating_add(bytes);
        } else {
            entry.other_tensor_bytes = entry.other_tensor_bytes.saturating_add(bytes);
        }
    }
    let mut layers: Vec<_> = layers.into_values().collect();
    for layer in &mut layers {
        if layer.expert_count > 0 {
            layer.per_expert_bytes = layer
                .routed_tensor_bytes
                .checked_add(layer.expert_count - 1)
                .and_then(|bytes| bytes.checked_div(layer.expert_count));
        }
    }
    let total_routed_expert_bytes = layers.iter().map(|layer| layer.routed_tensor_bytes).sum();
    let total_shared_expert_bytes = layers.iter().map(|layer| layer.shared_expert_bytes).sum();
    let total_other_bytes = layers.iter().map(|layer| layer.other_tensor_bytes).sum();
    ExpertLayout {
        architecture: info.architecture.clone(),
        expert_count,
        expert_used_count: info.expert_used_count,
        layers,
        total_routed_expert_bytes,
        total_shared_expert_bytes,
        total_other_bytes,
        note: "GGUF tensor-name/byte arithmetic; expert row sizes assume uniform experts per layer and are not native allocations"
            .to_string(),
    }
}

fn layer_from_tensor_name(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse::<u32>().ok()
}

/// One expert's byte range inside a file-backed expert tensor.
///
/// This is read-only metadata. The pinned native API accepts tensor indices
/// and whole-tensor page prefetch/release, not sub-tensor expert ranges, so
/// this range can describe a future capability but cannot currently be moved
/// independently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertByteRange {
    pub layer: u32,
    pub tensor_name: String,
    pub expert: u32,
    pub file_path: Option<String>,
    pub file_offset: u64,
    pub bytes: u64,
    pub page_start: u64,
    pub page_end: u64,
    pub mapping_id: Option<u64>,
    pub mapping_generation: Option<u64>,
}

/// Compute per-expert byte ranges for the selected experts in one layer.
///
/// The GGUF expert dimension is the outermost tensor dimension, so each expert
/// occupies a contiguous `bytes / expert_count` block after the tensor's file
/// offset. The calculation is rejected for non-file-backed tensors, unknown
/// dimensions, or an inconsistent byte length.
pub fn expert_byte_ranges(
    model: &GgufModel,
    mappings: &[crate::TensorMappingInfo],
    layer: u32,
    experts: &[u32],
) -> Result<Vec<ExpertByteRange>, String> {
    let info = model.info();
    let expert_count = info
        .expert_count
        .ok_or_else(|| "expert_count metadata is unavailable".to_string())?;
    if expert_count == 0 {
        return Err("expert_count is zero".to_string());
    }
    let mut output = Vec::new();
    for tensor in &model.tensors {
        if layer_from_tensor_name(&tensor.name) != Some(layer) {
            continue;
        }
        if !(tensor.name.contains("_exps") || tensor.name.contains(".expert")) {
            continue;
        }
        let Some(mapping) = mappings.iter().find(|mapping| mapping.name == tensor.name) else {
            return Err(format!("mapping metadata missing for {}", tensor.name));
        };
        if !mapping.file_backed || mapping.page_size == 0 {
            return Err(format!("{} is not file-backed", tensor.name));
        }
        let Some(bytes) = tensor.byte_length else {
            return Err(format!("{} byte length is unknown", tensor.name));
        };
        if bytes % expert_count != 0 {
            return Err(format!(
                "{} byte length {} is not divisible by {} experts",
                tensor.name, bytes, expert_count
            ));
        }
        let per_expert = bytes / expert_count;
        for expert in experts {
            if u64::from(*expert) >= expert_count {
                return Err(format!(
                    "expert {} is outside 0..{} for {}",
                    expert, expert_count, tensor.name
                ));
            }
            let file_offset = mapping
                .offset
                .checked_add(u64::from(*expert).saturating_mul(per_expert))
                .ok_or_else(|| format!("byte offset overflow for {}", tensor.name))?;
            let end = file_offset
                .checked_add(per_expert)
                .ok_or_else(|| format!("byte range overflow for {}", tensor.name))?;
            output.push(ExpertByteRange {
                layer,
                tensor_name: tensor.name.clone(),
                expert: *expert,
                file_path: mapping.file_path.clone(),
                file_offset,
                bytes: per_expert,
                page_start: file_offset / mapping.page_size * mapping.page_size,
                page_end: end.div_ceil(mapping.page_size) * mapping.page_size,
                mapping_id: mapping.mapping_id,
                mapping_generation: mapping.mapping_generation,
            });
        }
    }
    if output.is_empty() {
        return Err(format!("no routed expert tensors found for layer {layer}"));
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingObservationRecord {
    pub execution_id: u64,
    pub sequence: u64,
    pub layer: Option<i32>,
    pub selected_experts: Vec<u32>,
    pub tokens_in_graph: Option<u32>,
    pub complete_for_token: bool,
    pub invalid_indices: u32,
}

impl From<&ExpertRoutingObservation> for RoutingObservationRecord {
    fn from(observation: &ExpertRoutingObservation) -> Self {
        Self {
            execution_id: observation.execution_id,
            sequence: observation.sequence,
            layer: observation.layer,
            selected_experts: observation.selected_experts.clone(),
            tokens_in_graph: observation.tokens_in_graph,
            complete_for_token: observation.complete_for_token,
            invalid_indices: observation.invalid_indices,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerRoutingReport {
    pub layer: u32,
    pub observations: u64,
    pub unique_experts: u32,
    /// Selection count indexed by expert id.
    pub selection_counts: Vec<u64>,
    /// Highest-frequency experts, descending, with counts.
    pub top_experts: Vec<(u32, u64)>,
    /// Mean intersection of consecutive token selections, basis points of k.
    pub consecutive_overlap_bps: Option<u32>,
    /// Mean distance (in layer observations) between repeated selections of the
    /// same expert, excluding the first occurrence.
    pub mean_reuse_distance: Option<f64>,
    pub per_expert_bytes: Option<u64>,
    pub unique_expert_bytes: Option<u64>,
    pub mean_active_bytes_per_token: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationPolicy {
    /// Online least-recently-used cache.
    Lru,
    /// Offline static frequency cache: the most frequently selected experts
    /// across the trace. This is an upper bound, not an implementable online
    /// policy unless selection frequencies are stable and known in advance.
    OfflineFrequency,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidencySimulation {
    pub policy: SimulationPolicy,
    /// Global expert slots across all layers.
    pub capacity_slots: u32,
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate_bps: u32,
    /// Sum of per-expert bytes for hits; theoretical avoided transfer volume.
    pub avoided_transfer_bytes: u64,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefetchAssessment {
    /// Bytes in the previous token's active expert set across all layers.
    pub previous_token_active_set_bytes: u64,
    /// Measured consecutive-token overlap, used as the expected reuse.
    pub expected_reuse_bps: Option<u32>,
    pub expected_used_bytes: Option<u64>,
    pub expected_wasted_bytes: Option<u64>,
    /// False when the current graph/API cannot use this prefetch safely or
    /// profitably. The note states the concrete reason.
    pub feasible: bool,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingAggregate {
    pub schema_version: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub total_observations: usize,
    pub complete_decode_observations: u64,
    pub partial_observations: u64,
    pub invalid_indices: u64,
    pub executions: u64,
    pub selected_expert_slots: u64,
    pub unique_experts: u32,
    /// Aggregate selection count by expert id across all layers.
    pub selection_counts: Vec<u64>,
    pub layers: Vec<LayerRoutingReport>,
    pub consecutive_overlap_bps: Option<u32>,
    pub mean_reuse_distance: Option<f64>,
    pub simulations: Vec<ResidencySimulation>,
    pub prefetch: Option<PrefetchAssessment>,
    pub notes: Vec<String>,
}

/// Aggregate observations. Only complete single-token observations contribute
/// to locality and cache-simulation statistics; prefill observations are
/// counted separately because the prototype copies only the first token row.
pub fn aggregate_routing(
    observations: &[ExpertRoutingObservation],
    layout: Option<&ExpertLayout>,
    capacities: &[u32],
) -> Option<RoutingAggregate> {
    let first = observations.first()?;
    let expert_count = first.expert_count;
    let expert_used_count = first.expert_used_count;
    if expert_count == 0 || expert_used_count == 0 {
        return None;
    }
    let mut records: Vec<RoutingObservationRecord> = observations
        .iter()
        .map(RoutingObservationRecord::from)
        .collect();
    records.sort_by_key(|record| (record.execution_id, record.sequence));

    let decode: Vec<&RoutingObservationRecord> = records
        .iter()
        .filter(|record| record.complete_for_token && record.invalid_indices == 0)
        .collect();
    let partial = records
        .iter()
        .filter(|record| !record.complete_for_token)
        .count() as u64;
    let invalid_indices = records
        .iter()
        .map(|record| u64::from(record.invalid_indices))
        .sum();

    let mut selection_counts = vec![0u64; expert_count as usize];
    for record in &records {
        for expert in &record.selected_experts {
            if let Some(count) = selection_counts.get_mut(*expert as usize) {
                *count += 1;
            }
        }
    }

    let mut by_layer: BTreeMap<u32, Vec<&RoutingObservationRecord>> = BTreeMap::new();
    for record in &decode {
        if let Some(layer) = record.layer.and_then(|layer| u32::try_from(layer).ok()) {
            by_layer.entry(layer).or_default().push(record);
        }
    }

    let mut layers = Vec::new();
    let mut all_overlaps = Vec::new();
    let mut all_distances = Vec::new();
    let mut layer_unique = 0u32;
    for (layer, records) in &by_layer {
        let per_expert_bytes = layout
            .and_then(|layout| layout.layers.iter().find(|entry| entry.layer == *layer))
            .and_then(|entry| entry.per_expert_bytes);
        let mut counts = vec![0u64; expert_count as usize];
        let mut last_seen: HashMap<u32, usize> = HashMap::new();
        let mut distances = Vec::new();
        let mut overlaps = Vec::new();
        let mut previous: Option<Vec<u32>> = None;
        for (position, record) in records.iter().enumerate() {
            let mut unique = record.selected_experts.clone();
            unique.sort_unstable();
            unique.dedup();
            for expert in &unique {
                if let Some(count) = counts.get_mut(*expert as usize) {
                    *count += 1;
                }
                if let Some(previous_position) = last_seen.insert(*expert, position) {
                    distances.push((position - previous_position) as f64);
                }
            }
            if let Some(previous) = previous.as_ref() {
                let intersection = unique
                    .iter()
                    .filter(|expert| previous.binary_search(expert).is_ok())
                    .count();
                overlaps.push((intersection * 10_000 / unique.len().max(1)) as u32);
            }
            previous = Some(unique);
        }
        let unique_count = counts.iter().filter(|count| **count > 0).count() as u32;
        layer_unique += unique_count;
        let mut top: Vec<(u32, u64)> = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(expert, count)| (expert as u32, *count))
            .collect();
        top.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
        top.truncate(10);
        let overlap = mean_u32(&overlaps);
        let distance = mean_f64(&distances);
        all_overlaps.extend(overlaps);
        all_distances.extend(distances);
        layers.push(LayerRoutingReport {
            layer: *layer,
            observations: records.len() as u64,
            unique_experts: unique_count,
            selection_counts: counts,
            top_experts: top,
            consecutive_overlap_bps: overlap,
            mean_reuse_distance: distance,
            per_expert_bytes,
            unique_expert_bytes: per_expert_bytes
                .and_then(|bytes| bytes.checked_mul(u64::from(unique_count))),
            mean_active_bytes_per_token: per_expert_bytes
                .and_then(|bytes| bytes.checked_mul(u64::from(expert_used_count))),
        });
    }

    let simulations = capacities
        .iter()
        .flat_map(|capacity| {
            [
                simulate(&decode, layout, *capacity, SimulationPolicy::Lru),
                simulate(
                    &decode,
                    layout,
                    *capacity,
                    SimulationPolicy::OfflineFrequency,
                ),
            ]
        })
        .collect();

    let overlap = mean_u32(&all_overlaps);
    let previous_token_active_set_bytes = layers.iter().fold(0u64, |total, layer| {
        total.saturating_add(
            layer
                .per_expert_bytes
                .and_then(|bytes| bytes.checked_mul(u64::from(expert_used_count)))
                .unwrap_or(0),
        )
    });
    let prefetch = (previous_token_active_set_bytes > 0).then(|| {
        let expected_used = overlap
            .map(|bps| previous_token_active_set_bytes.saturating_mul(u64::from(bps)) / 10_000);
        let expected_wasted = expected_used
            .map(|used| previous_token_active_set_bytes.saturating_sub(used));
        PrefetchAssessment {
            previous_token_active_set_bytes,
            expected_reuse_bps: overlap,
            expected_used_bytes: expected_used,
            expected_wasted_bytes: expected_wasted,
            feasible: false,
            note: "The native top-k tensor is produced immediately before MUL_MAT_ID, so there is no lead time within a layer; cross-token prefetch would move the previous token's active set, whose measured reuse is only the reported overlap. The pinned API also accepts whole-tensor page prefetch, not expert-row ranges."
                .to_string(),
        }
    });

    Some(RoutingAggregate {
        schema_version: MOE_ROUTING_SCHEMA_VERSION,
        expert_count,
        expert_used_count,
        total_observations: records.len(),
        complete_decode_observations: decode.len() as u64,
        partial_observations: partial,
        invalid_indices,
        executions: records
            .iter()
            .map(|record| record.execution_id)
            .max()
            .unwrap_or(0),
        selected_expert_slots: decode
            .iter()
            .map(|record| record.selected_experts.len() as u64)
            .sum(),
        unique_experts: layer_unique,
        selection_counts,
        layers,
        consecutive_overlap_bps: overlap,
        mean_reuse_distance: mean_f64(&all_distances),
        simulations,
        prefetch,
        notes: vec![
            "Routing is copied read-only from the native ffn_moe_topk tensor after CPU node completion."
                .to_string(),
            "Only single-token decode observations are complete; prefill copies the first token row only."
                .to_string(),
            "Cache simulations are theoretical and do not move data or prove residency."
                .to_string(),
        ],
    })
}

fn simulate(
    records: &[&RoutingObservationRecord],
    layout: Option<&ExpertLayout>,
    capacity_slots: u32,
    policy: SimulationPolicy,
) -> ResidencySimulation {
    if capacity_slots == 0 || records.is_empty() {
        return ResidencySimulation {
            policy,
            capacity_slots,
            requests: 0,
            hits: 0,
            misses: 0,
            hit_rate_bps: 0,
            avoided_transfer_bytes: 0,
            note: "no complete decode observations or zero capacity".to_string(),
        };
    }
    let mut frequency: HashMap<(i32, u32), u64> = HashMap::new();
    for record in records {
        let layer = record.layer.unwrap_or(-1);
        for expert in &record.selected_experts {
            *frequency.entry((layer, *expert)).or_default() += 1;
        }
    }
    let mut fixed = Vec::new();
    if policy == SimulationPolicy::OfflineFrequency {
        let mut ranked: Vec<((i32, u32), u64)> = frequency.into_iter().collect();
        ranked.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
        fixed = ranked
            .into_iter()
            .take(capacity_slots as usize)
            .map(|(key, _)| key)
            .collect();
    }
    let fixed: std::collections::HashSet<(i32, u32)> = fixed.into_iter().collect();

    let mut lru: VecDeque<(i32, u32)> = VecDeque::new();
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut avoided = 0u64;
    for record in records {
        let layer = record.layer.unwrap_or(-1);
        for expert in &record.selected_experts {
            let key = (layer, *expert);
            let hit = if policy == SimulationPolicy::OfflineFrequency {
                fixed.contains(&key)
            } else {
                if let Some(position) = lru.iter().position(|entry| *entry == key) {
                    lru.remove(position);
                    true
                } else {
                    false
                }
            };
            if hit {
                hits += 1;
                if let Some(bytes) = layout
                    .and_then(|layout| {
                        u32::try_from(layer).ok().and_then(|layer| {
                            layout.layers.iter().find(|entry| entry.layer == layer)
                        })
                    })
                    .and_then(|entry| entry.per_expert_bytes)
                {
                    avoided = avoided.saturating_add(bytes);
                }
            } else {
                misses += 1;
            }
            if policy == SimulationPolicy::Lru {
                if let Some(position) = lru.iter().position(|entry| *entry == key) {
                    lru.remove(position);
                }
                lru.push_back(key);
                while lru.len() > capacity_slots as usize {
                    lru.pop_front();
                }
            }
        }
    }
    let requests = hits + misses;
    ResidencySimulation {
        policy,
        capacity_slots,
        requests,
        hits,
        misses,
        hit_rate_bps: hits
            .checked_mul(10_000)
            .and_then(|value| value.checked_div(requests))
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(0),
        avoided_transfer_bytes: avoided,
        note: match policy {
            SimulationPolicy::Lru => {
                "online LRU over (layer, expert) slots; theoretical transfer avoidance".to_string()
            }
            SimulationPolicy::OfflineFrequency => {
                "offline top-frequency set; upper bound that uses the whole trace".to_string()
            }
        },
    }
}

fn mean_u32(values: &[u32]) -> Option<u32> {
    if values.is_empty() {
        None
    } else {
        let sum: u64 = values.iter().map(|value| u64::from(*value)).sum();
        u32::try_from(sum / values.len() as u64).ok()
    }
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ramforge_core::{GgmlType, MetadataValue, TensorDescriptor};

    use super::*;

    fn model(names: &[(&str, u64)]) -> GgufModel {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            MetadataValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.expert_count".to_string(),
            MetadataValue::UInt64(4),
        );
        metadata.insert(
            "deepseek2.expert_used_count".to_string(),
            MetadataValue::UInt64(2),
        );
        GgufModel {
            path: std::path::PathBuf::from("/m"),
            file_size: names.iter().map(|(_, bytes)| *bytes).sum(),
            version: 3,
            metadata,
            tensors: names
                .iter()
                .map(|(name, bytes)| TensorDescriptor {
                    name: (*name).to_string(),
                    dimensions: vec![1],
                    ggml_type: GgmlType::Q4_K,
                    offset: 0,
                    file_offset: 0,
                    byte_length: Some(*bytes),
                    num_elements: 1,
                })
                .collect(),
            alignment: 32,
            data_start_offset: 0,
        }
    }

    fn observation(
        execution: u64,
        sequence: u64,
        layer: i32,
        experts: &[u32],
        complete: bool,
    ) -> ExpertRoutingObservation {
        ExpertRoutingObservation {
            execution_id: execution,
            sequence,
            layer: Some(layer),
            node_name: format!("ffn_moe_topk-{layer}"),
            expert_count: 4,
            expert_used_count: 2,
            topk_bytes: 8,
            selected_experts: experts.to_vec(),
            invalid_indices: 0,
            token_index: 0,
            tokens_in_graph: Some(if complete { 1 } else { 2 }),
            complete_for_token: complete,
        }
    }

    #[test]
    fn layout_separates_expert_kinds_and_sizes() {
        let layout = expert_layout(&model(&[
            ("blk.0.ffn_gate_exps.weight", 400),
            ("blk.0.ffn_up_exps.weight", 400),
            ("blk.0.ffn_down_shexp.weight", 100),
            ("blk.0.attn_q.weight", 50),
            ("blk.1.ffn_gate_exps.weight", 800),
        ]));
        assert_eq!(layout.layers.len(), 2);
        let layer0 = &layout.layers[0];
        assert_eq!(layer0.routed_tensor_bytes, 800);
        assert_eq!(layer0.per_expert_bytes, Some(200));
        assert_eq!(layer0.shared_expert_bytes, 100);
        assert_eq!(layer0.other_tensor_bytes, 50);
        assert_eq!(layout.layers[1].per_expert_bytes, Some(200));
    }

    #[test]
    fn aggregate_measures_overlap_and_lru_hits() {
        let observations = vec![
            observation(1, 0, 1, &[0, 1], false),
            observation(2, 0, 1, &[0, 1], true),
            observation(2, 1, 1, &[1, 2], true),
            observation(2, 2, 1, &[0, 1], true),
        ];
        let layout = expert_layout(&model(&[("blk.1.ffn_gate_exps.weight", 400)]));
        let aggregate = aggregate_routing(&observations, Some(&layout), &[1, 2, 4]).expect("agg");
        assert_eq!(aggregate.complete_decode_observations, 3);
        assert_eq!(aggregate.partial_observations, 1);
        // 0/1 -> 1/2 = 50%; 1/2 -> 0/1 = 50%.
        assert_eq!(aggregate.layers[0].consecutive_overlap_bps, Some(5000));
        let lru1 = aggregate
            .simulations
            .iter()
            .find(|sim| sim.capacity_slots == 1 && sim.policy == SimulationPolicy::Lru)
            .unwrap();
        // Slot 1: 0 miss, 1 miss/evict 0, 1 hit, 2 miss/evict 1, 0 miss, 1 miss.
        assert_eq!(lru1.requests, 6);
        assert_eq!(lru1.hits, 1);
    }

    #[test]
    fn expert_byte_ranges_are_contiguous_and_bounded() {
        let model = model(&[("blk.3.ffn_gate_exps.weight", 4 * 100)]);
        let mappings = vec![crate::TensorMappingInfo {
            name: "blk.3.ffn_gate_exps.weight".to_string(),
            file_backed: true,
            mapping_index: Some(0),
            mapping_size: 1000,
            offset: 200,
            size: 400,
            page_size: 64,
            page_start: 192,
            page_end: 640,
            mapping_id: Some(7),
            mapping_generation: Some(3),
            file_path: Some("/tmp/model.gguf".to_string()),
            file_size: Some(1000),
        }];
        let ranges = expert_byte_ranges(&model, &mappings, 3, &[0, 3]).expect("ranges");
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].file_offset, 200);
        assert_eq!(ranges[1].file_offset, 500);
        assert_eq!(ranges[0].bytes, 100);
        assert_eq!(ranges[0].page_start, 192);
        assert_eq!(ranges[0].page_end, 320);
        assert!(expert_byte_ranges(&model, &mappings, 3, &[4]).is_err());
    }

    #[test]
    fn incomplete_and_invalid_observations_do_not_enter_simulation() {
        let mut invalid = observation(2, 0, 1, &[9, 1], true);
        invalid.invalid_indices = 1;
        let observations = vec![observation(1, 0, 1, &[0, 1], false), invalid];
        let aggregate = aggregate_routing(&observations, None, &[4]).expect("agg");
        assert_eq!(aggregate.complete_decode_observations, 0);
        assert_eq!(aggregate.invalid_indices, 1);
        assert!(aggregate.simulations.iter().all(|sim| sim.requests == 0));
    }
}
