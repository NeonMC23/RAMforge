//! Bounded, in-memory execution profiles derived from graph manifests.
//!
//! Profiles contain stable tensor names and observation counters only. Native
//! node identities are never persisted.

use std::collections::BTreeMap;

use forge_core::TraceEvent;

use crate::graph_manifest::GraphManifest;
use crate::residency_engine::{ModelGeneration, ModelId};

pub const EXECUTION_PROFILE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessPattern {
    pub first_order: usize,
    pub last_order: usize,
    pub observed_consumers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionProfile {
    pub version: u32,
    pub model_id: ModelId,
    pub generation: ModelGeneration,
    pub complete_observations: u64,
    pub incomplete_observations: u64,
    pub last_node_count: usize,
    pub last_tensor_count: usize,
    pub access: BTreeMap<String, AccessPattern>,
    pub invalidated_reason: Option<String>,
}

impl ExecutionProfile {
    pub fn new(model_id: ModelId, generation: ModelGeneration) -> Self {
        Self {
            version: EXECUTION_PROFILE_VERSION,
            model_id,
            generation,
            complete_observations: 0,
            incomplete_observations: 0,
            last_node_count: 0,
            last_tensor_count: 0,
            access: BTreeMap::new(),
            invalidated_reason: None,
        }
    }

    /// Update from bounded trace events. A capture can contain several graph
    /// executions; each execution is analyzed separately so transient native
    /// node IDs are never merged across executions.
    pub fn update_from_trace(&mut self, events: &[TraceEvent], trace_complete: bool) {
        let mut execution_ids = std::collections::BTreeSet::new();
        for event in events {
            execution_ids.insert(event.execution_id);
        }
        if execution_ids.is_empty() {
            self.incomplete_observations += 1;
            return;
        }
        for execution_id in execution_ids {
            let execution_events: Vec<TraceEvent> = events
                .iter()
                .filter(|event| event.execution_id == execution_id)
                .cloned()
                .collect();
            let manifest = GraphManifest::from_trace(&execution_events);
            self.update(&manifest, trace_complete);
        }
    }

    /// Update from a manifest. Incomplete traces are counted but do not
    /// overwrite a previously complete access map.
    pub fn update(&mut self, manifest: &GraphManifest, trace_complete: bool) {
        if manifest.execution_id.is_none() {
            self.incomplete_observations += 1;
            return;
        }
        let usable = trace_complete
            && manifest.edges_known
            && manifest.completion_proven
            && manifest.dependency_cycles() == 0
            && !manifest.intra_decode_safe;
        self.last_node_count = manifest.nodes.len();
        self.last_tensor_count = manifest.tensor_to_nodes.len();
        if !usable {
            self.incomplete_observations += 1;
            return;
        }
        self.complete_observations += 1;
        let mut access = BTreeMap::new();
        for (tensor, consumers) in &manifest.tensor_to_nodes {
            let first = consumers.iter().copied().min().unwrap_or(0);
            let last = consumers.iter().copied().max().unwrap_or(0);
            access.insert(
                tensor.clone(),
                AccessPattern {
                    first_order: first,
                    last_order: last,
                    observed_consumers: consumers.len(),
                },
            );
        }
        self.access = access;
        self.invalidated_reason = None;
    }

    pub fn invalidate(&mut self, reason: impl Into<String>) {
        self.access.clear();
        self.last_node_count = 0;
        self.last_tensor_count = 0;
        self.invalidated_reason = Some(reason.into());
    }

    pub fn is_valid_for(&self, model_id: &ModelId, generation: ModelGeneration) -> bool {
        self.model_id == *model_id
            && self.generation == generation
            && self.complete_observations > 0
            && self.invalidated_reason.is_none()
    }

    pub fn last_order(&self, tensor_name: &str) -> Option<usize> {
        self.access
            .get(tensor_name)
            .map(|pattern| pattern.last_order)
    }

    pub fn is_complete(&self) -> bool {
        self.complete_observations > 0 && self.invalidated_reason.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_manifest::{GraphManifest, LastUse};
    use forge_core::{TraceCompletion, TracePhase};
    use std::collections::BTreeMap;

    fn manifest() -> GraphManifest {
        GraphManifest {
            execution_id: Some(1),
            nodes: Vec::new(),
            tensor_to_nodes: BTreeMap::from([
                ("blk.0.weight".to_string(), vec![0, 3]),
                ("output.weight".to_string(), vec![5]),
            ]),
            last_use: BTreeMap::<String, LastUse>::new(),
            edges_known: true,
            completion_proven: true,
            intra_decode_safe: false,
        }
    }

    #[test]
    fn complete_profile_records_names_and_orders() {
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.update(&manifest(), true);
        assert_eq!(profile.complete_observations, 1);
        assert_eq!(profile.last_order("blk.0.weight"), Some(3));
        assert_eq!(profile.last_order("output.weight"), Some(5));
        assert!(profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(1)));
        assert_eq!(profile.access.len(), 2);
    }

    #[test]
    fn incomplete_trace_does_not_overwrite_profile() {
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.update(&manifest(), true);
        let mut incomplete = manifest();
        incomplete.edges_known = false;
        profile.update(&incomplete, false);
        assert_eq!(profile.complete_observations, 1);
        assert_eq!(profile.incomplete_observations, 1);
        assert_eq!(profile.last_order("blk.0.weight"), Some(3));
    }

    #[test]
    fn multi_execution_trace_is_not_merged_by_native_node_id() {
        let event = |execution_id, sequence, name: &str| TraceEvent {
            execution_id,
            sequence,
            node_id: 1,
            node_name: Some("node".to_string()),
            op_name: None,
            bytes: 4096,
            dependencies: vec![name.to_string()],
            dependency_ids: vec![2],
            phase: TracePhase::Requested,
            completion: TraceCompletion::NotCompleted,
        };
        let completed = |execution_id, sequence| TraceEvent {
            execution_id,
            sequence,
            node_id: 1,
            node_name: Some("node".to_string()),
            op_name: None,
            bytes: 4096,
            dependencies: Vec::new(),
            dependency_ids: Vec::new(),
            phase: TracePhase::CompletedSynchronized,
            completion: TraceCompletion::BackendSynchronized,
        };
        let events = vec![
            event(1, 0, "blk.0.a"),
            completed(1, 1),
            event(2, 0, "blk.0.b"),
            completed(2, 1),
        ];
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.update_from_trace(&events, true);
        assert_eq!(profile.complete_observations, 2);
        assert_eq!(profile.last_order("blk.0.a"), None);
        assert_eq!(profile.last_order("blk.0.b"), Some(0));
        assert!(profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(1)));
    }

    #[test]
    fn invalidation_and_generation_mismatch_disable_profile() {
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.update(&manifest(), true);
        assert!(profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(1)));
        assert!(!profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(2)));
        profile.invalidate("reload");
        assert!(!profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(1)));
    }
}
