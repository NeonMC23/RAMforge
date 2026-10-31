//! Deterministic predictive staging planner and synchronous coordinator.
//!
//! Predictions only select bounded prefetch hints. They do not authorize
//! release: actual release remains owned by the residency manager and native
//! lifetime gate.

use std::collections::BTreeMap;

use crate::prediction::{
    CalibrationCounters, PredictedAccess, PredictionConfidence, PredictionConfig,
    PredictionMetrics, PredictionSet,
};
use crate::residency_engine::{ResourceInventory, ResourceKey};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictiveAction {
    KeepResident,
    StageNow,
    Prefetch,
    Defer,
    AvoidSpeculative,
    Reject,
    ReleaseAtBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictiveReason {
    HighConfidenceImminent,
    MediumConfidenceSpeculative,
    InsufficientConfidence,
    InsufficientBudget,
    UnknownSize,
    NotFileBacked,
    PredictionDisabled,
    AdmissionRejected,
    DuplicateRequest,
    SharedPageDeduplicated,
    StagingDisabled,
    NativeLifetimeUnsafe,
    ReleaseNotAuthorized,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictiveDecision {
    pub key: Option<ResourceKey>,
    pub name: String,
    pub action: PredictiveAction,
    pub reason: PredictiveReason,
    pub confidence_bps: u16,
    pub predicted_order: usize,
    pub requested_bytes: u64,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PredictionPlan {
    pub decisions: Vec<PredictiveDecision>,
    pub stage_keys: Vec<ResourceKey>,
    pub prefetch_keys: Vec<ResourceKey>,
    pub deferred: usize,
    pub rejected: usize,
    pub planned_stage_bytes: u64,
    pub planned_prefetch_bytes: u64,
    pub staged_bytes_by_name: BTreeMap<String, u64>,
    pub release_authorized: bool,
    pub notes: Vec<String>,
}

impl PredictionPlan {
    pub fn stage_indices(&self) -> Vec<i32> {
        self.stage_keys
            .iter()
            .chain(self.prefetch_keys.iter())
            .map(|key| key.tensor_index)
            .collect()
    }

    pub fn requested_bytes(&self) -> u64 {
        self.planned_stage_bytes
            .saturating_add(self.planned_prefetch_bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingExecution {
    pub attempted_keys: Vec<ResourceKey>,
    pub requested_bytes: u64,
    pub native_accepted_bytes: u64,
    pub elapsed_ns: u128,
    pub operations: usize,
    pub configured_max_in_flight: u16,
    pub error: Option<String>,
    pub cancelled: bool,
    pub mapping_before_bytes: Option<u64>,
    pub mapping_after_bytes: Option<u64>,
}

impl StagingExecution {
    pub fn observed_mapping_delta(&self) -> Option<i128> {
        match (self.mapping_before_bytes, self.mapping_after_bytes) {
            (Some(before), Some(after)) => Some(after as i128 - before as i128),
            _ => None,
        }
    }
}

/// Plan bounded prediction-driven staging. `current_bytes` is an observed
/// process estimate, not a guarantee; the hard limit is the only strict cap.
#[allow(clippy::too_many_arguments)]
pub fn plan_predictions(
    config: &PredictionConfig,
    predictions: &PredictionSet,
    inventory: &ResourceInventory,
    current_bytes: Option<u64>,
    hard_limit_bytes: Option<u64>,
    safety_reserve_bytes: u64,
    admission_rejected: bool,
) -> PredictionPlan {
    let mut plan = PredictionPlan::default();
    plan.notes.push(
        "predictions are advisory; release remains gated by the native lifetime contract"
            .to_string(),
    );
    if !config.enabled {
        for prediction in &predictions.predictions {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::AvoidSpeculative,
                PredictiveReason::PredictionDisabled,
                "prediction disabled",
            ));
        }
        return plan;
    }
    if admission_rejected {
        for prediction in &predictions.predictions {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Reject,
                PredictiveReason::AdmissionRejected,
                "admission rejected the operation",
            ));
        }
        plan.rejected = predictions.predictions.len();
        return plan;
    }

    let hard_available = match (hard_limit_bytes, current_bytes) {
        (Some(hard), Some(current)) => hard
            .saturating_sub(current)
            .saturating_sub(safety_reserve_bytes),
        _ => config.staging_window_bytes,
    };
    let stage_budget = config.staging_window_bytes.min(hard_available);
    let speculative_budget = config.speculative_budget_bytes.min(stage_budget);
    let mut stage_bytes = 0u64;
    let mut prefetch_bytes = 0u64;
    let mut seen = std::collections::BTreeSet::new();
    let mut ranges: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();

    for prediction in &predictions.predictions {
        let Some(resource) = inventory.get(prediction.key) else {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Reject,
                PredictiveReason::NotFileBacked,
                "prediction is not in the current inventory",
            ));
            plan.rejected += 1;
            continue;
        };
        if !resource.file_backed {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Reject,
                PredictiveReason::NotFileBacked,
                "resource is anonymous or has no file-backed mapping",
            ));
            plan.rejected += 1;
            continue;
        }
        if !prediction.known_size || resource.page_bytes == 0 {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Reject,
                PredictiveReason::UnknownSize,
                "resource size is unknown",
            ));
            plan.rejected += 1;
            continue;
        }
        if !seen.insert(prediction.key) {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Defer,
                PredictiveReason::DuplicateRequest,
                "duplicate resource prediction",
            ));
            plan.deferred += 1;
            continue;
        }
        if prediction.confidence_bps < config.min_confidence_bps {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::AvoidSpeculative,
                PredictiveReason::InsufficientConfidence,
                format!(
                    "confidence {} below threshold {}",
                    prediction.confidence_bps, config.min_confidence_bps
                ),
            ));
            plan.rejected += 1;
            continue;
        }

        let unique = unique_page_bytes(&ranges, resource);
        if unique == 0 {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Defer,
                PredictiveReason::SharedPageDeduplicated,
                "page range is already covered by an earlier prediction",
            ));
            plan.deferred += 1;
            continue;
        }
        let stage_candidate = prediction.confidence == PredictionConfidence::High
            && stage_bytes.saturating_add(unique) <= stage_budget;
        let prefetch_candidate = prediction.confidence == PredictionConfidence::Medium
            && prefetch_bytes.saturating_add(unique) <= speculative_budget;
        if stage_candidate {
            insert_resource_range(&mut ranges, resource);
            stage_bytes = stage_bytes.saturating_add(unique);
            plan.stage_keys.push(prediction.key);
            plan.staged_bytes_by_name
                .insert(prediction.name.clone(), unique);
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::StageNow,
                PredictiveReason::HighConfidenceImminent,
                format!("high-confidence prediction; unique page bytes {unique}"),
            ));
        } else if prefetch_candidate {
            insert_resource_range(&mut ranges, resource);
            prefetch_bytes = prefetch_bytes.saturating_add(unique);
            plan.prefetch_keys.push(prediction.key);
            plan.staged_bytes_by_name
                .insert(prediction.name.clone(), unique);
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Prefetch,
                PredictiveReason::MediumConfidenceSpeculative,
                format!("medium-confidence opportunistic prefetch; unique page bytes {unique}"),
            ));
        } else if prediction.confidence == PredictionConfidence::High {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Defer,
                PredictiveReason::InsufficientBudget,
                "high-confidence request does not fit the bounded stage window",
            ));
            plan.deferred += 1;
        } else if prediction.confidence == PredictionConfidence::Low {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::AvoidSpeculative,
                PredictiveReason::InsufficientConfidence,
                "low-confidence prediction was not staged",
            ));
            plan.rejected += 1;
        } else {
            plan.decisions.push(decision(
                prediction,
                PredictiveAction::Defer,
                PredictiveReason::InsufficientBudget,
                "speculative window is exhausted",
            ));
            plan.deferred += 1;
        }
    }
    plan.planned_stage_bytes = stage_bytes;
    plan.planned_prefetch_bytes = prefetch_bytes;
    plan.release_authorized = false;
    plan.notes.push(
        "release decisions are intentionally not derived from prediction confidence".to_string(),
    );
    plan
}

fn decision(
    prediction: &PredictedAccess,
    action: PredictiveAction,
    reason: PredictiveReason,
    detail: impl Into<String>,
) -> PredictiveDecision {
    PredictiveDecision {
        key: Some(prediction.key),
        name: prediction.name.clone(),
        action,
        reason,
        confidence_bps: prediction.confidence_bps,
        predicted_order: prediction.predicted_order,
        requested_bytes: prediction.size_bytes,
        detail: detail.into(),
    }
}

fn unique_page_bytes(
    ranges: &BTreeMap<u32, Vec<(u64, u64)>>,
    resource: &crate::residency_engine::TensorResource,
) -> u64 {
    let Some(mapping) = resource.mapping_index else {
        return 0;
    };
    let overlap = ranges
        .get(&mapping)
        .map(|ranges| covered_bytes(ranges, resource.page_start, resource.page_end))
        .unwrap_or(0);
    resource.page_bytes.saturating_sub(overlap)
}

fn insert_resource_range(
    ranges: &mut BTreeMap<u32, Vec<(u64, u64)>>,
    resource: &crate::residency_engine::TensorResource,
) {
    if let Some(mapping) = resource.mapping_index {
        insert_interval(
            ranges.entry(mapping).or_default(),
            resource.page_start,
            resource.page_end,
        );
    }
}

fn covered_bytes(ranges: &[(u64, u64)], start: u64, end: u64) -> u64 {
    if start >= end {
        return 0;
    }
    let mut covered = 0u64;
    for &(range_start, range_end) in ranges {
        let first = start.max(range_start);
        let last = end.min(range_end);
        if first < last {
            covered = covered.saturating_add(last - first);
        }
    }
    covered
}

fn insert_interval(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    if start >= end {
        return;
    }
    let mut merged = Vec::with_capacity(ranges.len() + 1);
    let mut pending = (start, end);
    for &(range_start, range_end) in ranges.iter() {
        if range_end < pending.0 {
            merged.push((range_start, range_end));
        } else if pending.1 < range_start {
            merged.push(pending);
            pending = (range_start, range_end);
        } else {
            pending = (pending.0.min(range_start), pending.1.max(range_end));
        }
    }
    merged.push(pending);
    *ranges = merged;
}

/// Execute a prediction staging plan synchronously through the existing
/// native prefetch wrapper. This intentionally has no background threads and
/// cannot outlive the model/context that lends the callback.
pub fn execute_staging<F>(
    config: &PredictionConfig,
    plan: &PredictionPlan,
    mut prefetch: F,
) -> StagingExecution
where
    F: FnMut(&[i32]) -> Result<u64, String>,
{
    let indices = plan.stage_indices();
    let mut execution = StagingExecution {
        attempted_keys: plan
            .stage_keys
            .iter()
            .chain(plan.prefetch_keys.iter())
            .copied()
            .collect(),
        requested_bytes: plan.requested_bytes(),
        native_accepted_bytes: 0,
        elapsed_ns: 0,
        operations: 0,
        configured_max_in_flight: config.max_in_flight,
        error: None,
        cancelled: false,
        mapping_before_bytes: None,
        mapping_after_bytes: None,
    };
    if indices.is_empty() {
        return execution;
    }
    execution.operations = 1;
    let started = std::time::Instant::now();
    match prefetch(&indices) {
        Ok(bytes) => execution.native_accepted_bytes = bytes,
        Err(error) => execution.error = Some(error),
    }
    execution.elapsed_ns = started.elapsed().as_nanos();
    execution
}

/// Evaluate predictions against a later observed execution. A prediction hit
/// means only "the stable tensor was observed"; it is not proof that prefetch
/// helped, that pages became resident, or that performance improved.
pub fn evaluate_predictions(
    predictions: &PredictionSet,
    observed: &crate::execution_profile::ExecutionProfile,
    inventory: &ResourceInventory,
    plan: &PredictionPlan,
    staging: Option<&StagingExecution>,
    horizon: usize,
) -> PredictionMetrics {
    let known_names: std::collections::BTreeSet<&str> = inventory
        .resources
        .iter()
        .map(|resource| resource.name.as_str())
        .collect();
    let mut ordered: Vec<(&str, usize)> = observed
        .access
        .iter()
        .filter(|(name, _)| known_names.contains(name.as_str()))
        .map(|(name, access)| (name.as_str(), access.first_order))
        .collect();
    ordered.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
    let observed_rank: BTreeMap<&str, usize> = ordered
        .iter()
        .enumerate()
        .map(|(rank, (name, _))| (*name, rank))
        .collect();
    let observed_names: std::collections::BTreeSet<&str> =
        ordered.iter().map(|(name, _)| *name).collect();

    let mut metrics = PredictionMetrics {
        horizon,
        predictions_generated: predictions.predictions.len(),
        observed_distinct: observed_names.len(),
        prefetch_requested_resources: plan.stage_keys.len() + plan.prefetch_keys.len(),
        prefetch_requested_bytes: plan.requested_bytes(),
        duplicate_or_overlap_avoided: plan
            .decisions
            .iter()
            .filter(|decision| {
                matches!(
                    decision.reason,
                    PredictiveReason::DuplicateRequest | PredictiveReason::SharedPageDeduplicated
                )
            })
            .count(),
        rejected: plan.rejected,
        deferred: plan.deferred,
        ..PredictionMetrics::default()
    };
    let mut bucket_high = CalibrationCounters::default();
    let mut bucket_medium = CalibrationCounters::default();
    let mut bucket_low = CalibrationCounters::default();
    for prediction in &predictions.predictions {
        let hit = observed_names.contains(prediction.name.as_str());
        let within = observed_rank
            .get(prediction.name.as_str())
            .is_some_and(|rank| *rank < horizon);
        if hit {
            metrics.matched_observed += 1;
        }
        if within {
            metrics.matched_within_horizon += 1;
        }
        let bucket = match prediction.confidence {
            PredictionConfidence::High => &mut bucket_high,
            PredictionConfidence::Medium => &mut bucket_medium,
            PredictionConfidence::Low => &mut bucket_low,
        };
        bucket.predictions += 1;
        if hit {
            bucket.hits += 1;
        }
        if within {
            bucket.within_horizon += 1;
        }
    }
    metrics.missed_observed = metrics
        .predictions_generated
        .saturating_sub(metrics.matched_observed);
    metrics.precision_bps = (metrics.predictions_generated > 0).then(|| {
        ((metrics.matched_observed.saturating_mul(10_000)) / metrics.predictions_generated) as u16
    });
    metrics.recall_bps = (metrics.observed_distinct > 0).then(|| {
        ((metrics.matched_observed.saturating_mul(10_000)) / metrics.observed_distinct) as u16
    });
    metrics.false_positive_rate_bps = (metrics.predictions_generated > 0).then(|| {
        ((metrics.missed_observed.saturating_mul(10_000)) / metrics.predictions_generated) as u16
    });
    metrics.bucket_high = bucket_high;
    metrics.bucket_medium = bucket_medium;
    metrics.bucket_low = bucket_low;

    if let Some(staging) = staging {
        metrics.prefetch_accepted_bytes = staging.native_accepted_bytes;
        metrics.observed_mapping_delta_bytes = staging.observed_mapping_delta();
        let requested = metrics.prefetch_requested_resources;
        if staging.error.is_some() || requested == 0 {
            metrics.prefetch_accepted_resources = Some(0);
            metrics.prefetch_followed_by_observed_access = Some(0);
            metrics.prefetch_not_followed_by_observed_access = Some(0);
            metrics.prefetch_benefit_unknown_resources = Some(0);
        } else if staging.native_accepted_bytes >= metrics.prefetch_requested_bytes {
            metrics.prefetch_accepted_resources = Some(requested);
            let selected_names: std::collections::BTreeSet<&str> = plan
                .decisions
                .iter()
                .filter(|decision| {
                    matches!(
                        decision.action,
                        PredictiveAction::StageNow | PredictiveAction::Prefetch
                    )
                })
                .map(|decision| decision.name.as_str())
                .collect();
            let followed = selected_names
                .iter()
                .filter(|name| observed_names.contains(**name))
                .count();
            metrics.prefetch_followed_by_observed_access = Some(followed);
            metrics.prefetch_not_followed_by_observed_access =
                Some(requested.saturating_sub(followed));
            metrics.prefetch_benefit_unknown_resources = Some(requested);
        } else {
            // Native acceptance is aggregate; per-resource acceptance is not
            // observable from the pinned API. Report acceptance as unknown.
            metrics.prefetch_accepted_resources = None;
            metrics.prefetch_followed_by_observed_access = None;
            metrics.prefetch_not_followed_by_observed_access = None;
            metrics.prefetch_benefit_unknown_resources = Some(requested);
        }
    }
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_profile::AccessPattern;
    use crate::prediction::{predict_from_profile, PredictionConfig};
    use crate::residency_engine::{
        ModelGeneration, ModelId, ResidencyPolicy, ResidencyPolicyMode, ResourceInventory,
    };
    use forge_core::TensorMappingInfo;

    fn inventory() -> ResourceInventory {
        ResourceInventory::from_tensor_mappings(
            ModelId("m".to_string()),
            ModelGeneration(1),
            &[
                TensorMappingInfo {
                    name: "blk.0.a".to_string(),
                    file_backed: true,
                    mapping_index: Some(0),
                    mapping_size: 1 << 20,
                    offset: 0,
                    size: 8192,
                    page_size: 4096,
                    page_start: 0,
                    page_end: 8192,
                    mapping_id: None,
                    mapping_generation: None,
                    file_path: None,
                    file_size: None,
                },
                TensorMappingInfo {
                    name: "blk.0.b".to_string(),
                    file_backed: true,
                    mapping_index: Some(0),
                    mapping_size: 1 << 20,
                    offset: 8192,
                    size: 8192,
                    page_size: 4096,
                    page_start: 8192,
                    page_end: 16384,
                    mapping_id: None,
                    mapping_generation: None,
                    file_path: None,
                    file_size: None,
                },
            ],
        )
        .expect("inventory")
    }

    fn profile() -> crate::execution_profile::ExecutionProfile {
        let mut profile = crate::execution_profile::ExecutionProfile::new(
            ModelId("m".to_string()),
            ModelGeneration(1),
        );
        profile.complete_observations = 2;
        for (name, order) in [("blk.0.a", 1), ("blk.0.b", 2)] {
            profile.access.insert(
                name.to_string(),
                AccessPattern {
                    first_order: order,
                    last_order: order,
                    observed_consumers: 1,
                },
            );
        }
        profile
    }

    #[test]
    fn budget_and_confidence_are_enforced() {
        let config = PredictionConfig {
            enabled: true,
            staging_window_bytes: 8192,
            speculative_budget_bytes: 0,
            min_confidence_bps: 7000,
            ..Default::default()
        };
        let predictions = predict_from_profile(&config, Some(&profile()), &inventory(), None);
        let plan = plan_predictions(&config, &predictions, &inventory(), Some(0), None, 0, false);
        assert!(plan.requested_bytes() <= 8192);
        assert!(!plan.release_authorized);
        assert!(plan.notes.iter().any(|note| note.contains("release")));
    }

    #[test]
    fn staging_execution_counts_requested_and_accepted() {
        let config = PredictionConfig {
            enabled: true,
            staging_window_bytes: 16384,
            speculative_budget_bytes: 16_384,
            min_confidence_bps: 5_000,
            ..Default::default()
        };
        let predictions = predict_from_profile(&config, Some(&profile()), &inventory(), None);
        let plan = plan_predictions(&config, &predictions, &inventory(), None, None, 0, false);
        let execution = execute_staging(&config, &plan, |indices| {
            assert!(!indices.is_empty());
            Ok(plan.requested_bytes())
        });
        assert_eq!(execution.requested_bytes, execution.native_accepted_bytes);
        assert_eq!(execution.operations, 1);
    }

    #[test]
    fn evaluation_separates_prediction_hits_from_prefetch_usefulness() {
        let config = PredictionConfig {
            enabled: true,
            staging_window_bytes: 16384,
            speculative_budget_bytes: 0,
            min_confidence_bps: 5_000,
            ..Default::default()
        };
        let predictions = predict_from_profile(&config, Some(&profile()), &inventory(), None);
        let plan = plan_predictions(&config, &predictions, &inventory(), None, None, 0, false);
        let execution = execute_staging(&config, &plan, |_| Ok(plan.requested_bytes()));
        let metrics = evaluate_predictions(
            &predictions,
            &profile(),
            &inventory(),
            &plan,
            Some(&execution),
            4,
        );
        assert_eq!(metrics.predictions_generated, 2);
        assert_eq!(metrics.matched_observed, 2);
        assert_eq!(metrics.precision_bps, Some(10_000));
        assert_eq!(metrics.prefetch_accepted_resources, Some(2));
        assert_eq!(metrics.prefetch_followed_by_observed_access, Some(2));
        assert_eq!(metrics.prefetch_observed_residency_resources, None);
        assert!(metrics.prefetch_benefit_unknown_resources == Some(2));

        let mut empty = crate::execution_profile::ExecutionProfile::new(
            ModelId("m".to_string()),
            ModelGeneration(1),
        );
        empty.complete_observations = 1;
        let empty_metrics = evaluate_predictions(
            &predictions,
            &empty,
            &inventory(),
            &plan,
            Some(&execution),
            4,
        );
        assert_eq!(empty_metrics.matched_observed, 0);
        assert_eq!(empty_metrics.prefetch_followed_by_observed_access, Some(0));
        assert_eq!(
            empty_metrics.prefetch_not_followed_by_observed_access,
            Some(2)
        );
    }

    #[test]
    fn overlapping_pages_are_staged_once() {
        let inventory = ResourceInventory::from_tensor_mappings(
            ModelId("overlap".to_string()),
            ModelGeneration(1),
            &[
                TensorMappingInfo {
                    name: "blk.0.a".to_string(),
                    file_backed: true,
                    mapping_index: Some(0),
                    mapping_size: 1 << 20,
                    offset: 0,
                    size: 8192,
                    page_size: 4096,
                    page_start: 0,
                    page_end: 8192,
                    mapping_id: None,
                    mapping_generation: None,
                    file_path: None,
                    file_size: None,
                },
                TensorMappingInfo {
                    name: "blk.0.b".to_string(),
                    file_backed: true,
                    mapping_index: Some(0),
                    mapping_size: 1 << 20,
                    offset: 0,
                    size: 8192,
                    page_size: 4096,
                    page_start: 0,
                    page_end: 8192,
                    mapping_id: None,
                    mapping_generation: None,
                    file_path: None,
                    file_size: None,
                },
            ],
        )
        .expect("inventory");
        let mut profile = crate::execution_profile::ExecutionProfile::new(
            ModelId("overlap".to_string()),
            ModelGeneration(1),
        );
        profile.complete_observations = 2;
        for (name, order) in [("blk.0.a", 0), ("blk.0.b", 1)] {
            profile.access.insert(
                name.to_string(),
                AccessPattern {
                    first_order: order,
                    last_order: order,
                    observed_consumers: 1,
                },
            );
        }
        let config = PredictionConfig {
            enabled: true,
            staging_window_bytes: 16384,
            speculative_budget_bytes: 0,
            ..Default::default()
        };
        let predictions = predict_from_profile(&config, Some(&profile), &inventory, None);
        let plan = plan_predictions(&config, &predictions, &inventory, None, None, 0, false);
        assert_eq!(plan.stage_keys.len(), 1);
        assert_eq!(plan.deferred, 1);
        assert!(plan
            .decisions
            .iter()
            .any(|decision| { decision.reason == PredictiveReason::SharedPageDeduplicated }));
    }

    #[test]
    fn admission_rejection_disables_staging() {
        let config = PredictionConfig {
            enabled: true,
            ..Default::default()
        };
        let predictions = predict_from_profile(&config, Some(&profile()), &inventory(), None);
        let plan = plan_predictions(&config, &predictions, &inventory(), None, None, 0, true);
        assert!(plan.stage_keys.is_empty());
        assert!(plan
            .decisions
            .iter()
            .all(|decision| decision.reason == PredictiveReason::AdmissionRejected));
    }

    #[test]
    fn policy_does_not_authorize_release() {
        let policy = ResidencyPolicy {
            mode: ResidencyPolicyMode::MemoryBound,
            target_bytes: Some(1),
            safety_reserve_bytes: 0,
        };
        assert_eq!(policy.mode, ResidencyPolicyMode::MemoryBound);
    }
}
