//! Deterministic staged-layer planning and budget states.

use serde::{Deserialize, Serialize};

use crate::host_staging::HostStagingReport;

pub const STAGING_PLAN_SCHEMA_VERSION: u32 = 8;
pub const DEFAULT_STAGING_SAFETY_MARGIN_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPolicy {
    #[default]
    Ascending,
    SmallestFirst,
    LargestFirst,
    ResidentFirst,
}

/// Deterministic candidate ordering for automatic staging.
///
/// `size` is the native staged-layer byte size when known. `resident` reports
/// whether the layer is already staged in the current context. Unknown sizes
/// sort after known sizes for smallest-first and before known sizes for
/// largest-first only when the value is actually unknown; this function never
/// invents a size.
pub fn order_candidate_layers<F, G>(
    policy: SelectionPolicy,
    layers: Vec<u32>,
    mut size: F,
    resident: G,
) -> Vec<u32>
where
    F: FnMut(u32) -> Option<u64>,
    G: Fn(u32) -> bool,
{
    let mut ordered = layers;
    match policy {
        SelectionPolicy::Ascending => {}
        SelectionPolicy::SmallestFirst => {
            ordered.sort_by_key(|layer| (size(*layer).unwrap_or(u64::MAX), *layer));
        }
        SelectionPolicy::LargestFirst => {
            ordered.sort_by_key(|layer| {
                (std::cmp::Reverse(size(*layer).unwrap_or(0)), *layer)
            });
        }
        SelectionPolicy::ResidentFirst => {
            ordered.sort_by_key(|layer| {
                (
                    !resident(*layer),
                    size(*layer).unwrap_or(u64::MAX),
                    *layer,
                )
            });
        }
    }
    ordered
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingRequest {
    pub layers: Vec<u32>,
    pub budget_bytes: Option<u64>,
    pub auto: bool,
    pub safety_margin_bytes: u64,
}

impl Default for StagingRequest {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            budget_bytes: None,
            auto: false,
            safety_margin_bytes: DEFAULT_STAGING_SAFETY_MARGIN_BYTES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StagingPlanState {
    Exact,
    Estimated,
    Partial,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagingLayerPlan {
    pub layer: u32,
    pub bytes: Option<u64>,
    pub accepted: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagingPlan {
    pub schema_version: u32,
    pub requested_layers: Vec<u32>,
    pub auto_selected: bool,
    pub selection_policy: SelectionPolicy,
    pub configured_budget_bytes: Option<u64>,
    pub effective_budget_bytes: Option<u64>,
    pub observed_free_device_bytes: Option<u64>,
    pub safety_margin_bytes: u64,
    pub existing_staged_bytes: u64,
    pub planned_bytes: u64,
    pub available_bytes_after_plan: Option<u64>,
    pub current_resident_layers: Vec<u32>,
    pub current_resident_bytes: u64,
    pub kept_layers: Vec<u32>,
    pub evicted_layers: Vec<u32>,
    pub newly_staged_layers: Vec<u32>,
    pub reclaimable_bytes: u64,
    pub required_new_bytes: u64,
    pub temporary_headroom_bytes: u64,
    pub pipeline: bool,
    pub pipeline_active_bytes: u64,
    pub pipeline_pending_bytes: u64,
    pub pipeline_preserved_bytes: u64,
    pub pipeline_new_bytes: u64,
    pub pipeline_removed_bytes: u64,
    pub pipeline_partial: bool,
    pub standby_candidates: Vec<u32>,
    pub standby_candidate_bytes: u64,
    pub standby_reused_layers: Vec<u32>,
    pub standby_reused_bytes: u64,
    pub pipeline_pool_bytes: u64,
    pub pipeline_required_bytes: u64,
    pub pipeline_fits: bool,
    pub host_staging: Option<HostStagingReport>,
    pub replacement_invalidation_ns: Option<u64>,
    pub replacement_eviction_ns: Option<u64>,
    pub replacement_staging_ns: Option<u64>,
    pub replacement_total_ns: Option<u64>,
    pub layers: Vec<StagingLayerPlan>,
    pub accepted_layers: Vec<u32>,
    pub rejected_layers: Vec<u32>,
    pub eviction_supported: bool,
    pub allocation_reuse_supported: bool,
    pub transfer_compute_overlap_supported: bool,
    pub state: StagingPlanState,
    pub notes: Vec<String>,
}

impl StagingPlan {
    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn size(layer: u32) -> Option<u64> {
        Some(match layer {
            0 => 100,
            1 => 300,
            2 => 200,
            _ => 50,
        })
    }

    #[test]
    fn policies_order_layers_deterministically() {
        let layers = vec![0, 1, 2];
        assert_eq!(
            order_candidate_layers(SelectionPolicy::Ascending, layers.clone(), size, |_| false),
            vec![0, 1, 2]
        );
        assert_eq!(
            order_candidate_layers(
                SelectionPolicy::SmallestFirst,
                layers.clone(),
                size,
                |_| false
            ),
            vec![0, 2, 1]
        );
        assert_eq!(
            order_candidate_layers(
                SelectionPolicy::LargestFirst,
                layers.clone(),
                size,
                |_| false
            ),
            vec![1, 2, 0]
        );
        assert_eq!(
            order_candidate_layers(
                SelectionPolicy::ResidentFirst,
                layers,
                size,
                |layer| layer == 2
            ),
            vec![2, 0, 1]
        );
    }

    #[test]
    fn unknown_sizes_do_not_claim_a_measurement() {
        let ordered = order_candidate_layers(
            SelectionPolicy::SmallestFirst,
            vec![0, 1, 2],
            |layer| (layer != 1).then_some(10),
            |_| false,
        );
        // Known layers first, unknown last, stable by layer id.
        assert_eq!(ordered, vec![0, 2, 1]);
    }
}
