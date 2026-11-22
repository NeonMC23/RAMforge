//! Deterministic staged-layer planning and budget states.

use serde::{Deserialize, Serialize};

pub const STAGING_PLAN_SCHEMA_VERSION: u32 = 4;
pub const DEFAULT_STAGING_SAFETY_MARGIN_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPolicy {
    #[default]
    Ascending,
    SmallestFirst,
    ResidentFirst,
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
    pub pipeline_pool_bytes: u64,
    pub pipeline_required_bytes: u64,
    pub pipeline_fits: bool,
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
