//! Execution-aware residency inventory, policy, planning, and lifecycle.
//!
//! This module consumes forgeCORE tensor-mapping metadata. It does not call
//! native FFI and does not claim that a page-release request reduced RSS.
//! RAMforge owns policy; forgeCORE owns exact native mapping metadata and
//! page-release operations.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use forge_core::{PageReleaseResult, TensorMappingInfo};

use crate::cgroup::{self, CgroupEvents, CgroupStats};
use crate::memory_report::{process_rss, process_rss_peak};
use crate::page_residency::model_mapping_residency;

/// Stable identity for one loaded model within a runtime session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelId(pub String);

/// Monotonic generation for one model handle. Refreshing/reloading invalidates
/// resource keys from older generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ModelGeneration(pub u64);

/// Resource key scoped to one model generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceKey {
    pub generation: u64,
    pub tensor_index: i32,
}

/// How a layer identity was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerProvenance {
    /// Inferred from the model tensor name `blk.N.*`.
    TensorNameConvention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerIdentity {
    Known {
        layer: u32,
        provenance: LayerProvenance,
    },
    Unknown,
}

impl LayerIdentity {
    pub fn layer(self) -> Option<u32> {
        match self {
            Self::Known { layer, .. } => Some(layer),
            Self::Unknown => None,
        }
    }

    fn sort_key(self) -> (u8, u32) {
        match self {
            Self::Known { layer, .. } => (0, layer),
            Self::Unknown => (1, u32::MAX),
        }
    }
}

/// Native metadata confidence for one resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetadataConfidence {
    /// Exact native mapping metadata.
    Native,
    /// Name-derived layer identity only.
    Inferred,
    /// No usable native mapping metadata.
    Unknown,
}

/// Why a resource is not eligible for file-backed page release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IneligibilityReason {
    AnonymousOrRepacked,
    EmptyPageRange,
    InvalidMappingMetadata,
}

/// One model tensor resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorResource {
    pub key: ResourceKey,
    pub name: String,
    pub layer: LayerIdentity,
    pub file_backed: bool,
    pub mapping_index: Option<u32>,
    pub mapping_size: u64,
    pub offset: u64,
    pub logical_bytes: u64,
    pub page_size: u64,
    pub page_start: u64,
    pub page_end: u64,
    pub page_bytes: u64,
    /// Union of pages shared with other file-backed tensors in this model.
    pub shared_page_bytes: u64,
    pub eligibility: Result<(), IneligibilityReason>,
    pub metadata_confidence: MetadataConfidence,
}

impl TensorResource {
    /// Conservative estimate of pages this tensor could release without
    /// touching a page shared with another tensor.
    pub fn unique_reclaimable_bytes(&self) -> u64 {
        self.page_bytes.saturating_sub(self.shared_page_bytes)
    }
}

/// Summary totals for an inventory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InventorySummary {
    pub tensors: usize,
    pub file_backed_tensors: usize,
    pub anonymous_tensors: usize,
    pub unknown_tensors: usize,
    pub mappings: usize,
    pub logical_file_backed_bytes: u64,
    pub logical_anonymous_bytes: u64,
    pub page_bytes: u64,
    pub shared_page_bytes: u64,
}

/// Model-scoped resource inventory derived from native tensor metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceInventory {
    pub model_id: ModelId,
    pub generation: ModelGeneration,
    pub resources: Vec<TensorResource>,
    pub by_index: BTreeMap<i32, usize>,
    pub summary: InventorySummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InventoryError {
    EmptyModel,
    InvalidPageSize {
        tensor: String,
        page_size: u64,
    },
    UnalignedPage {
        tensor: String,
        page_start: u64,
        page_size: u64,
    },
    InvalidRange {
        tensor: String,
        offset: u64,
        size: u64,
        mapping_size: u64,
    },
}

impl std::fmt::Display for InventoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModel => write!(f, "model exposes no tensor resources"),
            Self::InvalidPageSize { tensor, page_size } => {
                write!(f, "tensor {tensor} has invalid page size {page_size}")
            }
            Self::UnalignedPage {
                tensor,
                page_start,
                page_size,
            } => write!(
                f,
                "tensor {tensor} page start {page_start} is not aligned to {page_size}"
            ),
            Self::InvalidRange {
                tensor,
                offset,
                size,
                mapping_size,
            } => write!(
                f,
                "tensor {tensor} range offset={offset} size={size} exceeds mapping {mapping_size}"
            ),
        }
    }
}

impl std::error::Error for InventoryError {}

impl ResourceInventory {
    pub fn from_tensor_mappings(
        model_id: ModelId,
        generation: ModelGeneration,
        mappings: &[TensorMappingInfo],
    ) -> Result<Self, InventoryError> {
        if mappings.is_empty() {
            return Err(InventoryError::EmptyModel);
        }
        let mut resources = Vec::with_capacity(mappings.len());
        let mut by_index = BTreeMap::new();
        let mut mapping_indices = BTreeSet::new();
        let mut summary = InventorySummary::default();

        for (position, mapping) in mappings.iter().enumerate() {
            let tensor_index = i32::try_from(position).map_err(|_| InventoryError::EmptyModel)?;
            let key = ResourceKey {
                generation: generation.0,
                tensor_index,
            };
            let layer = layer_from_name(&mapping.name);
            let file_backed = mapping.file_backed;
            let mut page_start = mapping.page_start;
            let mut page_end = mapping.page_end;
            let mut eligibility = Ok(());
            let mut confidence = MetadataConfidence::Native;

            if file_backed {
                if mapping.page_size == 0 {
                    return Err(InventoryError::InvalidPageSize {
                        tensor: mapping.name.clone(),
                        page_size: 0,
                    });
                }
                if mapping.page_start % mapping.page_size != 0 {
                    return Err(InventoryError::UnalignedPage {
                        tensor: mapping.name.clone(),
                        page_start: mapping.page_start,
                        page_size: mapping.page_size,
                    });
                }
                if mapping
                    .offset
                    .checked_add(mapping.size)
                    .is_none_or(|end| end > mapping.mapping_size)
                {
                    return Err(InventoryError::InvalidRange {
                        tensor: mapping.name.clone(),
                        offset: mapping.offset,
                        size: mapping.size,
                        mapping_size: mapping.mapping_size,
                    });
                }
                page_start = page_start.min(mapping.mapping_size);
                page_end = page_end.min(mapping.mapping_size);
                if page_end <= page_start {
                    eligibility = Err(IneligibilityReason::EmptyPageRange);
                }
                mapping_indices.insert(mapping.mapping_index.unwrap_or(u32::MAX));
                summary.file_backed_tensors += 1;
                summary.logical_file_backed_bytes += mapping.size;
                summary.page_bytes += page_end - page_start;
            } else {
                eligibility = Err(IneligibilityReason::AnonymousOrRepacked);
                confidence = MetadataConfidence::Unknown;
                summary.anonymous_tensors += 1;
                summary.logical_anonymous_bytes += mapping.size;
                page_start = 0;
                page_end = 0;
            }

            let resource = TensorResource {
                key,
                name: mapping.name.clone(),
                layer,
                file_backed,
                mapping_index: mapping.mapping_index,
                mapping_size: mapping.mapping_size,
                offset: mapping.offset,
                logical_bytes: mapping.size,
                page_size: mapping.page_size,
                page_start,
                page_end,
                page_bytes: page_end.saturating_sub(page_start),
                shared_page_bytes: 0,
                eligibility,
                metadata_confidence: confidence,
            };
            by_index.insert(tensor_index, resources.len());
            resources.push(resource);
        }

        // Conservative page-sharing analysis within each mapping.
        let mut shared: HashMap<usize, u64> = HashMap::new();
        for (index, resource) in resources.iter().enumerate() {
            if !resource.file_backed || resource.page_bytes == 0 {
                continue;
            }
            let mut intervals = Vec::new();
            for (other_index, other) in resources.iter().enumerate() {
                if index == other_index
                    || !other.file_backed
                    || other.mapping_index != resource.mapping_index
                    || other.page_bytes == 0
                {
                    continue;
                }
                let first = resource.page_start.max(other.page_start);
                let last = resource.page_end.min(other.page_end);
                if first < last {
                    intervals.push((first, last));
                }
            }
            shared.insert(index, merged_interval_bytes(&mut intervals));
        }
        for (index, bytes) in shared {
            if let Some(resource) = resources.get_mut(index) {
                resource.shared_page_bytes = bytes;
                summary.shared_page_bytes += bytes;
            }
        }
        summary.tensors = resources.len();
        summary.mappings = mapping_indices.len();
        summary.unknown_tensors = resources
            .iter()
            .filter(|resource| !resource.file_backed)
            .count();

        Ok(Self {
            model_id,
            generation,
            resources,
            by_index,
            summary,
        })
    }

    pub fn get(&self, key: ResourceKey) -> Option<&TensorResource> {
        if key.generation != self.generation.0 {
            return None;
        }
        self.by_index
            .get(&key.tensor_index)
            .and_then(|index| self.resources.get(*index))
    }

    pub fn is_current(&self, generation: ModelGeneration) -> bool {
        self.generation == generation
    }

    /// All file-backed resource keys. Used when the next operation may need
    /// every layer (for example prompt prefill).
    pub fn file_backed_keys(&self) -> BTreeSet<ResourceKey> {
        self.resources
            .iter()
            .filter(|resource| resource.file_backed)
            .map(|resource| resource.key)
            .collect()
    }

    pub fn resources_in_layer(&self, layer: u32) -> Vec<&TensorResource> {
        self.resources
            .iter()
            .filter(|resource| resource.layer.layer() == Some(layer))
            .collect()
    }
}

fn layer_from_name(name: &str) -> LayerIdentity {
    let Some(rest) = name.strip_prefix("blk.") else {
        return LayerIdentity::Unknown;
    };
    let Some((number, suffix)) = rest.split_once('.') else {
        return LayerIdentity::Unknown;
    };
    if suffix.is_empty() {
        return LayerIdentity::Unknown;
    }
    let Ok(layer) = number.parse::<u32>() else {
        return LayerIdentity::Unknown;
    };
    LayerIdentity::Known {
        layer,
        provenance: LayerProvenance::TensorNameConvention,
    }
}

fn merged_interval_bytes(intervals: &mut [(u64, u64)]) -> u64 {
    if intervals.is_empty() {
        return 0;
    }
    intervals.sort_unstable();
    let mut total = 0;
    let mut first = intervals[0].0;
    let mut last = intervals[0].1;
    for &(next_first, next_last) in &intervals[1..] {
        if next_first > last {
            total += last - first;
            first = next_first;
            last = next_last;
        } else if next_last > last {
            last = next_last;
        }
    }
    total + last - first
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityState {
    Supported,
    PartiallySupported,
    Unsupported,
    Unknown,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyCapabilities {
    pub tensor_mapping_metadata: CapabilityState,
    pub file_backed_release: CapabilityState,
    pub page_prefetch: CapabilityState,
    pub graph_trace: CapabilityState,
    pub graph_dependency_analysis: CapabilityState,
    pub proven_last_use_completion: CapabilityState,
    pub between_operation_residency: CapabilityState,
    pub intra_decode_residency: CapabilityState,
    pub hard_cgroup_enforcement: CapabilityState,
    pub cpu_execution: CapabilityState,
    pub gpu_execution: CapabilityState,
}

impl ResidencyCapabilities {
    pub fn current() -> Self {
        let linux = cfg!(target_os = "linux");
        let cgroup = cgroup::self_cgroup_path()
            .ok()
            .and_then(|path| cgroup::stats_at(&path).ok())
            .is_some();
        Self {
            tensor_mapping_metadata: CapabilityState::Supported,
            file_backed_release: if linux {
                CapabilityState::Supported
            } else {
                CapabilityState::Unsupported
            },
            page_prefetch: if linux {
                CapabilityState::Supported
            } else {
                CapabilityState::Unsupported
            },
            graph_trace: CapabilityState::Supported,
            graph_dependency_analysis: CapabilityState::PartiallySupported,
            // The callback proves observed node completion, not that the next
            // graph/token cannot reuse a weight.
            proven_last_use_completion: CapabilityState::PartiallySupported,
            between_operation_residency: CapabilityState::Supported,
            intra_decode_residency: CapabilityState::Unsupported,
            hard_cgroup_enforcement: if cgroup {
                CapabilityState::Supported
            } else {
                CapabilityState::Unknown
            },
            cpu_execution: CapabilityState::Supported,
            gpu_execution: CapabilityState::Unsupported,
        }
    }
}

/// Policy mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidencyPolicyMode {
    /// No planning or mutation.
    #[default]
    Off,
    /// Build/print plans and decisions but never mutate residency.
    Observe,
    /// Apply eligible release decisions at a safe boundary.
    Active,
}

/// RAMforge soft residency policy. This is not a hard cgroup limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyPolicy {
    pub mode: ResidencyPolicyMode,
    pub target_bytes: Option<u64>,
    pub safety_reserve_bytes: u64,
}

impl Default for ResidencyPolicy {
    fn default() -> Self {
        Self {
            mode: ResidencyPolicyMode::Off,
            target_bytes: None,
            safety_reserve_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    ReserveExceedsTarget { target: u64, reserve: u64 },
    ActiveWithoutTarget,
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReserveExceedsTarget { target, reserve } => {
                write!(f, "residency reserve {reserve} exceeds target {target}")
            }
            Self::ActiveWithoutTarget => {
                write!(f, "active residency policy requires a soft target")
            }
        }
    }
}

impl std::error::Error for PolicyError {}

impl ResidencyPolicy {
    pub fn validate(self) -> Result<Self, PolicyError> {
        if let Some(target) = self.target_bytes {
            if self.safety_reserve_bytes > target {
                return Err(PolicyError::ReserveExceedsTarget {
                    target,
                    reserve: self.safety_reserve_bytes,
                });
            }
        }
        if self.mode == ResidencyPolicyMode::Active && self.target_bytes.is_none() {
            return Err(PolicyError::ActiveWithoutTarget);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OperationContext {
    pub required: BTreeSet<ResourceKey>,
    pub pinned: BTreeSet<ResourceKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionAction {
    Release,
    Retain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    PolicyOff,
    RequiredByOperation,
    Pinned,
    AnonymousOrRepacked,
    EmptyPageRange,
    AlreadyAttempted,
    AllPagesShared,
    BudgetAlreadyMet,
    SelectedEligible,
    EligibleButNotNeeded,
    InsufficientReclaimableCapacity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDecision {
    pub key: ResourceKey,
    pub name: String,
    pub layer: LayerIdentity,
    pub action: DecisionAction,
    pub reason: DecisionReason,
    pub page_bytes: u64,
    pub shared_page_bytes: u64,
    pub estimated_reclaim_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementSource {
    Measured,
    EstimatedFromMapping,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyPlan {
    pub mode: ResidencyPolicyMode,
    pub target_bytes: Option<u64>,
    pub safety_reserve_bytes: u64,
    pub observed_bytes: Option<u64>,
    pub measurement_source: MeasurementSource,
    pub decisions: Vec<ResourceDecision>,
    pub selected: Vec<ResourceKey>,
    pub estimated_reclaim_bytes: u64,
    pub estimated_after_bytes: Option<u64>,
    pub estimated_budget_met: bool,
    pub notes: Vec<String>,
}

impl ResidencyPlan {
    pub fn is_noop(&self) -> bool {
        self.selected.is_empty()
    }
}

pub struct Planner<'a> {
    inventory: &'a ResourceInventory,
    attempted: &'a BTreeSet<ResourceKey>,
}

impl<'a> Planner<'a> {
    pub fn new(inventory: &'a ResourceInventory, attempted: &'a BTreeSet<ResourceKey>) -> Self {
        Self {
            inventory,
            attempted,
        }
    }

    pub fn plan(
        &self,
        policy: ResidencyPolicy,
        observed_bytes: Option<u64>,
        context: &OperationContext,
    ) -> ResidencyPlan {
        let target = policy.target_bytes;
        let reserve = policy.safety_reserve_bytes;
        let effective_target = target.map(|value| value.saturating_sub(reserve));
        let (current, source) = match observed_bytes {
            Some(bytes) => (Some(bytes), MeasurementSource::Measured),
            None => (
                Some(self.inventory.summary.page_bytes),
                MeasurementSource::EstimatedFromMapping,
            ),
        };

        let mut decisions = Vec::with_capacity(self.inventory.resources.len());
        let mut candidates = Vec::new();
        let need = match (current, effective_target) {
            (Some(current), Some(target)) => current.saturating_sub(target),
            _ => 0,
        };

        for resource in &self.inventory.resources {
            let base = ResourceDecision {
                key: resource.key,
                name: resource.name.clone(),
                layer: resource.layer,
                action: DecisionAction::Retain,
                reason: DecisionReason::EligibleButNotNeeded,
                page_bytes: resource.page_bytes,
                shared_page_bytes: resource.shared_page_bytes,
                estimated_reclaim_bytes: resource.unique_reclaimable_bytes(),
            };
            if policy.mode == ResidencyPolicyMode::Off {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::PolicyOff,
                    ..base
                });
                continue;
            }
            if context.required.contains(&resource.key) {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::RequiredByOperation,
                    ..base
                });
                continue;
            }
            if context.pinned.contains(&resource.key) {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::Pinned,
                    ..base
                });
                continue;
            }
            if !resource.file_backed {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::AnonymousOrRepacked,
                    ..base
                });
                continue;
            }
            if resource.page_bytes == 0 {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::EmptyPageRange,
                    ..base
                });
                continue;
            }
            if self.attempted.contains(&resource.key) {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::AlreadyAttempted,
                    ..base
                });
                continue;
            }
            if resource.unique_reclaimable_bytes() == 0 {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::AllPagesShared,
                    ..base
                });
                continue;
            }
            if need == 0 {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::BudgetAlreadyMet,
                    ..base
                });
                continue;
            }
            candidates.push(base);
        }

        candidates.sort_by(|a, b| {
            let a_shared = a.shared_page_bytes > 0;
            let b_shared = b.shared_page_bytes > 0;
            a_shared
                .cmp(&b_shared)
                .then_with(|| b.estimated_reclaim_bytes.cmp(&a.estimated_reclaim_bytes))
                .then_with(|| a.layer.sort_key().cmp(&b.layer.sort_key()))
                .then_with(|| a.key.tensor_index.cmp(&b.key.tensor_index))
        });

        let mut selected = Vec::new();
        let mut reclaim = 0u64;
        for candidate in candidates {
            if need > 0 && reclaim < need {
                reclaim += candidate.estimated_reclaim_bytes;
                selected.push(candidate.key);
                decisions.push(ResourceDecision {
                    action: DecisionAction::Release,
                    reason: DecisionReason::SelectedEligible,
                    ..candidate
                });
            } else {
                decisions.push(ResourceDecision {
                    reason: DecisionReason::EligibleButNotNeeded,
                    ..candidate
                });
            }
        }

        let estimated_after = current.map(|current| current.saturating_sub(reclaim));
        let estimated_budget_met = match (estimated_after, effective_target) {
            (Some(after), Some(target)) => after <= target,
            (_, None) => true,
            _ => false,
        };
        let mut notes = Vec::new();
        if policy.mode == ResidencyPolicyMode::Off {
            notes.push("residency policy is off".to_string());
        }
        if source == MeasurementSource::EstimatedFromMapping {
            notes.push("current residency is estimated from mapped tensor pages".to_string());
        }
        if !estimated_budget_met && selected.is_empty() {
            notes.push("insufficient eligible reclaimable capacity".to_string());
        } else if !estimated_budget_met {
            notes.push("selected candidates do not fully meet the soft target".to_string());
        }

        ResidencyPlan {
            mode: policy.mode,
            target_bytes: target,
            safety_reserve_bytes: reserve,
            observed_bytes: current,
            measurement_source: source,
            decisions,
            selected,
            estimated_reclaim_bytes: reclaim,
            estimated_after_bytes: estimated_after,
            estimated_budget_met,
            notes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionOutcome {
    Admitted,
    AdmittedWithWarning,
    Rejected,
    Unknown,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionDecision {
    pub outcome: AdmissionOutcome,
    pub soft_target_bytes: Option<u64>,
    pub hard_limit_bytes: Option<u64>,
    pub safety_reserve_bytes: u64,
    pub current_bytes: Option<u64>,
    pub known_tensor_bytes: u64,
    pub reasons: Vec<String>,
    pub unknown_components: Vec<String>,
}

pub fn admit(
    soft_target_bytes: Option<u64>,
    hard_limit_bytes: Option<u64>,
    safety_reserve_bytes: u64,
    current_bytes: Option<u64>,
    known_tensor_bytes: u64,
) -> AdmissionDecision {
    let mut reasons = Vec::new();
    let unknown_components = vec![
        "KV cache".to_string(),
        "graph/scratch buffers".to_string(),
        "allocator/runtime overhead".to_string(),
    ];
    let outcome = if let Some(current) = current_bytes {
        let current_with_reserve = current.saturating_add(safety_reserve_bytes);
        if let Some(hard) = hard_limit_bytes {
            if current_with_reserve > hard
                || known_tensor_bytes.saturating_add(safety_reserve_bytes) > hard
            {
                reasons.push(
                    "current/known memory plus reserve exceeds the hard cgroup limit".to_string(),
                );
                AdmissionOutcome::Rejected
            } else if soft_target_bytes.is_some_and(|target| current_with_reserve > target) {
                reasons.push("current memory plus reserve exceeds the soft target".to_string());
                AdmissionOutcome::AdmittedWithWarning
            } else {
                AdmissionOutcome::Admitted
            }
        } else if soft_target_bytes.is_some_and(|target| current_with_reserve > target) {
            reasons.push("current memory plus reserve exceeds the soft target".to_string());
            AdmissionOutcome::AdmittedWithWarning
        } else {
            AdmissionOutcome::Admitted
        }
    } else if hard_limit_bytes.is_some() || soft_target_bytes.is_some() {
        reasons.push("current memory is unavailable".to_string());
        AdmissionOutcome::Unknown
    } else {
        reasons.push("no soft or hard budget was configured".to_string());
        AdmissionOutcome::Admitted
    };
    AdmissionDecision {
        outcome,
        soft_target_bytes,
        hard_limit_bytes,
        safety_reserve_bytes,
        current_bytes,
        known_tensor_bytes,
        reasons,
        unknown_components,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyState {
    ModelReady,
    PreparingOperation,
    Executing,
    Synchronizing,
    SafeBoundary,
    Planning,
    Applying,
    Failed,
    ShuttingDown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleError {
    pub from: ResidencyState,
    pub to: ResidencyState,
}

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "illegal residency transition {:?} -> {:?}",
            self.from, self.to
        )
    }
}

impl std::error::Error for LifecycleError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyLifecycle {
    state: ResidencyState,
}

impl Default for ResidencyLifecycle {
    fn default() -> Self {
        Self {
            state: ResidencyState::ModelReady,
        }
    }
}

impl ResidencyLifecycle {
    pub fn state(self) -> ResidencyState {
        self.state
    }

    fn transition(&mut self, to: ResidencyState) -> Result<(), LifecycleError> {
        let legal = matches!(
            (self.state, to),
            (
                ResidencyState::ModelReady,
                ResidencyState::PreparingOperation
            ) | (ResidencyState::ModelReady, ResidencyState::Planning)
                | (
                    ResidencyState::PreparingOperation,
                    ResidencyState::Executing
                )
                | (ResidencyState::PreparingOperation, ResidencyState::Planning)
                | (
                    ResidencyState::PreparingOperation,
                    ResidencyState::SafeBoundary
                )
                | (ResidencyState::Executing, ResidencyState::Synchronizing)
                | (ResidencyState::Synchronizing, ResidencyState::SafeBoundary)
                | (ResidencyState::SafeBoundary, ResidencyState::Executing)
                | (ResidencyState::SafeBoundary, ResidencyState::Planning)
                | (
                    ResidencyState::SafeBoundary,
                    ResidencyState::PreparingOperation
                )
                | (ResidencyState::Planning, ResidencyState::Applying)
                | (ResidencyState::Planning, ResidencyState::SafeBoundary)
                | (ResidencyState::Applying, ResidencyState::SafeBoundary)
                | (ResidencyState::Failed, ResidencyState::SafeBoundary)
                | (ResidencyState::ModelReady, ResidencyState::ShuttingDown)
                | (
                    ResidencyState::PreparingOperation,
                    ResidencyState::ShuttingDown
                )
                | (ResidencyState::Executing, ResidencyState::ShuttingDown)
                | (ResidencyState::Synchronizing, ResidencyState::ShuttingDown)
                | (ResidencyState::SafeBoundary, ResidencyState::ShuttingDown)
                | (ResidencyState::Planning, ResidencyState::ShuttingDown)
                | (ResidencyState::Applying, ResidencyState::ShuttingDown)
                | (ResidencyState::Failed, ResidencyState::ShuttingDown)
        );
        if !legal {
            return Err(LifecycleError {
                from: self.state,
                to,
            });
        }
        self.state = to;
        Ok(())
    }

    fn fail(&mut self) {
        if self.state != ResidencyState::ShuttingDown {
            self.state = ResidencyState::Failed;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingPlan {
    pub selected: Vec<ResourceKey>,
    pub selected_bytes: u64,
    pub skipped_bytes: u64,
    pub notes: Vec<String>,
}

impl StagingPlan {
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }
}

/// One measured/estimated residency snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencySnapshot {
    pub process_rss_bytes: Option<u64>,
    pub process_rss_peak_bytes: Option<u64>,
    pub mapping_rss_bytes: Option<u64>,
    pub mapping_pss_bytes: Option<u64>,
    pub cgroup_current_bytes: Option<u64>,
    pub cgroup_peak_bytes: Option<u64>,
    pub cgroup_events: Option<CgroupEvents>,
}

impl ResidencySnapshot {
    pub fn capture(model_path: &Path) -> Self {
        Self::capture_paths(std::slice::from_ref(&model_path.to_path_buf()))
    }

    /// Capture mapping residency across all model shard paths.
    pub fn capture_paths(model_paths: &[PathBuf]) -> Self {
        let mut mapping_rss = None;
        let mut mapping_pss = None;
        for path in model_paths {
            if let Some(mapping) = model_mapping_residency(path) {
                mapping_rss = Some(
                    mapping_rss
                        .unwrap_or(0u64)
                        .saturating_add(mapping.rss_bytes),
                );
                mapping_pss = Some(
                    mapping_pss
                        .unwrap_or(0u64)
                        .saturating_add(mapping.pss_bytes),
                );
            }
        }
        let cgroup = cgroup::self_cgroup_path()
            .ok()
            .and_then(|path| cgroup::stats_at(&path).ok());
        Self::from_parts(
            process_rss(),
            process_rss_peak(),
            mapping_rss,
            mapping_pss,
            cgroup,
        )
    }

    fn from_parts(
        process_rss_bytes: Option<u64>,
        process_rss_peak_bytes: Option<u64>,
        mapping_rss_bytes: Option<u64>,
        mapping_pss_bytes: Option<u64>,
        cgroup: Option<CgroupStats>,
    ) -> Self {
        Self {
            process_rss_bytes,
            process_rss_peak_bytes,
            mapping_rss_bytes,
            mapping_pss_bytes,
            cgroup_current_bytes: cgroup.as_ref().and_then(|value| value.current_bytes),
            cgroup_peak_bytes: cgroup.as_ref().and_then(|value| value.peak_bytes),
            cgroup_events: cgroup.and_then(|value| value.events),
        }
    }

    pub fn mapping_delta(&self, after: &Self) -> Option<i128> {
        match (self.mapping_rss_bytes, after.mapping_rss_bytes) {
            (Some(before), Some(after)) => Some(after as i128 - before as i128),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplicationReport {
    pub attempted_keys: Vec<ResourceKey>,
    pub native_indices: Vec<i32>,
    pub native_result: Option<PageReleaseResult>,
    pub error: Option<String>,
    pub elapsed: Duration,
}

/// Structured per-operation residency report. Estimates and observations are
/// separate fields; a native release result is not an RSS measurement.
#[derive(Debug, Clone)]
pub struct ResidencyReport {
    pub model_id: ModelId,
    pub inventory: InventorySummary,
    pub admission: AdmissionDecision,
    pub pre_plan: Option<ResidencyPlan>,
    pub post_plan: Option<ResidencyPlan>,
    pub application: Option<ApplicationReport>,
    pub before: ResidencySnapshot,
    pub after: ResidencySnapshot,
    pub elapsed: Duration,
}

/// Single-model residency manager. It owns no native handles; mutations are
/// applied through a caller-provided forgeCORE-backed closure at a safe
/// lifecycle boundary.
#[derive(Debug)]
pub struct ResidencyManager {
    model_id: ModelId,
    inventory: ResourceInventory,
    policy: ResidencyPolicy,
    hard_limit_bytes: Option<u64>,
    lifecycle: ResidencyLifecycle,
    attempted: BTreeSet<ResourceKey>,
    last_plan: Option<ResidencyPlan>,
    application: Option<ApplicationReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagerError {
    Policy(PolicyError),
    Inventory(InventoryError),
    Lifecycle(LifecycleError),
    StalePlan,
    NoPlan,
    ObserveOnly,
    Release(String),
}

impl std::fmt::Display for ManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Policy(error) => write!(f, "{error}"),
            Self::Inventory(error) => write!(f, "{error}"),
            Self::Lifecycle(error) => write!(f, "{error}"),
            Self::StalePlan => write!(f, "residency plan is stale for this model generation"),
            Self::NoPlan => write!(f, "no residency plan is available"),
            Self::ObserveOnly => write!(f, "observation-only policy cannot apply releases"),
            Self::Release(error) => write!(f, "residency release failed: {error}"),
        }
    }
}

impl std::error::Error for ManagerError {}

impl ResidencyManager {
    pub fn new(
        model_id: ModelId,
        inventory: ResourceInventory,
        policy: ResidencyPolicy,
        hard_limit_bytes: Option<u64>,
    ) -> Result<Self, ManagerError> {
        let policy = policy.validate().map_err(ManagerError::Policy)?;
        Ok(Self {
            model_id,
            inventory,
            policy,
            hard_limit_bytes,
            lifecycle: ResidencyLifecycle::default(),
            attempted: BTreeSet::new(),
            last_plan: None,
            application: None,
        })
    }

    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    pub fn inventory(&self) -> &ResourceInventory {
        &self.inventory
    }

    pub fn policy(&self) -> ResidencyPolicy {
        self.policy
    }

    pub fn hard_limit_bytes(&self) -> Option<u64> {
        self.hard_limit_bytes
    }

    pub fn state(&self) -> ResidencyState {
        self.lifecycle.state()
    }

    pub fn last_plan(&self) -> Option<&ResidencyPlan> {
        self.last_plan.as_ref()
    }

    pub fn application(&self) -> Option<&ApplicationReport> {
        self.application.as_ref()
    }

    /// Deterministic bounded staging plan. This is a prefetch hint only; it
    /// does not reserve memory or guarantee residency.
    pub fn staging_plan(&self, max_bytes: u64) -> StagingPlan {
        let mut candidates: Vec<&TensorResource> = self
            .inventory
            .resources
            .iter()
            .filter(|resource| resource.file_backed && resource.page_bytes > 0)
            .collect();
        candidates.sort_by(|a, b| {
            a.layer
                .sort_key()
                .cmp(&b.layer.sort_key())
                .then_with(|| a.key.tensor_index.cmp(&b.key.tensor_index))
        });
        let mut selected = Vec::new();
        let mut selected_bytes = 0u64;
        let mut skipped = 0u64;
        for resource in candidates {
            if max_bytes == 0 || selected_bytes.saturating_add(resource.page_bytes) > max_bytes {
                skipped = skipped.saturating_add(resource.page_bytes);
                continue;
            }
            selected.push(resource.key);
            selected_bytes += resource.page_bytes;
        }
        let mut notes = vec![
            "prefetch is a MADV_WILLNEED hint, not a synchronous load or residency guarantee"
                .to_string(),
        ];
        if skipped > 0 {
            notes.push("some eligible pages exceeded the staging limit".to_string());
        }
        StagingPlan {
            selected,
            selected_bytes,
            skipped_bytes: skipped,
            notes,
        }
    }

    pub fn admission(&self, current_bytes: Option<u64>) -> AdmissionDecision {
        admit(
            self.policy.target_bytes,
            self.hard_limit_bytes,
            self.policy.safety_reserve_bytes,
            current_bytes,
            self.inventory
                .summary
                .logical_file_backed_bytes
                .saturating_add(self.inventory.summary.logical_anonymous_bytes),
        )
    }

    pub fn begin_operation(&mut self) -> Result<(), ManagerError> {
        self.lifecycle
            .transition(ResidencyState::PreparingOperation)
            .map_err(ManagerError::Lifecycle)
    }

    pub fn begin_decode(&mut self) -> Result<(), ManagerError> {
        self.lifecycle
            .transition(ResidencyState::Executing)
            .map_err(ManagerError::Lifecycle)
    }

    pub fn finish_decode(&mut self) -> Result<(), ManagerError> {
        self.lifecycle
            .transition(ResidencyState::Synchronizing)
            .map_err(ManagerError::Lifecycle)?;
        self.lifecycle
            .transition(ResidencyState::SafeBoundary)
            .map_err(ManagerError::Lifecycle)
    }

    pub fn plan(
        &mut self,
        observed_bytes: Option<u64>,
        context: &OperationContext,
    ) -> Result<ResidencyPlan, ManagerError> {
        self.lifecycle
            .transition(ResidencyState::Planning)
            .map_err(ManagerError::Lifecycle)?;
        let plan = Planner::new(&self.inventory, &self.attempted).plan(
            self.policy,
            observed_bytes,
            context,
        );
        self.lifecycle
            .transition(ResidencyState::SafeBoundary)
            .map_err(ManagerError::Lifecycle)?;
        self.last_plan = Some(plan.clone());
        Ok(plan)
    }

    pub fn apply_last_plan<F>(&mut self, release: F) -> Result<ApplicationReport, ManagerError>
    where
        F: FnOnce(&[i32]) -> Result<PageReleaseResult, String>,
    {
        if self.policy.mode == ResidencyPolicyMode::Observe {
            return Err(ManagerError::ObserveOnly);
        }
        if self.policy.mode == ResidencyPolicyMode::Off {
            return Err(ManagerError::NoPlan);
        }
        let plan = self.last_plan.clone().ok_or(ManagerError::NoPlan)?;
        if plan.selected.is_empty() {
            return Ok(ApplicationReport {
                attempted_keys: Vec::new(),
                native_indices: Vec::new(),
                native_result: None,
                error: None,
                elapsed: Duration::ZERO,
            });
        }
        if self.state() != ResidencyState::SafeBoundary {
            return Err(ManagerError::Lifecycle(LifecycleError {
                from: self.state(),
                to: ResidencyState::Applying,
            }));
        }
        let mut indices = Vec::with_capacity(plan.selected.len());
        for key in &plan.selected {
            let Some(resource) = self.inventory.get(*key) else {
                self.lifecycle.fail();
                return Err(ManagerError::StalePlan);
            };
            if !resource.file_backed || resource.page_bytes == 0 {
                self.lifecycle.fail();
                return Err(ManagerError::StalePlan);
            }
            indices.push(resource.key.tensor_index);
        }

        self.lifecycle
            .transition(ResidencyState::Planning)
            .map_err(ManagerError::Lifecycle)?;
        self.lifecycle
            .transition(ResidencyState::Applying)
            .map_err(ManagerError::Lifecycle)?;
        let start = Instant::now();
        let result = release(&indices);
        let report = match result {
            Ok(native_result) => {
                for key in &plan.selected {
                    self.attempted.insert(*key);
                }
                self.lifecycle
                    .transition(ResidencyState::SafeBoundary)
                    .map_err(ManagerError::Lifecycle)?;
                ApplicationReport {
                    attempted_keys: plan.selected.clone(),
                    native_indices: indices,
                    native_result: Some(native_result),
                    error: None,
                    elapsed: start.elapsed(),
                }
            }
            Err(error) => {
                self.lifecycle.fail();
                ApplicationReport {
                    attempted_keys: plan.selected.clone(),
                    native_indices: indices,
                    native_result: None,
                    error: Some(error.clone()),
                    elapsed: start.elapsed(),
                }
            }
        };
        self.application = Some(report.clone());
        if let Some(error) = &report.error {
            return Err(ManagerError::Release(error.clone()));
        }
        Ok(report)
    }

    /// Mark a failed operation. The manager can be recovered to SafeBoundary
    /// after the caller has handled the error.
    pub fn fail(&mut self) {
        self.lifecycle.fail();
    }

    pub fn recover(&mut self) -> Result<(), ManagerError> {
        self.lifecycle
            .transition(ResidencyState::SafeBoundary)
            .map_err(ManagerError::Lifecycle)
    }

    pub fn refresh(&mut self, inventory: ResourceInventory) -> Result<(), ManagerError> {
        if inventory.model_id != self.model_id {
            return Err(ManagerError::StalePlan);
        }
        if self.state() != ResidencyState::SafeBoundary
            && self.state() != ResidencyState::ModelReady
        {
            return Err(ManagerError::Lifecycle(LifecycleError {
                from: self.state(),
                to: ResidencyState::Planning,
            }));
        }
        self.inventory = inventory;
        self.attempted.clear();
        self.last_plan = None;
        self.application = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping(
        name: &str,
        file_backed: bool,
        mapping_index: i32,
        offset: u64,
        size: u64,
    ) -> TensorMappingInfo {
        TensorMappingInfo {
            name: name.to_string(),
            file_backed,
            mapping_index: file_backed.then_some(mapping_index as u32),
            mapping_size: 1 << 20,
            offset,
            size,
            page_size: 4096,
            page_start: offset & !4095,
            page_end: (offset + size + 4095) & !4095,
        }
    }

    fn inventory() -> ResourceInventory {
        ResourceInventory::from_tensor_mappings(
            ModelId("test-model".to_string()),
            ModelGeneration(1),
            &[
                mapping("blk.0.attn.weight", true, 0, 0, 8192),
                mapping("blk.0.ffn.weight", true, 0, 8192, 16384),
                mapping("blk.1.attn.weight", true, 0, 32768, 8192),
                mapping("output.weight", false, -1, 0, 4096),
            ],
        )
        .expect("inventory")
    }

    #[test]
    fn inventory_classifies_and_aligns_resources() {
        let inventory = inventory();
        assert_eq!(inventory.summary.tensors, 4);
        assert_eq!(inventory.summary.file_backed_tensors, 3);
        assert_eq!(inventory.summary.anonymous_tensors, 1);
        let first = &inventory.resources[0];
        assert_eq!(first.layer.layer(), Some(0));
        assert_eq!(first.page_start % 4096, 0);
        assert!(first.file_backed);
        assert_eq!(inventory.resources[3].layer, LayerIdentity::Unknown);
        assert_eq!(
            inventory.resources[3].eligibility,
            Err(IneligibilityReason::AnonymousOrRepacked)
        );
    }

    #[test]
    fn shared_page_accounting_is_conservative() {
        let inventory = ResourceInventory::from_tensor_mappings(
            ModelId("shared".to_string()),
            ModelGeneration(1),
            &[
                mapping("blk.0.a", true, 0, 0, 4096),
                mapping("blk.0.b", true, 0, 2048, 4096),
            ],
        )
        .expect("inventory");
        assert!(inventory.resources[0].shared_page_bytes > 0);
        assert!(inventory.resources[1].shared_page_bytes > 0);
        assert_eq!(inventory.resources[0].unique_reclaimable_bytes(), 0);
    }

    #[test]
    fn planner_is_deterministic_and_excludes_required_and_anonymous() {
        let inventory = ResourceInventory::from_tensor_mappings(
            ModelId("plan".to_string()),
            ModelGeneration(1),
            &[
                mapping("blk.0.a", true, 0, 0, 16384),
                mapping("blk.1.b", true, 0, 16384, 32768),
                mapping("output", false, -1, 0, 4096),
            ],
        )
        .expect("inventory");
        let mut required = BTreeSet::new();
        required.insert(inventory.resources[0].key);
        let context = OperationContext {
            required,
            pinned: BTreeSet::new(),
        };
        let policy = ResidencyPolicy {
            mode: ResidencyPolicyMode::Active,
            target_bytes: Some(1024),
            safety_reserve_bytes: 0,
        };
        let plan =
            Planner::new(&inventory, &BTreeSet::new()).plan(policy, Some(64 * 1024), &context);
        assert_eq!(plan.selected, vec![inventory.resources[1].key]);
        assert!(plan.decisions.iter().any(|decision| {
            decision.key == inventory.resources[0].key
                && decision.reason == DecisionReason::RequiredByOperation
        }));
        assert!(plan.decisions.iter().any(|decision| {
            decision.key == inventory.resources[2].key
                && decision.reason == DecisionReason::AnonymousOrRepacked
        }));
        let again =
            Planner::new(&inventory, &BTreeSet::new()).plan(policy, Some(64 * 1024), &context);
        assert_eq!(plan, again);
    }

    #[test]
    fn planner_retains_attempted_resources() {
        let inventory = inventory();
        let attempted: BTreeSet<_> = inventory
            .resources
            .iter()
            .filter(|resource| resource.file_backed)
            .map(|resource| resource.key)
            .collect();
        let plan = Planner::new(&inventory, &attempted).plan(
            ResidencyPolicy {
                mode: ResidencyPolicyMode::Active,
                target_bytes: Some(1),
                safety_reserve_bytes: 0,
            },
            Some(1 << 20),
            &OperationContext::default(),
        );
        assert!(plan.selected.is_empty());
        assert!(plan
            .decisions
            .iter()
            .any(|decision| { decision.reason == DecisionReason::AlreadyAttempted }));
    }

    #[test]
    fn planner_respects_safety_reserve_and_noop_budget() {
        let inventory = inventory();
        let plan = Planner::new(&inventory, &BTreeSet::new()).plan(
            ResidencyPolicy {
                mode: ResidencyPolicyMode::Active,
                target_bytes: Some(4096),
                safety_reserve_bytes: 2048,
            },
            Some(2048),
            &OperationContext::default(),
        );
        assert!(plan.selected.is_empty());
        assert!(plan
            .decisions
            .iter()
            .any(|decision| decision.reason == DecisionReason::BudgetAlreadyMet));
        assert!(plan.estimated_budget_met);
    }

    #[test]
    fn planner_excludes_pinned_resources() {
        let inventory = inventory();
        let pinned = [inventory.resources[1].key].into_iter().collect();
        let plan = Planner::new(&inventory, &BTreeSet::new()).plan(
            ResidencyPolicy {
                mode: ResidencyPolicyMode::Active,
                target_bytes: Some(1),
                safety_reserve_bytes: 0,
            },
            Some(1 << 20),
            &OperationContext {
                required: BTreeSet::new(),
                pinned,
            },
        );
        assert!(!plan.selected.contains(&inventory.resources[1].key));
        assert!(plan
            .decisions
            .iter()
            .any(|decision| decision.reason == DecisionReason::Pinned));
    }

    #[test]
    fn duplicate_names_remain_distinct_by_native_index() {
        let inventory = ResourceInventory::from_tensor_mappings(
            ModelId("dupes".to_string()),
            ModelGeneration(1),
            &[
                mapping("blk.0.same", true, 0, 0, 4096),
                mapping("blk.0.same", true, 0, 4096, 4096),
            ],
        )
        .expect("inventory");
        assert_eq!(inventory.resources[0].name, inventory.resources[1].name);
        assert_ne!(inventory.resources[0].key, inventory.resources[1].key);
    }

    #[test]
    fn invalid_mapping_range_is_rejected() {
        let mut mapping = mapping("blk.0.bad", true, 0, 1 << 20, 4096);
        mapping.mapping_size = 1 << 20;
        let error = ResourceInventory::from_tensor_mappings(
            ModelId("bad".to_string()),
            ModelGeneration(1),
            &[mapping],
        )
        .unwrap_err();
        assert!(matches!(error, InventoryError::InvalidRange { .. }));
    }

    #[test]
    fn two_managers_apply_their_own_indices() {
        let make = |id: &str| {
            ResourceInventory::from_tensor_mappings(
                ModelId(id.to_string()),
                ModelGeneration(1),
                &[mapping("blk.0.a", true, 0, 0, 16384)],
            )
            .expect("inventory")
        };
        let policy = ResidencyPolicy {
            mode: ResidencyPolicyMode::Active,
            target_bytes: Some(1),
            safety_reserve_bytes: 0,
        };
        let mut first = ResidencyManager::new(ModelId("a".to_string()), make("a"), policy, None)
            .expect("manager a");
        let mut second = ResidencyManager::new(ModelId("b".to_string()), make("b"), policy, None)
            .expect("manager b");
        first
            .plan(Some(1 << 20), &OperationContext::default())
            .expect("plan a");
        second
            .plan(Some(1 << 20), &OperationContext::default())
            .expect("plan b");
        let mut seen_a = Vec::new();
        let mut seen_b = Vec::new();
        first
            .apply_last_plan(|indices| {
                seen_a.extend_from_slice(indices);
                Ok(PageReleaseResult::default())
            })
            .expect("apply a");
        second
            .apply_last_plan(|indices| {
                seen_b.extend_from_slice(indices);
                Ok(PageReleaseResult::default())
            })
            .expect("apply b");
        assert_eq!(seen_a, vec![0]);
        assert_eq!(seen_b, vec![0]);
    }

    #[test]
    fn admission_distinguishes_soft_and_hard_limits() {
        let soft = admit(Some(1024), None, 0, Some(2048), 1024);
        assert_eq!(soft.outcome, AdmissionOutcome::AdmittedWithWarning);
        let hard = admit(Some(4096), Some(1024), 0, Some(2048), 1024);
        assert_eq!(hard.outcome, AdmissionOutcome::Rejected);
        let unknown = admit(Some(1024), None, 0, None, 1024);
        assert_eq!(unknown.outcome, AdmissionOutcome::Unknown);
    }

    #[test]
    fn lifecycle_rejects_release_while_executing() {
        let inventory = inventory();
        let policy = ResidencyPolicy {
            mode: ResidencyPolicyMode::Active,
            target_bytes: Some(1),
            safety_reserve_bytes: 0,
        };
        let mut manager = ResidencyManager::new(ModelId("m".to_string()), inventory, policy, None)
            .expect("manager");
        manager.begin_operation().expect("prepare");
        manager.begin_decode().expect("execute");
        assert_eq!(manager.state(), ResidencyState::Executing);
        let error = manager
            .plan(None, &OperationContext::default())
            .unwrap_err();
        assert!(matches!(error, ManagerError::Lifecycle(_)));
        manager.finish_decode().expect("safe");
        assert_eq!(manager.state(), ResidencyState::SafeBoundary);
    }

    #[test]
    fn manager_applies_only_selected_indices() {
        let inventory = ResourceInventory::from_tensor_mappings(
            ModelId("apply".to_string()),
            ModelGeneration(1),
            &[
                mapping("blk.0.a", true, 0, 0, 16384),
                mapping("blk.1.b", true, 0, 16384, 32768),
            ],
        )
        .expect("inventory");
        let policy = ResidencyPolicy {
            mode: ResidencyPolicyMode::Active,
            target_bytes: Some(1),
            safety_reserve_bytes: 0,
        };
        let mut manager =
            ResidencyManager::new(ModelId("apply".to_string()), inventory, policy, None)
                .expect("manager");
        let plan = manager
            .plan(Some(1 << 20), &OperationContext::default())
            .expect("plan");
        assert!(!plan.selected.is_empty());
        let mut seen = Vec::new();
        let report = manager
            .apply_last_plan(|indices| {
                seen.extend_from_slice(indices);
                Ok(PageReleaseResult {
                    released_bytes: 4096,
                    skipped_shared_bytes: 0,
                    non_file_backed_tensors: 0,
                    selected_tensors: indices.len() as u64,
                })
            })
            .expect("apply");
        assert_eq!(seen, report.native_indices);
        assert_eq!(report.native_result.unwrap().released_bytes, 4096);
        assert_eq!(manager.state(), ResidencyState::SafeBoundary);
    }

    #[test]
    fn observe_mode_never_applies() {
        let inventory = inventory();
        let policy = ResidencyPolicy {
            mode: ResidencyPolicyMode::Observe,
            target_bytes: Some(1),
            safety_reserve_bytes: 0,
        };
        let mut manager =
            ResidencyManager::new(ModelId("observe".to_string()), inventory, policy, None)
                .expect("manager");
        manager
            .plan(Some(1 << 20), &OperationContext::default())
            .expect("plan");
        assert!(matches!(
            manager.apply_last_plan(|_| Ok(PageReleaseResult::default())),
            Err(ManagerError::ObserveOnly)
        ));
    }

    #[test]
    fn refresh_invalidates_old_attempts_and_plans() {
        let inventory = inventory();
        let policy = ResidencyPolicy {
            mode: ResidencyPolicyMode::Active,
            target_bytes: Some(1),
            safety_reserve_bytes: 0,
        };
        let mut manager =
            ResidencyManager::new(ModelId("refresh".to_string()), inventory, policy, None)
                .expect("manager");
        manager
            .plan(Some(1 << 20), &OperationContext::default())
            .expect("plan");
        let refreshed = ResourceInventory::from_tensor_mappings(
            ModelId("refresh".to_string()),
            ModelGeneration(2),
            &[mapping("blk.0.a", true, 0, 0, 4096)],
        )
        .expect("refreshed");
        manager.refresh(refreshed).expect("refresh");
        assert!(manager.last_plan().is_none());
        assert_eq!(manager.inventory().generation, ModelGeneration(2));
    }
}
