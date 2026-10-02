//! Explicit representation and accounting primitives for Phase 3.
//!
//! This module describes resources; it does not move, evict, prefetch, place,
//! or execute them.  A GGUF tensor can therefore have three deliberately
//! separate views:
//!
//! * [`GgufTensorMetadata`] and [`FileBackedRange`] describe the file-backed
//!   source without implying that the bytes are resident;
//! * [`LogicalTensorProperties`] describe shape, dtype, elements, and the
//!   logical representation size; and
//! * [`AllocationRecord`] describes a requested/runtime allocation, whose
//!   actual size, domain, and residency may be explicitly unknown.
//!
//! [`ResourceAccountingSnapshot`] keeps file/logical bytes, requested bytes,
//! actual allocation bytes, padding, and resident bytes in separate totals.
//! Exact, estimated, and unavailable values are never silently collapsed into
//! one number.  In particular, storage, host RAM, device memory, mapped
//! virtual address space, and an unknown domain are separate accounting keys.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::datasource::GgufDataSource;
use crate::memory::MemoryBudget;
use crate::model::{GgufModel, TensorDescriptor};
use crate::types::GgmlType;

macro_rules! define_string_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

define_string_id!(ModelId);
define_string_id!(TensorId);
define_string_id!(ResourceId);
define_string_id!(AllocationId);
define_string_id!(DeviceId);
define_string_id!(BackendId);

impl ModelId {
    /// Create a deterministic model identifier from the caller's path spelling.
    /// No random or process-local value is used.  Callers that need a path-
    /// independent identity can pass a content/version identifier to `new`.
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self::new(path.as_ref().to_string_lossy().into_owned())
    }
}

impl TensorId {
    /// Construct a stable tensor identifier from a stable model identifier and
    /// the GGUF tensor name.  Tensor names are the GGUF identity within a
    /// model; no allocation order is part of this ID.
    pub fn from_model_and_name(model_id: &ModelId, tensor_name: &str) -> Self {
        Self::new(format!(
            "model={};tensor={}",
            model_id.as_str(),
            tensor_name
        ))
    }
}

impl ResourceId {
    pub fn for_tensor(tensor_id: &TensorId) -> Self {
        Self::new(format!("tensor-resource:{}", tensor_id.as_str()))
    }
}

impl AllocationId {
    pub fn for_resource(resource_id: &ResourceId) -> Self {
        Self::new(format!("allocation:{}", resource_id.as_str()))
    }
}

/// Stable transformer layer identity used by ownership and classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerId(u64);

impl LayerId {
    pub const fn new(index: u64) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u64 {
        self.0
    }
}

impl fmt::Display for LayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Physical/accounting domains are intentionally not interchangeable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryDomain {
    /// Bytes in the model's storage/file representation.  This is not RAM.
    Storage,
    /// RAMforge-managed or otherwise identified host memory.
    HostRam,
    /// Device-local memory, including VRAM.  The device identity is explicit.
    Device(DeviceId),
    /// Virtual address space used for a mapping.  This is not resident RAM.
    MappedVirtual,
    /// A measurement whose domain is not available from the safe API.
    Unknown,
}

impl fmt::Display for MemoryDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage => f.write_str("storage"),
            Self::HostRam => f.write_str("host-ram"),
            Self::Device(device) => write!(f, "device:{device}"),
            Self::MappedVirtual => f.write_str("mapped-virtual"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// The semantic class used by category totals.  The domain remains a separate
/// key, so a host allocation and a device allocation are never combined here.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceCategory {
    Weights,
    KvCache,
    Activations,
    ScratchWorkspace,
    Staging,
    PinnedHost,
    MappedModelData,
    Other(String),
}

/// More specific resource kind retained on each record.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceKind {
    ModelMetadata,
    Weight(WeightScope),
    KvCache,
    Activation,
    ScratchWorkspace,
    Staging,
    PinnedHost,
    MappedModelData,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WeightScope {
    Embedding,
    Layer(LayerId),
    Output,
    Other(String),
}

impl ResourceKind {
    pub fn category(&self) -> ResourceCategory {
        match self {
            Self::Weight(_) => ResourceCategory::Weights,
            Self::KvCache => ResourceCategory::KvCache,
            Self::Activation => ResourceCategory::Activations,
            Self::ScratchWorkspace => ResourceCategory::ScratchWorkspace,
            Self::Staging => ResourceCategory::Staging,
            Self::PinnedHost => ResourceCategory::PinnedHost,
            Self::MappedModelData => ResourceCategory::MappedModelData,
            Self::ModelMetadata | Self::Other(_) => ResourceCategory::Other("other".to_string()),
        }
    }
}

/// Evidence attached to every byte measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementProvenance {
    GgufDescriptor,
    DerivedFromGgufType,
    TensorDataRepresentation,
    RuntimeMemoryBudget,
    RuntimeAllocator,
    ForgeCoreSafeApi,
    OperatingSystem,
    PlannerEstimate,
    CallerProvided,
    Derived(String),
    Unavailable(String),
}

impl MeasurementProvenance {
    pub fn description(&self) -> String {
        match self {
            Self::GgufDescriptor => "GGUF tensor descriptor".to_string(),
            Self::DerivedFromGgufType => "derived from GGUF shape and dtype".to_string(),
            Self::TensorDataRepresentation => "TensorData resident representation".to_string(),
            Self::RuntimeMemoryBudget => "RAMforge MemoryBudget".to_string(),
            Self::RuntimeAllocator => "runtime/backend allocator".to_string(),
            Self::ForgeCoreSafeApi => "ForgeCore safe API".to_string(),
            Self::OperatingSystem => "operating-system report".to_string(),
            Self::PlannerEstimate => "planner estimate".to_string(),
            Self::CallerProvided => "caller-provided measurement".to_string(),
            Self::Derived(detail) => format!("derived: {detail}"),
            Self::Unavailable(reason) => format!("unavailable: {reason}"),
        }
    }
}

/// Whether the numeric value is exact, estimated, or unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementCertainty {
    Exact,
    Estimated,
    Unknown,
}

/// A byte value that cannot represent an unavailable value as zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteMeasurement {
    bytes: Option<u64>,
    certainty: MeasurementCertainty,
    provenance: MeasurementProvenance,
}

impl Default for ByteMeasurement {
    fn default() -> Self {
        Self::unknown("measurement unavailable")
    }
}

impl ByteMeasurement {
    pub fn exact(bytes: u64, provenance: MeasurementProvenance) -> Self {
        Self {
            bytes: Some(bytes),
            certainty: MeasurementCertainty::Exact,
            provenance,
        }
    }

    pub fn estimated(bytes: u64, provenance: MeasurementProvenance) -> Self {
        Self {
            bytes: Some(bytes),
            certainty: MeasurementCertainty::Estimated,
            provenance,
        }
    }

    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            bytes: None,
            certainty: MeasurementCertainty::Unknown,
            provenance: MeasurementProvenance::Unavailable(reason.into()),
        }
    }

    pub fn unknown_with_provenance(provenance: MeasurementProvenance) -> Self {
        Self {
            bytes: None,
            certainty: MeasurementCertainty::Unknown,
            provenance,
        }
    }

    pub fn bytes(&self) -> Option<u64> {
        self.bytes
    }

    pub fn certainty(&self) -> MeasurementCertainty {
        self.certainty
    }

    pub fn provenance(&self) -> &MeasurementProvenance {
        &self.provenance
    }

    pub fn is_exact(&self) -> bool {
        self.certainty == MeasurementCertainty::Exact
    }

    pub fn is_estimated(&self) -> bool {
        self.certainty == MeasurementCertainty::Estimated
    }

    pub fn is_unknown(&self) -> bool {
        self.certainty == MeasurementCertainty::Unknown
    }
}

/// A domain/category total that keeps exact, estimated, and unknown
/// contributions separate.  `exact_bytes` is a lower-bound-like exact sum;
/// it is never presented as a complete total when unknown contributions exist.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ByteTotal {
    pub exact_bytes: u64,
    pub estimated_bytes: u64,
    pub unknown_contributions: u64,
    pub unknown_reasons: Vec<String>,
}

impl ByteTotal {
    pub fn is_exact(&self) -> bool {
        self.estimated_bytes == 0 && self.unknown_contributions == 0
    }

    pub fn is_complete(&self) -> bool {
        self.is_exact()
    }

    pub fn known_lower_bound_bytes(&self) -> u64 {
        self.exact_bytes
    }

    pub fn add_measurement(
        &mut self,
        measurement: &ByteMeasurement,
        label: impl Into<String>,
    ) -> Result<(), ResourceAccountingError> {
        let label = label.into();
        match (measurement.certainty, measurement.bytes) {
            (MeasurementCertainty::Exact, Some(bytes)) => {
                self.exact_bytes = self
                    .exact_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| ResourceAccountingError::ArithmeticOverflow(label.clone()))?;
            }
            (MeasurementCertainty::Estimated, Some(bytes)) => {
                self.estimated_bytes = self
                    .estimated_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| ResourceAccountingError::ArithmeticOverflow(label.clone()))?;
            }
            (MeasurementCertainty::Unknown, None) => {
                self.unknown_contributions = self
                    .unknown_contributions
                    .checked_add(1)
                    .ok_or_else(|| ResourceAccountingError::ArithmeticOverflow(label.clone()))?;
                self.unknown_reasons.push(format!(
                    "{} ({})",
                    label,
                    measurement.provenance.description()
                ));
            }
            // Constructors maintain this invariant.  Treat a malformed value
            // defensively as unknown rather than inventing a byte count.
            _ => {
                self.unknown_contributions = self
                    .unknown_contributions
                    .checked_add(1)
                    .ok_or_else(|| ResourceAccountingError::ArithmeticOverflow(label.clone()))?;
                self.unknown_reasons
                    .push(format!("{} (malformed measurement)", label));
            }
        }
        Ok(())
    }
}

/// A file-backed tensor range.  A range is source metadata, not a resident
/// allocation or an OS page-cache measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileBackedRange {
    pub model_id: ModelId,
    pub path: PathBuf,
    pub relative_offset: u64,
    pub file_offset: u64,
    pub byte_length: ByteMeasurement,
}

/// GGUF facts copied from a [`TensorDescriptor`], without adding a runtime
/// allocation interpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorMetadata {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub num_elements: u64,
    pub relative_offset: u64,
    pub file_offset: u64,
}

/// Logical tensor facts are separate from the GGUF range and from any
/// allocation.  A logical byte value can be estimated or unknown for a future
/// unsupported dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalTensorProperties {
    pub tensor_id: TensorId,
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    pub element_count: u64,
    pub logical_bytes: ByteMeasurement,
}

/// Source/representation information carried by a resource record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceOrigin {
    GgufTensor {
        metadata: Box<GgufTensorMetadata>,
        logical: Box<LogicalTensorProperties>,
        file_range: Box<FileBackedRange>,
    },
    Runtime {
        description: String,
        logical_bytes: ByteMeasurement,
    },
    Unmeasured {
        reason: String,
    },
}

impl ResourceOrigin {
    fn logical_bytes(&self) -> ByteMeasurement {
        match self {
            Self::GgufTensor { logical, .. } => logical.logical_bytes.clone(),
            Self::Runtime { logical_bytes, .. } => logical_bytes.clone(),
            Self::Unmeasured { reason } => ByteMeasurement::unknown(reason.clone()),
        }
    }

    fn file_range(&self) -> Option<&FileBackedRange> {
        match self {
            Self::GgufTensor { file_range, .. } => Some(file_range),
            Self::Runtime { .. } | Self::Unmeasured { .. } => None,
        }
    }
}

/// Ownership links a resource to a model, optional transformer layer, and
/// optional tensor.  Ownership is metadata; it does not imply residency.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceOwner {
    pub model_id: Option<ModelId>,
    pub layer_id: Option<LayerId>,
    pub tensor_id: Option<TensorId>,
}

/// Runtime/backend allocation facts.  `actual_bytes` and `resident_bytes` are
/// intentionally independent: a safe allocator/budget API may know a
/// reservation while not knowing OS residency or backend physical usage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationRecord {
    pub id: AllocationId,
    pub domain: MemoryDomain,
    pub backend: Option<BackendId>,
    pub requested_bytes: ByteMeasurement,
    pub actual_bytes: ByteMeasurement,
    pub alignment_bytes: Option<u64>,
    pub alignment_padding_bytes: ByteMeasurement,
    pub resident_bytes: ByteMeasurement,
}

impl AllocationRecord {
    /// Create an allocation and derive non-negative alignment padding when both
    /// requested and actual sizes are available.  Padding is reported
    /// separately and is not added again to `actual_bytes`.
    pub fn new(
        id: AllocationId,
        domain: MemoryDomain,
        requested_bytes: ByteMeasurement,
        actual_bytes: ByteMeasurement,
        resident_bytes: ByteMeasurement,
        alignment_bytes: Option<u64>,
        backend: Option<BackendId>,
    ) -> Self {
        let alignment_padding_bytes = alignment_padding(&requested_bytes, &actual_bytes);
        Self {
            id,
            domain,
            backend,
            requested_bytes,
            actual_bytes,
            alignment_bytes,
            alignment_padding_bytes,
            resident_bytes,
        }
    }

    /// Record the common case in which the requested amount is known but the
    /// backend/device allocation and residency are not exposed by the safe API.
    pub fn unmeasured(
        id: AllocationId,
        domain: MemoryDomain,
        requested_bytes: ByteMeasurement,
        alignment_bytes: Option<u64>,
        backend: Option<BackendId>,
        reason: impl Into<String>,
    ) -> Self {
        let actual_bytes = ByteMeasurement::unknown(reason.into());
        Self::new(
            id,
            domain,
            requested_bytes,
            actual_bytes,
            ByteMeasurement::unknown("resident bytes are not measured"),
            alignment_bytes,
            backend,
        )
    }
}

/// Compute allocator padding without ever turning an unknown value into zero.
pub fn alignment_padding(requested: &ByteMeasurement, actual: &ByteMeasurement) -> ByteMeasurement {
    let certainty = if requested.is_exact() && actual.is_exact() {
        MeasurementCertainty::Exact
    } else {
        MeasurementCertainty::Estimated
    };
    let (Some(requested), Some(actual)) = (requested.bytes(), actual.bytes()) else {
        return ByteMeasurement::unknown("alignment padding requires requested and actual bytes");
    };
    if actual < requested {
        return ByteMeasurement::unknown(format!(
            "actual allocation {actual} is smaller than requested allocation {requested}"
        ));
    }
    let provenance =
        MeasurementProvenance::Derived("actual minus requested allocation".to_string());
    match certainty {
        MeasurementCertainty::Exact => ByteMeasurement::exact(actual - requested, provenance),
        MeasurementCertainty::Estimated => {
            ByteMeasurement::estimated(actual - requested, provenance)
        }
        MeasurementCertainty::Unknown => unreachable!("certainty selected above"),
    }
}

/// A resource record can represent file-backed model data, a runtime buffer,
/// or an explicitly unmeasured object.  It may have no allocation yet; an
/// allocation with an unknown actual value must instead be represented by
/// `Some(AllocationRecord::unmeasured(...))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRecord {
    pub id: ResourceId,
    pub owner: ResourceOwner,
    pub kind: ResourceKind,
    pub source_domain: MemoryDomain,
    pub origin: ResourceOrigin,
    pub allocation: Option<AllocationRecord>,
}

impl ResourceRecord {
    pub fn from_descriptor(
        model_id: &ModelId,
        model_path: impl AsRef<Path>,
        descriptor: &TensorDescriptor,
    ) -> Self {
        let (kind, layer_id) = classify_tensor_name(&descriptor.name);
        Self::from_descriptor_with_kind(model_id, model_path, descriptor, kind, layer_id)
    }

    pub fn from_descriptor_with_kind(
        model_id: &ModelId,
        model_path: impl AsRef<Path>,
        descriptor: &TensorDescriptor,
        kind: ResourceKind,
        layer_id: Option<LayerId>,
    ) -> Self {
        let tensor_id = TensorId::from_model_and_name(model_id, &descriptor.name);
        let logical = LogicalTensorProperties {
            tensor_id: tensor_id.clone(),
            shape: descriptor.dimensions.clone(),
            ggml_type: descriptor.ggml_type,
            element_count: descriptor.num_elements,
            logical_bytes: measure_logical_tensor_bytes(
                descriptor.ggml_type,
                descriptor.num_elements,
            ),
        };
        let metadata = GgufTensorMetadata {
            name: descriptor.name.clone(),
            dimensions: descriptor.dimensions.clone(),
            ggml_type: descriptor.ggml_type,
            num_elements: descriptor.num_elements,
            relative_offset: descriptor.offset,
            file_offset: descriptor.file_offset,
        };
        let file_range = FileBackedRange {
            model_id: model_id.clone(),
            path: model_path.as_ref().to_path_buf(),
            relative_offset: descriptor.offset,
            file_offset: descriptor.file_offset,
            byte_length: match descriptor.byte_length {
                Some(bytes) => ByteMeasurement::exact(bytes, MeasurementProvenance::GgufDescriptor),
                None => ByteMeasurement::unknown(format!(
                    "GGUF byte length unavailable for tensor '{}'",
                    descriptor.name
                )),
            },
        };
        Self {
            id: ResourceId::for_tensor(&tensor_id),
            owner: ResourceOwner {
                model_id: Some(model_id.clone()),
                layer_id,
                tensor_id: Some(tensor_id),
            },
            kind,
            source_domain: MemoryDomain::Storage,
            origin: ResourceOrigin::GgufTensor {
                metadata: Box::new(metadata),
                logical: Box::new(logical),
                file_range: Box::new(file_range),
            },
            allocation: None,
        }
    }

    pub fn runtime(
        id: ResourceId,
        owner: ResourceOwner,
        kind: ResourceKind,
        source_domain: MemoryDomain,
        description: impl Into<String>,
        logical_bytes: ByteMeasurement,
        allocation: Option<AllocationRecord>,
    ) -> Self {
        Self {
            id,
            owner,
            kind,
            source_domain,
            origin: ResourceOrigin::Runtime {
                description: description.into(),
                logical_bytes,
            },
            allocation,
        }
    }

    pub fn unmeasured(
        id: ResourceId,
        owner: ResourceOwner,
        kind: ResourceKind,
        source_domain: MemoryDomain,
        reason: impl Into<String>,
        allocation: Option<AllocationRecord>,
    ) -> Self {
        Self {
            id,
            owner,
            kind,
            source_domain,
            origin: ResourceOrigin::Unmeasured {
                reason: reason.into(),
            },
            allocation,
        }
    }

    pub fn logical_bytes(&self) -> ByteMeasurement {
        self.origin.logical_bytes()
    }

    pub fn file_range(&self) -> Option<&FileBackedRange> {
        self.origin.file_range()
    }
}

/// Host and device budgets are separate configuration facts.  An unknown
/// budget is represented explicitly; no default zero means "no budget".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceBudgets {
    pub host_ram: ByteMeasurement,
    pub device_vram: BTreeMap<DeviceId, ByteMeasurement>,
}

impl ResourceBudgets {
    pub fn host_ram(bytes: u64) -> Self {
        Self {
            host_ram: ByteMeasurement::exact(bytes, MeasurementProvenance::RuntimeMemoryBudget),
            device_vram: BTreeMap::new(),
        }
    }

    pub fn with_device(mut self, device: DeviceId, bytes: u64) -> Self {
        self.device_vram.insert(
            device,
            ByteMeasurement::exact(bytes, MeasurementProvenance::CallerProvided),
        );
        self
    }

    pub fn host_ram_bytes(&self) -> Option<u64> {
        self.host_ram.bytes()
    }

    pub fn device_bytes(&self, device: &DeviceId) -> Option<u64> {
        self.device_vram
            .get(device)
            .and_then(ByteMeasurement::bytes)
    }

    pub fn from_memory_budget(budget: &MemoryBudget) -> Self {
        Self::host_ram(budget.total_bytes())
    }
}

/// The authoritative registry for Phase-3 representation records.  It only
/// validates identity and produces snapshots; it does not enforce budgets or
/// implement movement/residency policy.
#[derive(Debug, Clone, Default)]
pub struct ResourceAccounting {
    budgets: ResourceBudgets,
    resources: BTreeMap<ResourceId, ResourceRecord>,
    allocation_owners: BTreeMap<AllocationId, ResourceId>,
}

impl ResourceAccounting {
    pub fn new(budgets: ResourceBudgets) -> Self {
        Self {
            budgets,
            resources: BTreeMap::new(),
            allocation_owners: BTreeMap::new(),
        }
    }

    pub fn from_model(
        model: &GgufModel,
        model_id: ModelId,
        budgets: ResourceBudgets,
    ) -> Result<Self, ResourceAccountingError> {
        let mut accounting = Self::new(budgets);
        for descriptor in &model.tensors {
            accounting.insert(ResourceRecord::from_descriptor(
                &model_id,
                &model.path,
                descriptor,
            ))?;
        }
        Ok(accounting)
    }

    /// Build the same metadata-only registry from the existing file-backed
    /// datasource.  Calling this does not read tensor payloads or imply file
    /// residency; it reuses the datasource's parsed `GgufModel`.
    pub fn from_data_source(
        data_source: &GgufDataSource,
        model_id: ModelId,
        budgets: ResourceBudgets,
    ) -> Result<Self, ResourceAccountingError> {
        Self::from_model(data_source.model(), model_id, budgets)
    }

    pub fn budgets(&self) -> &ResourceBudgets {
        &self.budgets
    }

    pub fn insert(&mut self, record: ResourceRecord) -> Result<(), ResourceAccountingError> {
        if record.id.as_str().is_empty() {
            return Err(ResourceAccountingError::InvalidResource(
                "resource ID must not be empty".to_string(),
            ));
        }
        if self.resources.contains_key(&record.id) {
            return Err(ResourceAccountingError::DuplicateResourceId(record.id));
        }
        if let Some(allocation) = &record.allocation {
            if allocation.id.as_str().is_empty() {
                return Err(ResourceAccountingError::InvalidResource(
                    "allocation ID must not be empty".to_string(),
                ));
            }
            if self.allocation_owners.contains_key(&allocation.id) {
                return Err(ResourceAccountingError::DuplicateAllocationId(
                    allocation.id.clone(),
                ));
            }
        }

        if let Some(allocation) = &record.allocation {
            self.allocation_owners
                .insert(allocation.id.clone(), record.id.clone());
        }
        self.resources.insert(record.id.clone(), record);
        Ok(())
    }

    pub fn remove(&mut self, id: &ResourceId) -> Option<ResourceRecord> {
        let record = self.resources.remove(id)?;
        if let Some(allocation) = &record.allocation {
            self.allocation_owners.remove(&allocation.id);
        }
        Some(record)
    }

    pub fn get(&self, id: &ResourceId) -> Option<&ResourceRecord> {
        self.resources.get(id)
    }

    pub fn records(&self) -> impl Iterator<Item = &ResourceRecord> {
        self.resources.values()
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Produce a non-combining snapshot.  The operation is read-only and has
    /// no budget-enforcement side effects.
    pub fn snapshot(&self) -> Result<ResourceAccountingSnapshot, ResourceAccountingError> {
        let mut snapshot = ResourceAccountingSnapshot {
            resource_count: self.resources.len(),
            resource_ids: self.resources.keys().cloned().collect(),
            budgets: self.budgets.clone(),
            logical_model_bytes: ByteTotal::default(),
            file_backed_bytes: ByteTotal::default(),
            file_backed_bytes_by_domain: BTreeMap::new(),
            requested_allocation_bytes: BTreeMap::new(),
            actual_allocation_bytes: BTreeMap::new(),
            alignment_padding_bytes: BTreeMap::new(),
            resident_bytes: BTreeMap::new(),
            actual_by_category_and_domain: BTreeMap::new(),
        };

        for record in self.resources.values() {
            let logical = record.logical_bytes();
            if record.kind.category() == ResourceCategory::Weights {
                snapshot
                    .logical_model_bytes
                    .add_measurement(&logical, format!("logical model tensor '{}'", record.id))?;
            }

            if let Some(file_range) = record.file_range() {
                snapshot.file_backed_bytes.add_measurement(
                    &file_range.byte_length,
                    format!("file-backed range '{}'", record.id),
                )?;
                add_total(
                    &mut snapshot.file_backed_bytes_by_domain,
                    MemoryDomain::Storage,
                    &file_range.byte_length,
                    format!("storage range '{}'", record.id),
                )?;
            }

            let Some(allocation) = &record.allocation else {
                continue;
            };
            let domain = allocation.domain.clone();
            add_total(
                &mut snapshot.requested_allocation_bytes,
                domain.clone(),
                &allocation.requested_bytes,
                format!("requested allocation '{}'", record.id),
            )?;
            add_total(
                &mut snapshot.actual_allocation_bytes,
                domain.clone(),
                &allocation.actual_bytes,
                format!("actual allocation '{}'", record.id),
            )?;
            add_total(
                &mut snapshot.alignment_padding_bytes,
                domain.clone(),
                &allocation.alignment_padding_bytes,
                format!("allocation padding '{}'", record.id),
            )?;
            add_total(
                &mut snapshot.resident_bytes,
                domain.clone(),
                &allocation.resident_bytes,
                format!("resident allocation '{}'", record.id),
            )?;
            add_total(
                &mut snapshot.actual_by_category_and_domain,
                (domain, record.kind.category()),
                &allocation.actual_bytes,
                format!("category allocation '{}'", record.id),
            )?;
        }

        Ok(snapshot)
    }
}

fn add_total<K: Ord + Clone>(
    map: &mut BTreeMap<K, ByteTotal>,
    key: K,
    measurement: &ByteMeasurement,
    label: String,
) -> Result<(), ResourceAccountingError> {
    map.entry(key)
        .or_default()
        .add_measurement(measurement, label)
}

/// Read-only totals from [`ResourceAccounting::snapshot`].  Each map is
/// keyed by domain (and, for categories, by domain plus category), preventing
/// a storage/file value from being presented as host RAM or VRAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceAccountingSnapshot {
    pub resource_count: usize,
    pub resource_ids: Vec<ResourceId>,
    pub budgets: ResourceBudgets,
    pub logical_model_bytes: ByteTotal,
    pub file_backed_bytes: ByteTotal,
    pub file_backed_bytes_by_domain: BTreeMap<MemoryDomain, ByteTotal>,
    pub requested_allocation_bytes: BTreeMap<MemoryDomain, ByteTotal>,
    pub actual_allocation_bytes: BTreeMap<MemoryDomain, ByteTotal>,
    pub alignment_padding_bytes: BTreeMap<MemoryDomain, ByteTotal>,
    pub resident_bytes: BTreeMap<MemoryDomain, ByteTotal>,
    pub actual_by_category_and_domain: BTreeMap<(MemoryDomain, ResourceCategory), ByteTotal>,
}

impl ResourceAccountingSnapshot {
    pub fn host_actual(&self) -> Option<&ByteTotal> {
        self.actual_allocation_bytes.get(&MemoryDomain::HostRam)
    }

    pub fn device_actual(&self, device: &DeviceId) -> Option<&ByteTotal> {
        self.actual_allocation_bytes
            .get(&MemoryDomain::Device(device.clone()))
    }

    pub fn category_actual(
        &self,
        domain: &MemoryDomain,
        category: &ResourceCategory,
    ) -> Option<&ByteTotal> {
        self.actual_by_category_and_domain
            .get(&(domain.clone(), category.clone()))
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ResourceAccountingError {
    #[error("duplicate resource ID: {0}")]
    DuplicateResourceId(ResourceId),
    #[error("duplicate allocation ID: {0}")]
    DuplicateAllocationId(AllocationId),
    #[error("invalid resource: {0}")]
    InvalidResource(String),
    #[error("resource accounting arithmetic overflow: {0}")]
    ArithmeticOverflow(String),
}

/// Classify the same canonical tensor-name families used by the existing
/// layer grouping (`blk.<index>.*`).  Classification is metadata only.
pub fn classify_tensor_name(name: &str) -> (ResourceKind, Option<LayerId>) {
    if name == "token_embd.weight" || name.starts_with("token_embd.") {
        return (ResourceKind::Weight(WeightScope::Embedding), None);
    }
    if name == "output.weight" {
        return (ResourceKind::Weight(WeightScope::Output), None);
    }
    if name == "output_norm.weight" {
        return (
            ResourceKind::Weight(WeightScope::Other("persistent".to_string())),
            None,
        );
    }
    if let Some(layer_id) = layer_id_from_tensor_name(name) {
        return (
            ResourceKind::Weight(WeightScope::Layer(layer_id)),
            Some(layer_id),
        );
    }
    (
        ResourceKind::Weight(WeightScope::Other("model".to_string())),
        None,
    )
}

fn layer_id_from_tensor_name(name: &str) -> Option<LayerId> {
    let rest = name.strip_prefix("blk.")?;
    let (number, suffix) = rest.split_once('.')?;
    if suffix.is_empty() {
        return None;
    }
    number.parse::<u64>().ok().map(LayerId::new)
}

/// Compute the logical GGUF representation size from dtype and element count.
/// Unsupported types and malformed quantized counts remain unknown.  The
/// existing type table marks best-effort/extension entries as estimated rather
/// than presenting them as exact backend allocations.
pub fn measure_logical_tensor_bytes(ggml_type: GgmlType, element_count: u64) -> ByteMeasurement {
    if element_count == 0 {
        return ByteMeasurement::exact(0, MeasurementProvenance::DerivedFromGgufType);
    }

    let Some((block_size, type_size)) = ggml_type.type_info() else {
        return ByteMeasurement::unknown(format!(
            "logical size unavailable for GGML type {}",
            ggml_type.name()
        ));
    };
    if block_size == 0 {
        return ByteMeasurement::unknown(format!(
            "invalid zero block size for GGML type {}",
            ggml_type.name()
        ));
    }
    if block_size > 1 && !element_count.is_multiple_of(block_size) {
        return ByteMeasurement::unknown(format!(
            "{} elements is not a complete {}-element block for {}",
            element_count,
            block_size,
            ggml_type.name()
        ));
    }
    let blocks = element_count / block_size;
    let Some(bytes) = blocks.checked_mul(type_size) else {
        return ByteMeasurement::unknown(format!(
            "logical size overflow for {} elements of {}",
            element_count,
            ggml_type.name()
        ));
    };

    let estimated = matches!(
        ggml_type,
        GgmlType::Q8_1
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ3_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ4_NL
            | GgmlType::IQ3_S
            | GgmlType::IQ2_S
            | GgmlType::IQ4_XS
            | GgmlType::IQ1_M
            | GgmlType::TQ1_0
            | GgmlType::TQ2_0
            | GgmlType::MXFP4
    );
    if estimated {
        ByteMeasurement::estimated(bytes, MeasurementProvenance::DerivedFromGgufType)
    } else {
        ByteMeasurement::exact(bytes, MeasurementProvenance::DerivedFromGgufType)
    }
}

/// Compute an element count without wrapping on malformed dimensions.
pub fn checked_element_count(dimensions: &[u64]) -> Result<u64, ResourceAccountingError> {
    dimensions.iter().try_fold(1u64, |count, dimension| {
        count.checked_mul(*dimension).ok_or_else(|| {
            ResourceAccountingError::ArithmeticOverflow("tensor element count".to_string())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn descriptor(
        name: &str,
        dimensions: Vec<u64>,
        ggml_type: GgmlType,
        offset: u64,
        file_offset: u64,
        byte_length: Option<u64>,
    ) -> TensorDescriptor {
        let num_elements = checked_element_count(&dimensions).unwrap();
        TensorDescriptor {
            name: name.to_string(),
            dimensions,
            ggml_type,
            offset,
            file_offset,
            byte_length,
            num_elements,
        }
    }

    fn model_with_descriptors(tensors: Vec<TensorDescriptor>) -> GgufModel {
        GgufModel {
            path: PathBuf::from("/models/example.gguf"),
            file_size: 4096,
            version: 3,
            metadata: BTreeMap::new(),
            tensors,
            alignment: 32,
            data_start_offset: 128,
        }
    }

    #[test]
    fn tensor_sizing_covers_float_quantized_zero_and_unknown_types() {
        assert_eq!(
            measure_logical_tensor_bytes(GgmlType::F32, 10).bytes(),
            Some(40)
        );
        assert_eq!(
            measure_logical_tensor_bytes(GgmlType::F16, 10).bytes(),
            Some(20)
        );
        assert_eq!(
            measure_logical_tensor_bytes(GgmlType::Q4_0, 32).bytes(),
            Some(18)
        );
        assert_eq!(
            measure_logical_tensor_bytes(GgmlType::Q4_0, 31).bytes(),
            None
        );
        assert_eq!(
            measure_logical_tensor_bytes(GgmlType::Unknown(999), 10).bytes(),
            None
        );
        assert_eq!(
            measure_logical_tensor_bytes(GgmlType::Unknown(999), 0).bytes(),
            Some(0)
        );
        assert_eq!(
            measure_logical_tensor_bytes(GgmlType::Q8_1, 32).certainty(),
            MeasurementCertainty::Estimated
        );
    }

    #[test]
    fn file_range_preserves_relative_absolute_and_unknown_length() {
        let model_id = ModelId::new("model-A");
        let known = descriptor("known", vec![4], GgmlType::F32, 64, 192, Some(16));
        let unknown = descriptor("unknown", vec![4], GgmlType::Unknown(777), 96, 224, None);
        let known_record =
            ResourceRecord::from_descriptor(&model_id, "/models/example.gguf", &known);
        let unknown_record =
            ResourceRecord::from_descriptor(&model_id, "/models/example.gguf", &unknown);

        let Some(ResourceOrigin::GgufTensor {
            metadata,
            file_range,
            ..
        }) = Some(&known_record.origin)
        else {
            panic!("expected GGUF origin");
        };
        assert_eq!(metadata.relative_offset, 64);
        assert_eq!(metadata.file_offset, 192);
        assert_eq!(file_range.relative_offset, 64);
        assert_eq!(file_range.file_offset, 192);
        assert_eq!(file_range.byte_length.bytes(), Some(16));
        assert!(unknown_record
            .file_range()
            .unwrap()
            .byte_length
            .is_unknown());
    }

    #[test]
    fn classification_covers_persistent_layers_and_other_weights() {
        assert_eq!(
            classify_tensor_name("token_embd.weight"),
            (ResourceKind::Weight(WeightScope::Embedding), None)
        );
        assert_eq!(
            classify_tensor_name("output.weight"),
            (ResourceKind::Weight(WeightScope::Output), None)
        );
        assert_eq!(
            classify_tensor_name("blk.7.attn_q.weight"),
            (
                ResourceKind::Weight(WeightScope::Layer(LayerId::new(7))),
                Some(LayerId::new(7))
            )
        );
        assert_eq!(
            classify_tensor_name("output_norm.weight"),
            (
                ResourceKind::Weight(WeightScope::Other("persistent".to_string())),
                None
            )
        );
        assert_eq!(
            classify_tensor_name("adapter.weight"),
            (
                ResourceKind::Weight(WeightScope::Other("model".to_string())),
                None
            )
        );
    }

    #[test]
    fn domains_and_categories_are_separate_in_snapshot() {
        let mut accounting = ResourceAccounting::new(
            ResourceBudgets::host_ram(1024).with_device(DeviceId::new("gpu0"), 2048),
        );
        let host = ResourceRecord::runtime(
            ResourceId::new("kv-host"),
            ResourceOwner::default(),
            ResourceKind::KvCache,
            MemoryDomain::HostRam,
            "KV cache",
            ByteMeasurement::exact(100, MeasurementProvenance::CallerProvided),
            Some(AllocationRecord::new(
                AllocationId::new("alloc-kv-host"),
                MemoryDomain::HostRam,
                ByteMeasurement::exact(100, MeasurementProvenance::RuntimeMemoryBudget),
                ByteMeasurement::exact(112, MeasurementProvenance::RuntimeAllocator),
                ByteMeasurement::unknown("resident not measured"),
                Some(16),
                None,
            )),
        );
        let device = ResourceRecord::runtime(
            ResourceId::new("activation-gpu"),
            ResourceOwner::default(),
            ResourceKind::Activation,
            MemoryDomain::Device(DeviceId::new("gpu0")),
            "device activation",
            ByteMeasurement::unknown("logical activation shape is dynamic"),
            Some(AllocationRecord::unmeasured(
                AllocationId::new("alloc-activation-gpu"),
                MemoryDomain::Device(DeviceId::new("gpu0")),
                ByteMeasurement::estimated(256, MeasurementProvenance::PlannerEstimate),
                Some(256),
                Some(BackendId::new("backend-gpu0")),
                "ForgeCore does not expose backend allocation bytes",
            )),
        );
        let mapped = ResourceRecord::runtime(
            ResourceId::new("mapped-model"),
            ResourceOwner::default(),
            ResourceKind::MappedModelData,
            MemoryDomain::Storage,
            "model mapping",
            ByteMeasurement::exact(4096, MeasurementProvenance::GgufDescriptor),
            Some(AllocationRecord::new(
                AllocationId::new("alloc-mapped"),
                MemoryDomain::MappedVirtual,
                ByteMeasurement::exact(4096, MeasurementProvenance::CallerProvided),
                ByteMeasurement::exact(4096, MeasurementProvenance::RuntimeAllocator),
                ByteMeasurement::unknown("mapped residency not measured"),
                None,
                None,
            )),
        );
        accounting.insert(host).unwrap();
        accounting.insert(device).unwrap();
        accounting.insert(mapped).unwrap();

        let snapshot = accounting.snapshot().unwrap();
        assert_eq!(snapshot.host_actual().unwrap().exact_bytes, 112);
        assert_eq!(
            snapshot
                .device_actual(&DeviceId::new("gpu0"))
                .unwrap()
                .unknown_contributions,
            1
        );
        assert_eq!(
            snapshot
                .device_actual(&DeviceId::new("gpu0"))
                .unwrap()
                .exact_bytes,
            0
        );
        assert_eq!(
            snapshot
                .actual_allocation_bytes
                .get(&MemoryDomain::MappedVirtual)
                .unwrap()
                .exact_bytes,
            4096
        );
        assert!(!snapshot
            .actual_allocation_bytes
            .contains_key(&MemoryDomain::Storage));
        assert_eq!(snapshot.logical_model_bytes.exact_bytes, 0);
    }

    #[test]
    fn requested_actual_padding_and_residency_keep_distinct_provenance() {
        let allocation = AllocationRecord::new(
            AllocationId::new("alloc"),
            MemoryDomain::HostRam,
            ByteMeasurement::exact(100, MeasurementProvenance::RuntimeMemoryBudget),
            ByteMeasurement::exact(128, MeasurementProvenance::RuntimeAllocator),
            ByteMeasurement::unknown("OS residency not measured"),
            Some(64),
            None,
        );
        assert_eq!(allocation.requested_bytes.bytes(), Some(100));
        assert_eq!(allocation.actual_bytes.bytes(), Some(128));
        assert_eq!(allocation.alignment_padding_bytes.bytes(), Some(28));
        assert!(allocation.resident_bytes.is_unknown());
    }

    #[test]
    fn snapshot_preserves_unknown_and_estimated_values() {
        let mut accounting = ResourceAccounting::new(ResourceBudgets::default());
        let record = ResourceRecord::runtime(
            ResourceId::new("scratch"),
            ResourceOwner::default(),
            ResourceKind::ScratchWorkspace,
            MemoryDomain::Unknown,
            "graph scratch",
            ByteMeasurement::unknown("graph dimensions unavailable"),
            Some(AllocationRecord::new(
                AllocationId::new("scratch-allocation"),
                MemoryDomain::Unknown,
                ByteMeasurement::estimated(512, MeasurementProvenance::PlannerEstimate),
                ByteMeasurement::unknown("backend workspace not exposed"),
                ByteMeasurement::unknown("resident workspace not exposed"),
                None,
                Some(BackendId::new("forgecore")),
            )),
        );
        accounting.insert(record).unwrap();
        let snapshot = accounting.snapshot().unwrap();
        let total = snapshot
            .actual_allocation_bytes
            .get(&MemoryDomain::Unknown)
            .unwrap();
        assert_eq!(total.exact_bytes, 0);
        assert_eq!(total.estimated_bytes, 0);
        assert_eq!(total.unknown_contributions, 1);
        assert_eq!(snapshot.logical_model_bytes.unknown_contributions, 0);
        assert_eq!(snapshot.budgets.host_ram_bytes(), None);
    }

    #[test]
    fn budgets_preserve_host_device_separation_and_memory_budget_compatibility() {
        let memory_budget = MemoryBudget::new(4096).unwrap();
        let budgets = ResourceBudgets::from_memory_budget(&memory_budget)
            .with_device(DeviceId::new("gpu0"), 8192);
        assert_eq!(budgets.host_ram_bytes(), Some(4096));
        assert_eq!(budgets.device_bytes(&DeviceId::new("gpu0")), Some(8192));
        assert!(budgets.device_bytes(&DeviceId::new("gpu1")).is_none());
    }

    #[test]
    fn ids_are_deterministic_and_do_not_depend_on_insertion_order() {
        let model = ModelId::new("model-A");
        let tensor_a = TensorId::from_model_and_name(&model, "blk.0.attn_q.weight");
        let tensor_b = TensorId::from_model_and_name(&model, "blk.0.attn_q.weight");
        assert_eq!(tensor_a, tensor_b);
        assert_eq!(
            ResourceId::for_tensor(&tensor_a),
            ResourceId::for_tensor(&tensor_b)
        );
        assert_eq!(
            AllocationId::for_resource(&ResourceId::for_tensor(&tensor_a)),
            AllocationId::for_resource(&ResourceId::for_tensor(&tensor_b))
        );
    }

    #[test]
    fn duplicate_allocations_are_rejected_and_logical_file_actual_are_not_added() {
        let model_id = ModelId::new("model-A");
        let tensor = descriptor("weight", vec![4], GgmlType::F32, 32, 160, Some(16));
        let mut first = ResourceRecord::from_descriptor(&model_id, "/models/a.gguf", &tensor);
        first.allocation = Some(AllocationRecord::new(
            AllocationId::new("same-allocation"),
            MemoryDomain::HostRam,
            ByteMeasurement::exact(16, MeasurementProvenance::RuntimeMemoryBudget),
            ByteMeasurement::exact(16, MeasurementProvenance::RuntimeAllocator),
            ByteMeasurement::unknown("not measured"),
            None,
            None,
        ));
        let mut accounting = ResourceAccounting::new(ResourceBudgets::host_ram(1024));
        accounting.insert(first).unwrap();

        let mut second = ResourceRecord::runtime(
            ResourceId::new("alias"),
            ResourceOwner::default(),
            ResourceKind::Staging,
            MemoryDomain::HostRam,
            "alias of first allocation",
            ByteMeasurement::exact(16, MeasurementProvenance::CallerProvided),
            Some(AllocationRecord::new(
                AllocationId::new("same-allocation"),
                MemoryDomain::HostRam,
                ByteMeasurement::exact(16, MeasurementProvenance::RuntimeMemoryBudget),
                ByteMeasurement::exact(16, MeasurementProvenance::RuntimeAllocator),
                ByteMeasurement::unknown("not measured"),
                None,
                None,
            )),
        );
        assert!(matches!(
            accounting.insert(second.clone()),
            Err(ResourceAccountingError::DuplicateAllocationId(_))
        ));
        second.allocation = None;
        accounting.insert(second).unwrap();

        let snapshot = accounting.snapshot().unwrap();
        // The file/logical tensor bytes and actual host bytes are separate
        // ledgers.  The staging alias has no allocation, so it adds neither.
        assert_eq!(snapshot.logical_model_bytes.exact_bytes, 16);
        assert_eq!(snapshot.file_backed_bytes.exact_bytes, 16);
        assert_eq!(snapshot.host_actual().unwrap().exact_bytes, 16);
    }

    #[test]
    fn model_builder_keeps_tensor_ownership_and_ids_stable() {
        let model = model_with_descriptors(vec![
            descriptor(
                "token_embd.weight",
                vec![8, 4],
                GgmlType::F32,
                0,
                128,
                Some(128),
            ),
            descriptor(
                "blk.2.attn_q.weight",
                vec![4, 4],
                GgmlType::Q4_0,
                128,
                256,
                Some(9),
            ),
        ]);
        let model_id = ModelId::from_path(&model.path);
        let accounting = ResourceAccounting::from_model(
            &model,
            model_id.clone(),
            ResourceBudgets::host_ram(1_000),
        )
        .unwrap();
        assert_eq!(accounting.len(), 2);
        let ids = accounting
            .records()
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        assert_eq!(ids, {
            let mut sorted = ids.clone();
            sorted.sort();
            sorted
        });
        let layer_record = accounting
            .records()
            .find(|record| record.owner.layer_id == Some(LayerId::new(2)))
            .unwrap();
        assert_eq!(layer_record.owner.model_id, Some(model_id));
        assert_eq!(
            layer_record.owner.tensor_id.as_ref().unwrap().as_str(),
            "model=/models/example.gguf;tensor=blk.2.attn_q.weight"
        );
    }
}
