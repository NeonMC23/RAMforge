//! Static device/backend discovery and execution selection.
//!
//! This module is intentionally separate from resource policy and from the
//! legacy numerical backend. It snapshots the safe ForgeCore device registry,
//! validates an explicit selection, and describes the execution options that
//! the current ForgeCore adapter can actually consume.
//!
//! The verified ForgeCore revision exposes safe device enumeration and static
//! model-load/offload options. RAMforge intentionally does not enable those
//! non-CPU options in this refresh: non-CPU devices are discoverable when the
//! native build exposes them, but remain non-selectable by this adapter. No
//! CPU fallback is performed.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use forge_core::{enumerate_devices, DeviceInfo as ForgeDeviceInfo, DeviceType};
use ramforge_core::{
    BackendId, ByteMeasurement, DeviceId, MeasurementProvenance, MemoryDomain,
};

/// Execution/device kind reported by the safe ForgeCore registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Meta,
    Unknown(i32),
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.label())
    }
}

impl DeviceKind {
    fn from_forgecore(kind: DeviceType) -> Self {
        match kind {
            DeviceType::Cpu => Self::Cpu,
            DeviceType::Gpu => Self::Gpu,
            DeviceType::Igpu => Self::IntegratedGpu,
            DeviceType::Accel => Self::Accelerator,
            DeviceType::Meta => Self::Meta,
            DeviceType::Unknown(value) => Self::Unknown(value),
        }
    }

    fn label(self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Gpu => "gpu".to_string(),
            Self::IntegratedGpu => "igpu".to_string(),
            Self::Accelerator => "accelerator".to_string(),
            Self::Meta => "meta".to_string(),
            Self::Unknown(value) => format!("unknown-{value}"),
        }
    }

    pub fn is_device_memory(self) -> bool {
        matches!(self, Self::Gpu | Self::IntegratedGpu | Self::Accelerator)
    }
}

/// Whether the current safe API supports one capability. Unknown is distinct
/// from unsupported: it means the native/API fact was not exposed safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityState {
    Supported,
    Unsupported,
    Unknown,
}

impl fmt::Display for CapabilityState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Supported => "supported",
            Self::Unsupported => "unsupported",
            Self::Unknown => "unknown",
        })
    }
}

/// Capabilities relevant to static execution selection. These describe the
/// active RAMforge adapter contract, not proof that upstream can satisfy every
/// possible operation, and are not proof that a resource has been allocated or
/// is resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub backend_initialization: CapabilityState,
    pub model_execution: CapabilityState,
    pub static_model_offload: CapabilityState,
    pub memory_reporting: CapabilityState,
    pub cpu_thread_configuration: CapabilityState,
}

/// A stable-enough, owned device snapshot. The native registry index is kept
/// for selection, while `id` is used for diagnostics and accounting domains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub backend_id: BackendId,
    pub kind: DeviceKind,
    pub registry_index: Option<usize>,
    pub name: String,
    pub description: String,
    /// The verified ForgeCore device API does not expose a vendor field.
    pub vendor: Option<String>,
    pub memory_total: ByteMeasurement,
    pub memory_free: ByteMeasurement,
    pub capabilities: DeviceCapabilities,
    /// Present in the native registry, independent of model-execution support.
    pub available: bool,
    /// A device is selectable only when RAMforge's current ForgeCore adapter
    /// can actually execute the model on it.
    pub selectable: bool,
    pub selection_unavailable_reason: Option<String>,
}

impl DeviceInfo {
    pub fn accounting_domain(&self) -> MemoryDomain {
        match self.kind {
            DeviceKind::Cpu => MemoryDomain::HostRam,
            DeviceKind::Gpu | DeviceKind::IntegratedGpu | DeviceKind::Accelerator => {
                MemoryDomain::Device(self.id.clone())
            }
            DeviceKind::Meta | DeviceKind::Unknown(_) => MemoryDomain::Unknown,
        }
    }

    pub fn is_cpu(&self) -> bool {
        self.kind == DeviceKind::Cpu
    }

    pub fn display_label(&self) -> String {
        format!("{} ({})", self.name, self.id)
    }
}

/// A static selection request. It never means "try another device".
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceSelection {
    Cpu,
    ById(DeviceId),
    BackendIndex { backend: BackendId, index: usize },
}

impl DeviceSelection {
    pub fn by_id(id: impl Into<DeviceId>) -> Self {
        Self::ById(id.into())
    }

    pub fn backend_index(backend: impl Into<BackendId>, index: usize) -> Self {
        Self::BackendIndex {
            backend: backend.into(),
            index,
        }
    }
}

impl fmt::Display for DeviceSelection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::ById(id) => write!(formatter, "id:{id}"),
            Self::BackendIndex { backend, index } => {
                write!(formatter, "backend:{backend}:{index}")
            }
        }
    }
}

impl FromStr for DeviceSelection {
    type Err = DeviceSelectionError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if let Some(id) = value.strip_prefix("id:") {
            if id.is_empty() {
                return Err(DeviceSelectionError::InvalidSyntax(
                    "id: requires a non-empty device ID".to_string(),
                ));
            }
            return Ok(Self::ById(DeviceId::new(id)));
        }
        if let Some(value) = value.strip_prefix("backend:") {
            let Some((backend, index)) = value.rsplit_once(':') else {
                return Err(DeviceSelectionError::InvalidSyntax(
                    "backend selection must be backend:<backend-id>:<index>".to_string(),
                ));
            };
            if backend.is_empty() {
                return Err(DeviceSelectionError::InvalidSyntax(
                    "backend ID must not be empty".to_string(),
                ));
            }
            let index = index.parse::<usize>().map_err(|_| {
                DeviceSelectionError::InvalidSyntax(
                    "device index must be a non-negative integer".to_string(),
                )
            })?;
            return Ok(Self::BackendIndex {
                backend: BackendId::new(backend),
                index,
            });
        }
        Err(DeviceSelectionError::InvalidSyntax(
            "expected cpu, id:<device-id>, or backend:<backend-id>:<index>".to_string(),
        ))
    }
}

/// A validated selection. It contains a snapshot, not a native handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedDevice {
    selection: DeviceSelection,
    info: DeviceInfo,
}

impl SelectedDevice {
    pub fn selection(&self) -> &DeviceSelection {
        &self.selection
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn accounting_domain(&self) -> MemoryDomain {
        self.info.accounting_domain()
    }
}

/// Registry errors are explicit and never trigger fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSelectionError {
    InvalidSyntax(String),
    CpuUnavailable,
    BackendDeviceUnavailable { backend: BackendId, index: usize },
    DeviceIdUnavailable(DeviceId),
    DeviceNotSelectable { id: DeviceId, reason: String },
}

impl fmt::Display for DeviceSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSyntax(message) => write!(formatter, "invalid device selection: {message}"),
            Self::CpuUnavailable => formatter.write_str("CPU device is unavailable"),
            Self::BackendDeviceUnavailable { backend, index } => write!(
                formatter,
                "backend device is unavailable: backend={backend}, index={index}"
            ),
            Self::DeviceIdUnavailable(id) => write!(formatter, "device ID is unavailable: {id}"),
            Self::DeviceNotSelectable { id, reason } => {
                write!(formatter, "device {id} is not selectable: {reason}")
            }
        }
    }
}

impl std::error::Error for DeviceSelectionError {}

/// Read-only snapshot of ForgeCore's safe native device registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRegistry {
    devices: Vec<DeviceInfo>,
}

impl DeviceRegistry {
    /// Discover exactly what the verified ForgeCore native registry exposes.
    /// This performs no model load and no network access.
    pub fn discover() -> Self {
        let devices = enumerate_devices()
            .into_iter()
            .map(device_info_from_forgecore)
            .collect();
        Self::from_devices(devices)
    }

    /// CPU-only registry used by the existing default CPU adapter path. It is
    /// not a claim that another native backend exists.
    pub fn cpu_only() -> Self {
        Self::from_devices(vec![cpu_device_info()])
    }

    pub fn from_devices(mut devices: Vec<DeviceInfo>) -> Self {
        devices.sort_by(|left, right| left.id.cmp(&right.id));
        Self { devices }
    }

    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    pub fn selectable_devices(&self) -> impl Iterator<Item = &DeviceInfo> {
        self.devices.iter().filter(|device| device.selectable)
    }

    pub fn backend_ids(&self) -> Vec<BackendId> {
        let mut ids = BTreeSet::new();
        for device in &self.devices {
            ids.insert(device.backend_id.clone());
        }
        ids.into_iter().collect()
    }

    pub fn find_by_id(&self, id: &DeviceId) -> Option<&DeviceInfo> {
        self.devices.iter().find(|device| &device.id == id)
    }

    pub fn select(&self, selection: &DeviceSelection) -> Result<SelectedDevice, DeviceSelectionError> {
        let info = match selection {
            DeviceSelection::Cpu => self
                .devices
                .iter()
                .find(|device| device.is_cpu())
                .ok_or(DeviceSelectionError::CpuUnavailable)?,
            DeviceSelection::ById(id) => self
                .find_by_id(id)
                .ok_or_else(|| DeviceSelectionError::DeviceIdUnavailable(id.clone()))?,
            DeviceSelection::BackendIndex { backend, index } => self
                .devices
                .iter()
                .find(|device| {
                    device.backend_id == *backend && device.registry_index == Some(*index)
                })
                .ok_or_else(|| DeviceSelectionError::BackendDeviceUnavailable {
                    backend: backend.clone(),
                    index: *index,
                })?,
        };
        if !info.available || !info.selectable {
            return Err(DeviceSelectionError::DeviceNotSelectable {
                id: info.id.clone(),
                reason: info
                    .selection_unavailable_reason
                    .clone()
                    .unwrap_or_else(|| "native execution is unavailable".to_string()),
            });
        }
        Ok(SelectedDevice {
            selection: selection.clone(),
            info: info.clone(),
        })
    }
}

/// Execution options consumed by the current ForgeCore adapter. Unsupported
/// future placement/offload policy is intentionally absent rather than guessed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionConfig {
    pub device: DeviceSelection,
    pub cpu_thread_count: usize,
    pub context_length: Option<u32>,
    pub check_tensors: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self::cpu_default()
    }
}

impl ExecutionConfig {
    pub fn cpu_default() -> Self {
        Self {
            device: DeviceSelection::Cpu,
            cpu_thread_count: 1,
            context_length: None,
            check_tensors: false,
        }
    }

    pub fn new(
        device: DeviceSelection,
        cpu_thread_count: usize,
        context_length: Option<u32>,
        check_tensors: bool,
    ) -> Result<Self, ExecutionConfigError> {
        let config = Self {
            device,
            cpu_thread_count,
            context_length,
            check_tensors,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ExecutionConfigError> {
        if self.cpu_thread_count == 0 {
            return Err(ExecutionConfigError::ZeroCpuThreads);
        }
        if self.context_length == Some(0) {
            return Err(ExecutionConfigError::ZeroContextLength);
        }
        Ok(())
    }

    pub fn resolve<'a>(
        &self,
        registry: &'a DeviceRegistry,
    ) -> Result<SelectedDevice, ExecutionConfigError> {
        self.validate()?;
        registry
            .select(&self.device)
            .map_err(ExecutionConfigError::DeviceSelection)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionConfigError {
    ZeroCpuThreads,
    ZeroContextLength,
    DeviceSelection(DeviceSelectionError),
}

impl fmt::Display for ExecutionConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCpuThreads => formatter.write_str("CPU thread count must be greater than zero"),
            Self::ZeroContextLength => formatter.write_str("context length must be greater than zero"),
            Self::DeviceSelection(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ExecutionConfigError {}

fn cpu_device_info() -> DeviceInfo {
    DeviceInfo {
        id: DeviceId::new("cpu"),
        backend_id: BackendId::new("cpu"),
        kind: DeviceKind::Cpu,
        registry_index: None,
        name: "CPU".to_string(),
        description: "ForgeCore CPU model/context execution".to_string(),
        vendor: None,
        memory_total: ByteMeasurement::unknown("ForgeCore CPU device memory is not reported"),
        memory_free: ByteMeasurement::unknown("ForgeCore CPU device memory is not reported"),
        capabilities: DeviceCapabilities {
            backend_initialization: CapabilityState::Supported,
            model_execution: CapabilityState::Supported,
            static_model_offload: CapabilityState::Unsupported,
            memory_reporting: CapabilityState::Unknown,
            cpu_thread_configuration: CapabilityState::Supported,
        },
        available: true,
        selectable: true,
        selection_unavailable_reason: None,
    }
}

fn device_info_from_forgecore(info: ForgeDeviceInfo) -> DeviceInfo {
    let kind = DeviceKind::from_forgecore(info.device_type);
    let component = stable_component(&info.name);
    let backend_id = if kind == DeviceKind::Cpu {
        BackendId::new("cpu")
    } else {
        BackendId::new(format!("forgecore-{}-{component}", kind.label()))
    };
    let id = if kind == DeviceKind::Cpu {
        DeviceId::new("cpu")
    } else {
        DeviceId::new(format!(
            "forgecore/{}/{}/{}",
            kind.label(), info.index, component
        ))
    };
    let memory_reporting = if info.memory_total > 0 || info.memory_free > 0 {
        CapabilityState::Supported
    } else {
        CapabilityState::Unknown
    };
    let model_execution = if kind == DeviceKind::Cpu {
        CapabilityState::Supported
    } else {
        CapabilityState::Unsupported
    };
    let (selectable, reason) = if kind == DeviceKind::Cpu {
        (true, None)
    } else {
        (
            false,
            Some(
                "RAMforge does not enable non-CPU model binding in this refresh"
                    .to_string(),
            ),
        )
    };
    DeviceInfo {
        id,
        backend_id,
        kind,
        registry_index: Some(info.index),
        name: info.name,
        description: info.description,
        vendor: None,
        memory_total: native_memory_measurement(info.memory_total, "total"),
        memory_free: native_memory_measurement(info.memory_free, "free"),
        capabilities: DeviceCapabilities {
            backend_initialization: CapabilityState::Supported,
            model_execution,
            static_model_offload: CapabilityState::Unsupported,
            memory_reporting,
            cpu_thread_configuration: if kind == DeviceKind::Cpu {
                CapabilityState::Supported
            } else {
                CapabilityState::Unsupported
            },
        },
        available: true,
        selectable,
        selection_unavailable_reason: reason,
    }
}

fn native_memory_measurement(bytes: usize, label: &str) -> ByteMeasurement {
    if bytes == 0 {
        ByteMeasurement::unknown(format!(
            "ForgeCore native device {label} memory value is unavailable"
        ))
    } else {
        ByteMeasurement::exact(bytes as u64, MeasurementProvenance::ForgeCoreSafeApi)
    }
}

fn stable_component(value: &str) -> String {
    let mut output = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
        } else if !output.ends_with('-') {
            output.push('-');
        }
    }
    while output.ends_with('-') {
        output.pop();
    }
    if output.is_empty() {
        "unknown".to_string()
    } else {
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> DeviceInfo {
        cpu_device_info()
    }

    fn gpu() -> DeviceInfo {
        device_info_from_forgecore(ForgeDeviceInfo {
            index: 3,
            name: "CUDA0".to_string(),
            description: "test CUDA device".to_string(),
            device_type: DeviceType::Gpu,
            memory_free: 4096,
            memory_total: 8192,
        })
    }

    #[test]
    fn cpu_is_first_class_and_has_host_domain() {
        let registry = DeviceRegistry::cpu_only();
        let selected = registry.select(&DeviceSelection::Cpu).unwrap();
        assert_eq!(selected.info().id, DeviceId::new("cpu"));
        assert_eq!(selected.info().backend_id, BackendId::new("cpu"));
        assert_eq!(selected.accounting_domain(), MemoryDomain::HostRam);
        assert!(selected.info().selectable);
    }

    #[test]
    fn stable_device_identity_is_derived_from_native_facts() {
        let left = gpu();
        let right = gpu();
        assert_eq!(left.id, right.id);
        assert_eq!(left.backend_id, right.backend_id);
        assert_eq!(left.accounting_domain(), MemoryDomain::Device(left.id.clone()));
    }

    #[test]
    fn device_capabilities_preserve_known_and_unknown_memory() {
        let gpu = gpu();
        assert_eq!(gpu.memory_total.bytes(), Some(8192));
        assert_eq!(gpu.memory_free.bytes(), Some(4096));
        assert_eq!(gpu.capabilities.model_execution, CapabilityState::Unsupported);
        assert!(!gpu.selectable);
        assert!(cpu().memory_total.is_unknown());
    }

    #[test]
    fn explicit_selection_does_not_fallback() {
        let registry = DeviceRegistry::from_devices(vec![cpu(), gpu()]);
        let error = registry
            .select(&DeviceSelection::ById(DeviceId::new("missing")))
            .unwrap_err();
        assert_eq!(
            error,
            DeviceSelectionError::DeviceIdUnavailable(DeviceId::new("missing"))
        );
        let gpu_info = gpu();
        let error = registry
            .select(&DeviceSelection::ById(gpu_info.id.clone()))
            .unwrap_err();
        assert!(matches!(error, DeviceSelectionError::DeviceNotSelectable { .. }));
        let error = registry
            .select(&DeviceSelection::BackendIndex {
                backend: gpu_info.backend_id,
                index: 99,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            DeviceSelectionError::BackendDeviceUnavailable { index: 99, .. }
        ));
    }

    #[test]
    fn selection_syntax_is_explicit() {
        assert_eq!("cpu".parse::<DeviceSelection>().unwrap(), DeviceSelection::Cpu);
        assert_eq!(
            "id:forgecore/gpu/3/cuda0".parse::<DeviceSelection>().unwrap(),
            DeviceSelection::ById(DeviceId::new("forgecore/gpu/3/cuda0"))
        );
        assert_eq!(
            "backend:forgecore-gpu-cuda0:3"
                .parse::<DeviceSelection>()
                .unwrap(),
            DeviceSelection::BackendIndex {
                backend: BackendId::new("forgecore-gpu-cuda0"),
                index: 3,
            }
        );
        assert!("gpu".parse::<DeviceSelection>().is_err());
    }

    #[test]
    fn execution_config_validates_without_mixing_resource_policy() {
        let config = ExecutionConfig::new(DeviceSelection::Cpu, 2, Some(64), true).unwrap();
        let selected = config.resolve(&DeviceRegistry::cpu_only()).unwrap();
        assert_eq!(selected.info().kind, DeviceKind::Cpu);
        assert_eq!(config.cpu_thread_count, 2);
        assert!(config.check_tensors);
    }

    #[test]
    fn unknown_native_device_kind_is_not_selectable() {
        let info = device_info_from_forgecore(ForgeDeviceInfo {
            index: 4,
            name: "future".to_string(),
            description: "future".to_string(),
            device_type: DeviceType::Unknown(99),
            memory_free: 0,
            memory_total: 0,
        });
        assert_eq!(info.capabilities.model_execution, CapabilityState::Unsupported);
        assert!(!info.selectable);
        assert_eq!(info.accounting_domain(), MemoryDomain::Unknown);
    }
}
