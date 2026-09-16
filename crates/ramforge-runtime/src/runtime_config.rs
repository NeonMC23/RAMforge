//! Concrete configuration consumed by the inference runtime.
//!
//! This module intentionally contains no Planner profiles, policies, reason
//! codes, calibration data, or fingerprints. `RuntimeConfig` is the execution
//! contract produced after planning and validation are complete.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeExecutionDevice {
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub cpu_thread_count: usize,
    pub ram_budget_bytes: u64,
    pub layer_cache_enabled: bool,
    pub layer_cache_capacity_bytes: u64,
    pub read_coalescing_enabled: bool,
    pub grouped_read_buffer_reuse_enabled: bool,
    pub execution_device: RuntimeExecutionDevice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeConfigError {
    ZeroThreadCount,
    ZeroRamBudget,
    LayerCacheEnabledWithoutCapacity,
    LayerCacheDisabledWithCapacity,
    LayerCacheCapacityExceedsRamBudget {
        capacity_bytes: u64,
        ram_budget_bytes: u64,
    },
}

impl fmt::Display for RuntimeConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroThreadCount => write!(formatter, "CPU thread count must be greater than zero"),
            Self::ZeroRamBudget => write!(formatter, "RAM budget must be greater than zero"),
            Self::LayerCacheEnabledWithoutCapacity => {
                write!(formatter, "enabled layer cache must have non-zero capacity")
            }
            Self::LayerCacheDisabledWithCapacity => {
                write!(formatter, "disabled layer cache must have zero capacity")
            }
            Self::LayerCacheCapacityExceedsRamBudget {
                capacity_bytes,
                ram_budget_bytes,
            } => write!(
                formatter,
                "layer cache capacity {capacity_bytes} exceeds RAM budget {ram_budget_bytes}"
            ),
        }
    }
}

impl std::error::Error for RuntimeConfigError {}

impl RuntimeConfig {
    pub(crate) const CURRENT_READ_COALESCING_ENABLED: bool = true;
    pub(crate) const CURRENT_GROUPED_READ_BUFFER_REUSE_ENABLED: bool = true;

    pub fn new(
        cpu_thread_count: usize,
        ram_budget_bytes: u64,
        layer_cache_enabled: bool,
        layer_cache_capacity_bytes: u64,
        read_coalescing_enabled: bool,
        grouped_read_buffer_reuse_enabled: bool,
    ) -> Result<Self, RuntimeConfigError> {
        let config = Self {
            cpu_thread_count,
            ram_budget_bytes,
            layer_cache_enabled,
            layer_cache_capacity_bytes,
            read_coalescing_enabled,
            grouped_read_buffer_reuse_enabled,
            execution_device: RuntimeExecutionDevice::Cpu,
        };
        config.validate()?;
        Ok(config)
    }

    /// Materialize the policies used by the legacy `InferenceEngine::new`
    /// path once its model-dependent cache capacity is known.
    ///
    /// The RAM budget has no universal default, and the existing cache bound is
    /// derived from model descriptors and that budget. Requiring both values
    /// keeps this configuration concrete instead of introducing an "automatic"
    /// runtime policy.
    pub fn current_defaults(
        ram_budget_bytes: u64,
        layer_cache_capacity_bytes: u64,
    ) -> Result<Self, RuntimeConfigError> {
        let cpu_thread_count = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        Self::current_defaults_with_thread_count(
            cpu_thread_count,
            ram_budget_bytes,
            layer_cache_capacity_bytes,
        )
    }

    pub(crate) fn current_defaults_with_thread_count(
        cpu_thread_count: usize,
        ram_budget_bytes: u64,
        layer_cache_capacity_bytes: u64,
    ) -> Result<Self, RuntimeConfigError> {
        Self::new(
            cpu_thread_count,
            ram_budget_bytes,
            layer_cache_capacity_bytes > 0,
            layer_cache_capacity_bytes,
            Self::CURRENT_READ_COALESCING_ENABLED,
            Self::CURRENT_GROUPED_READ_BUFFER_REUSE_ENABLED,
        )
    }

    pub fn validate(&self) -> Result<(), RuntimeConfigError> {
        if self.cpu_thread_count == 0 {
            return Err(RuntimeConfigError::ZeroThreadCount);
        }
        if self.ram_budget_bytes == 0 {
            return Err(RuntimeConfigError::ZeroRamBudget);
        }
        if self.layer_cache_enabled && self.layer_cache_capacity_bytes == 0 {
            return Err(RuntimeConfigError::LayerCacheEnabledWithoutCapacity);
        }
        if !self.layer_cache_enabled && self.layer_cache_capacity_bytes != 0 {
            return Err(RuntimeConfigError::LayerCacheDisabledWithCapacity);
        }
        if self.layer_cache_capacity_bytes > self.ram_budget_bytes {
            return Err(RuntimeConfigError::LayerCacheCapacityExceedsRamBudget {
                capacity_bytes: self.layer_cache_capacity_bytes,
                ram_budget_bytes: self.ram_budget_bytes,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_construction_preserves_explicit_values() {
        let config = RuntimeConfig::new(3, 10_000, true, 2_000, true, false).unwrap();
        assert_eq!(config.cpu_thread_count, 3);
        assert_eq!(config.ram_budget_bytes, 10_000);
        assert!(config.layer_cache_enabled);
        assert_eq!(config.layer_cache_capacity_bytes, 2_000);
        assert!(config.read_coalescing_enabled);
        assert!(!config.grouped_read_buffer_reuse_enabled);
        assert_eq!(config.execution_device, RuntimeExecutionDevice::Cpu);
    }

    #[test]
    fn test_runtime_config_current_defaults_match_legacy_policies() {
        let config = RuntimeConfig::current_defaults(10_000, 4_000).unwrap();
        assert!(config.cpu_thread_count > 0);
        assert_eq!(config.ram_budget_bytes, 10_000);
        assert!(config.layer_cache_enabled);
        assert_eq!(config.layer_cache_capacity_bytes, 4_000);
        assert!(config.read_coalescing_enabled);
        assert!(config.grouped_read_buffer_reuse_enabled);
        assert_eq!(config.execution_device, RuntimeExecutionDevice::Cpu);

        let uncached = RuntimeConfig::current_defaults(10_000, 0).unwrap();
        assert!(!uncached.layer_cache_enabled);
        assert_eq!(uncached.layer_cache_capacity_bytes, 0);
    }

    #[test]
    fn test_runtime_config_rejects_zero_execution_limits() {
        assert_eq!(
            RuntimeConfig::new(0, 10_000, false, 0, true, false).unwrap_err(),
            RuntimeConfigError::ZeroThreadCount
        );
        assert_eq!(
            RuntimeConfig::new(1, 0, false, 0, true, false).unwrap_err(),
            RuntimeConfigError::ZeroRamBudget
        );
    }

    #[test]
    fn test_runtime_config_rejects_inconsistent_cache_state() {
        assert_eq!(
            RuntimeConfig::new(1, 10_000, true, 0, true, false).unwrap_err(),
            RuntimeConfigError::LayerCacheEnabledWithoutCapacity
        );
        assert_eq!(
            RuntimeConfig::new(1, 10_000, false, 1, true, false).unwrap_err(),
            RuntimeConfigError::LayerCacheDisabledWithCapacity
        );
    }

    #[test]
    fn test_runtime_config_preserves_independent_io_permissions() {
        let config = RuntimeConfig::new(1, 10_000, false, 0, false, true).unwrap();
        assert!(!config.read_coalescing_enabled);
        assert!(config.grouped_read_buffer_reuse_enabled);
    }
}
