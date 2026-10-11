//! Optional native ForgeCore device-registry coverage.
//!
//! The test is opt-in because loading a native backend registry is an
//! environment operation. CPU-only builds are valid and do not fail this
//! test; a discovered non-CPU device is reported but is not executed because
//! RAMforge's current adapter deliberately does not enable non-CPU binding.

use ramforge_runtime::{CapabilityState, DeviceKind, DeviceRegistry};

#[test]
fn native_device_registry_is_optional_and_cpu_safe() {
    if std::env::var("RAMFORGE_RUN_NATIVE_DEVICE_TESTS").ok().as_deref() != Some("1") {
        println!("SKIP: set RAMFORGE_RUN_NATIVE_DEVICE_TESTS=1 to query ForgeCore devices");
        return;
    }

    let registry = DeviceRegistry::discover();
    assert!(
        registry.devices().iter().any(|device| device.kind == DeviceKind::Cpu),
        "a native ForgeCore registry must expose its CPU device"
    );

    for device in registry.devices() {
        if device.kind != DeviceKind::Cpu {
            assert_eq!(device.capabilities.model_execution, CapabilityState::Unsupported);
            assert!(!device.selectable);
        }
    }
}
