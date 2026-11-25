//! Small, explainable dense-residency policy.
//!
//! This does not allocate or move memory. It decides whether the existing
//! bounded staged-layer mechanism should be enabled, based only on the model
//! byte count, the planning RAM budget, and observed GPU availability.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseResidencyDecision {
    pub model_bytes: u64,
    pub ram_budget_bytes: u64,
    pub gpu_available: bool,
    pub stage_layers: bool,
    pub feasible: bool,
    pub reason: String,
}

/// Decide whether bounded staged-layer residency is justified.
///
/// The actual stage budget is chosen later from free VRAM after the context
/// exists, minus a safety margin. This function only answers "should staging
/// be considered at all".
pub fn decide_dense_residency(
    model_bytes: u64,
    ram_budget_bytes: u64,
    gpu_available: bool,
) -> DenseResidencyDecision {
    if model_bytes <= ram_budget_bytes {
        return DenseResidencyDecision {
            model_bytes,
            ram_budget_bytes,
            gpu_available,
            stage_layers: false,
            feasible: true,
            reason: "model fits within the planning RAM budget; direct execution is preferred"
                .to_string(),
        };
    }
    if gpu_available {
        DenseResidencyDecision {
            model_bytes,
            ram_budget_bytes,
            gpu_available,
            stage_layers: true,
            feasible: true,
            reason: "model exceeds the planning RAM budget; use bounded staging within free VRAM after context creation"
                .to_string(),
        }
    } else {
        DenseResidencyDecision {
            model_bytes,
            ram_budget_bytes,
            gpu_available,
            stage_layers: false,
            feasible: false,
            reason: "model exceeds the planning RAM budget and no GPU backend is available for bounded staging"
                .to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitting_model_stays_direct() {
        let decision = decide_dense_residency(100, 200, true);
        assert!(!decision.stage_layers);
        assert!(decision.feasible);
    }

    #[test]
    fn oversized_model_uses_gpu_when_available() {
        let decision = decide_dense_residency(200, 100, true);
        assert!(decision.stage_layers);
        assert!(decision.feasible);
        assert!(decision.reason.contains("bounded staging"));
    }

    #[test]
    fn oversized_model_without_gpu_is_infeasible() {
        let decision = decide_dense_residency(200, 100, false);
        assert!(!decision.stage_layers);
        assert!(!decision.feasible);
        assert!(decision.reason.contains("no GPU"));
    }
}
