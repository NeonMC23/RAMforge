//! Adaptive transfer scheduling policy.
//!
//! Pure decision logic over calibration and in-session measurements. The policy
//! never executes a strategy itself; it explains which validated path should be
//! used next and why.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::hw_calibration::{CalibrationRecord, CalibrationStatus};
use crate::runtime_plan::RuntimeProfileKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferStrategy {
    SynchronousReplace,
    AsyncPrefetch,
    NoPrefetch,
    UnsupportedFallback,
}

impl TransferStrategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::SynchronousReplace => "synchronous_replace",
            Self::AsyncPrefetch => "async_prefetch",
            Self::NoPrefetch => "no_prefetch",
            Self::UnsupportedFallback => "unsupported_fallback",
        }
    }
}

impl fmt::Display for TransferStrategy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionConfidence {
    Measured,
    Calibrated,
    Advisory,
    Fallback,
}

impl DecisionConfidence {
    pub fn name(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Calibrated => "calibrated",
            Self::Advisory => "advisory",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyDecision {
    pub strategy: TransferStrategy,
    pub reason: String,
    pub confidence: DecisionConfidence,
    pub expected_host_prepare_ns: Option<u64>,
    pub expected_transfer_ns: Option<u64>,
    pub expected_compute_ns: Option<u64>,
    pub expected_exposed_transfer_ns: Option<u64>,
    pub predicted_sync_total_ns: Option<u64>,
    pub predicted_async_total_ns: Option<u64>,
    pub sync_samples: usize,
    pub async_samples: usize,
    pub measured_sync_median_ns: Option<u64>,
    pub measured_async_median_ns: Option<u64>,
    pub effective_budget_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct DecisionContext<'a> {
    pub profile: RuntimeProfileKind,
    pub calibration: Option<&'a CalibrationRecord>,
    pub calibration_status: CalibrationStatus,
    pub pending_bytes: u64,
    pub compute_estimate_ns: Option<u64>,
    pub host_prepare_estimate_ns: Option<u64>,
    pub budget_bytes: Option<u64>,
    pub user_override: Option<TransferStrategy>,
    pub allow_async: bool,
}

#[derive(Debug, Clone, Default)]
pub struct AdaptiveController {
    sync_samples_ns: Vec<u64>,
    async_samples_ns: Vec<u64>,
    pub min_samples: usize,
    pub hysteresis: f64,
}

impl AdaptiveController {
    pub fn new() -> Self {
        Self {
            sync_samples_ns: Vec::new(),
            async_samples_ns: Vec::new(),
            min_samples: 2,
            hysteresis: 0.05,
        }
    }

    pub fn record(&mut self, strategy: TransferStrategy, total_ns: u64) {
        match strategy {
            TransferStrategy::SynchronousReplace => self.sync_samples_ns.push(total_ns),
            TransferStrategy::AsyncPrefetch => self.async_samples_ns.push(total_ns),
            _ => {}
        }
    }

    pub fn sync_samples(&self) -> usize {
        self.sync_samples_ns.len()
    }

    pub fn async_samples(&self) -> usize {
        self.async_samples_ns.len()
    }

    pub fn sync_median_ns(&self) -> Option<u64> {
        median(&self.sync_samples_ns)
    }

    pub fn async_median_ns(&self) -> Option<u64> {
        median(&self.async_samples_ns)
    }

    pub fn decide(&self, context: &DecisionContext<'_>) -> StrategyDecision {
        let expected_transfer_ns = estimate_transfer_ns(context);
        let expected_compute_ns = context.compute_estimate_ns;
        let host_prepare = context.host_prepare_estimate_ns.unwrap_or(0);
        let (predicted_sync_total_ns, predicted_async_total_ns) =
            match (expected_transfer_ns, expected_compute_ns) {
                (Some(transfer), Some(compute)) => (
                    Some(
                        host_prepare
                            .saturating_add(transfer)
                            .saturating_add(compute),
                    ),
                    Some(host_prepare.saturating_add(transfer.max(compute))),
                ),
                _ => (None, None),
            };
        let exposed = |strategy: TransferStrategy| match (
            strategy,
            expected_transfer_ns,
            expected_compute_ns,
        ) {
            (TransferStrategy::SynchronousReplace, Some(transfer), _) => Some(transfer),
            (TransferStrategy::AsyncPrefetch, Some(transfer), Some(compute)) => {
                Some(transfer.saturating_sub(compute))
            }
            _ => None,
        };

        let mut decision = StrategyDecision {
            strategy: TransferStrategy::SynchronousReplace,
            reason: String::new(),
            confidence: DecisionConfidence::Fallback,
            expected_host_prepare_ns: context.host_prepare_estimate_ns,
            expected_transfer_ns,
            expected_compute_ns,
            expected_exposed_transfer_ns: None,
            predicted_sync_total_ns,
            predicted_async_total_ns,
            sync_samples: self.sync_samples(),
            async_samples: self.async_samples(),
            measured_sync_median_ns: self.sync_median_ns(),
            measured_async_median_ns: self.async_median_ns(),
            effective_budget_bytes: context.budget_bytes,
        };

        if context.pending_bytes == 0 {
            decision.strategy = TransferStrategy::NoPrefetch;
            decision.confidence = DecisionConfidence::Advisory;
            decision.reason = "no pending transfer is required".to_string();
            return decision;
        }
        if let Some(override_strategy) = context.user_override {
            decision.strategy = override_strategy;
            decision.confidence = DecisionConfidence::Advisory;
            decision.reason = "explicit user strategy override".to_string();
            decision.expected_exposed_transfer_ns = exposed(override_strategy);
            return decision;
        }
        if context.profile == RuntimeProfileKind::Background {
            decision.strategy = TransferStrategy::SynchronousReplace;
            decision.confidence = DecisionConfidence::Advisory;
            decision.reason = "background profile avoids concurrent staging transfers".to_string();
            decision.expected_exposed_transfer_ns = exposed(decision.strategy);
            return decision;
        }
        if context.calibration.is_none() || context.calibration_status != CalibrationStatus::Valid {
            decision.strategy = TransferStrategy::SynchronousReplace;
            decision.confidence = DecisionConfidence::Fallback;
            decision.reason = if context.calibration.is_none() {
                "no calibration record; deterministic synchronous fallback".to_string()
            } else {
                "calibration does not match this hardware/model; synchronous fallback".to_string()
            };
            decision.expected_exposed_transfer_ns = exposed(decision.strategy);
            return decision;
        }
        if !context.allow_async {
            decision.strategy = TransferStrategy::SynchronousReplace;
            decision.confidence = DecisionConfidence::Advisory;
            decision.reason = "asynchronous preparation is disabled for this request".to_string();
            decision.expected_exposed_transfer_ns = exposed(decision.strategy);
            return decision;
        }
        if expected_compute_ns.is_none() {
            decision.strategy = TransferStrategy::SynchronousReplace;
            decision.confidence = DecisionConfidence::Fallback;
            decision.reason =
                "no reliable compute-duration estimate; synchronous fallback".to_string();
            decision.expected_exposed_transfer_ns = exposed(decision.strategy);
            return decision;
        }

        // Collect a minimum number of samples for both strategies before
        // trusting a comparison. Interleave strategies so page-cache or
        // allocator warmup does not systematically bias one path.
        let async_profiles = matches!(
            context.profile,
            RuntimeProfileKind::MaximumPerformance | RuntimeProfileKind::Balanced
        );
        let needs_exploration = self.sync_samples() < self.min_samples
            || (async_profiles && self.async_samples() < self.min_samples);
        if needs_exploration {
            let choose_async = async_profiles && self.async_samples() < self.sync_samples();
            decision.strategy = if choose_async {
                TransferStrategy::AsyncPrefetch
            } else {
                TransferStrategy::SynchronousReplace
            };
            decision.confidence = DecisionConfidence::Calibrated;
            decision.reason = format!(
                "exploration: sync {}/{} async {}/{} samples",
                self.sync_samples(),
                self.min_samples,
                self.async_samples(),
                self.min_samples
            );
            decision.expected_exposed_transfer_ns = exposed(decision.strategy);
            return decision;
        }

        if let (Some(sync), Some(asynchronous)) = (self.sync_median_ns(), self.async_median_ns()) {
            let hysteresis = match context.profile {
                RuntimeProfileKind::MaximumPerformance => self.hysteresis.min(0.02),
                RuntimeProfileKind::Balanced => self.hysteresis,
                RuntimeProfileKind::Background | RuntimeProfileKind::Custom => self.hysteresis,
            };
            let async_better = (asynchronous as f64) < (sync as f64) * (1.0 - hysteresis);
            let sync_better = (asynchronous as f64) > (sync as f64) * (1.0 + hysteresis);
            if async_better {
                decision.strategy = TransferStrategy::AsyncPrefetch;
                decision.confidence = DecisionConfidence::Measured;
                decision.reason = format!(
                    "measured async median {} ns is faster than sync median {} ns",
                    asynchronous, sync
                );
            } else if sync_better {
                decision.strategy = TransferStrategy::SynchronousReplace;
                decision.confidence = DecisionConfidence::Measured;
                decision.reason = format!(
                    "measured sync median {} ns is faster than async median {} ns",
                    sync, asynchronous
                );
            } else {
                decision.strategy = TransferStrategy::SynchronousReplace;
                decision.confidence = DecisionConfidence::Measured;
                decision.reason = format!(
                    "measured medians are within {:.0}% hysteresis (sync {} ns, async {} ns); prefer synchronous",
                    hysteresis * 100.0,
                    sync,
                    asynchronous
                );
            }
            decision.expected_exposed_transfer_ns = exposed(decision.strategy);
            return decision;
        }

        decision.strategy = TransferStrategy::SynchronousReplace;
        decision.confidence = DecisionConfidence::Calibrated;
        decision.reason = "insufficient comparable samples; synchronous fallback".to_string();
        decision.expected_exposed_transfer_ns = exposed(decision.strategy);
        decision
    }
}

fn estimate_transfer_ns(context: &DecisionContext<'_>) -> Option<u64> {
    let record = context.calibration?;
    let bandwidth = record.pinned_bandwidth_for(context.pending_bytes)?;
    if bandwidth <= 0.0 {
        return None;
    }
    Some((context.pending_bytes as f64 / bandwidth).ceil() as u64)
}

fn median(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    Some(sorted[sorted.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw_calibration::{CalibrationIdentity, CalibrationRecord};
    use crate::runtime_plan::RuntimeProfileKind;

    fn calibration() -> CalibrationRecord {
        CalibrationRecord {
            schema_version: crate::hw_calibration::HW_CALIBRATION_SCHEMA_VERSION,
            created_unix_ms: 1,
            identity: CalibrationIdentity {
                device_name: "GPU".to_string(),
                device_index: 0,
                driver_version: 1,
                native_backend: "cuda".to_string(),
                model_path: "/m".to_string(),
                model_size_bytes: 1,
                model_modified_unix_ms: Some(1),
                calibration_tool_version: 1,
            },
            pinned: vec![crate::hw_calibration::TransferSample {
                memory_type: "pinned_host".to_string(),
                payload_bytes: 1024 * 1024,
                duration_ns: 1_000_000,
                submit_ns: 10_000,
                bandwidth_gbps: 1.0,
            }],
            pageable: vec![],
            chunks: vec![],
            allocations: vec![],
            event_overhead_ns: 100,
            notes: vec![],
        }
    }

    fn context<'a>(
        profile: RuntimeProfileKind,
        record: Option<&'a CalibrationRecord>,
    ) -> DecisionContext<'a> {
        DecisionContext {
            profile,
            calibration: record,
            calibration_status: if record.is_some() {
                CalibrationStatus::Valid
            } else {
                CalibrationStatus::Missing
            },
            pending_bytes: 10 * 1024 * 1024,
            compute_estimate_ns: Some(100_000_000),
            host_prepare_estimate_ns: Some(0),
            budget_bytes: Some(1 << 30),
            user_override: None,
            allow_async: true,
        }
    }

    #[test]
    fn missing_calibration_is_synchronous_fallback() {
        let controller = AdaptiveController::new();
        let decision = controller.decide(&context(RuntimeProfileKind::MaximumPerformance, None));
        assert_eq!(decision.strategy, TransferStrategy::SynchronousReplace);
        assert_eq!(decision.confidence, DecisionConfidence::Fallback);
    }

    #[test]
    fn background_profile_never_prefetches() {
        let record = calibration();
        let controller = AdaptiveController::new();
        let decision = controller.decide(&context(RuntimeProfileKind::Background, Some(&record)));
        assert_eq!(decision.strategy, TransferStrategy::SynchronousReplace);
    }

    #[test]
    fn exploration_then_measured_sync_wins() {
        let record = calibration();
        let mut controller = AdaptiveController::new();
        let first = controller.decide(&context(RuntimeProfileKind::Balanced, Some(&record)));
        assert_eq!(first.strategy, TransferStrategy::SynchronousReplace);
        controller.record(TransferStrategy::SynchronousReplace, 100);
        controller.record(TransferStrategy::SynchronousReplace, 110);
        let second = controller.decide(&context(RuntimeProfileKind::Balanced, Some(&record)));
        assert_eq!(second.strategy, TransferStrategy::AsyncPrefetch);
        controller.record(TransferStrategy::AsyncPrefetch, 500);
        controller.record(TransferStrategy::AsyncPrefetch, 510);
        let decided = controller.decide(&context(RuntimeProfileKind::Balanced, Some(&record)));
        assert_eq!(decided.strategy, TransferStrategy::SynchronousReplace);
        assert_eq!(decided.confidence, DecisionConfidence::Measured);
    }

    #[test]
    fn measured_async_win_is_selected() {
        let record = calibration();
        let mut controller = AdaptiveController::new();
        controller.record(TransferStrategy::SynchronousReplace, 500);
        controller.record(TransferStrategy::SynchronousReplace, 510);
        controller.record(TransferStrategy::AsyncPrefetch, 100);
        controller.record(TransferStrategy::AsyncPrefetch, 110);
        let decided = controller.decide(&context(
            RuntimeProfileKind::MaximumPerformance,
            Some(&record),
        ));
        assert_eq!(decided.strategy, TransferStrategy::AsyncPrefetch);
        assert_eq!(decided.confidence, DecisionConfidence::Measured);
    }

    #[test]
    fn user_override_wins() {
        let record = calibration();
        let controller = AdaptiveController::new();
        let mut ctx = context(RuntimeProfileKind::Background, Some(&record));
        ctx.user_override = Some(TransferStrategy::AsyncPrefetch);
        assert_eq!(
            controller.decide(&ctx).strategy,
            TransferStrategy::AsyncPrefetch
        );
    }

    #[test]
    fn maximum_performance_accepts_smaller_measured_edge() {
        let record = calibration();
        let mut controller = AdaptiveController::new();
        controller.record(TransferStrategy::SynchronousReplace, 100);
        controller.record(TransferStrategy::SynchronousReplace, 101);
        controller.record(TransferStrategy::AsyncPrefetch, 97);
        controller.record(TransferStrategy::AsyncPrefetch, 98);
        let maximum = controller.decide(&context(
            RuntimeProfileKind::MaximumPerformance,
            Some(&record),
        ));
        let balanced = controller.decide(&context(RuntimeProfileKind::Balanced, Some(&record)));
        assert_eq!(maximum.strategy, TransferStrategy::AsyncPrefetch);
        assert_eq!(balanced.strategy, TransferStrategy::SynchronousReplace);
    }

    #[test]
    fn transfer_estimate_uses_calibrated_bandwidth() {
        let record = calibration();
        let controller = AdaptiveController::new();
        let decision = controller.decide(&context(RuntimeProfileKind::Balanced, Some(&record)));
        assert_eq!(decision.expected_transfer_ns, Some(10 * 1024 * 1024));
        assert_eq!(decision.predicted_sync_total_ns, Some(110_485_760));
        assert_eq!(decision.predicted_async_total_ns, Some(100_000_000));
    }
}
