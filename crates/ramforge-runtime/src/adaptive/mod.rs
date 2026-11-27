//! Integrated adaptive inference, resource management, and capability layer.
//!
//! The modules separate model/backend capabilities, resource admission,
//! strategy calibration, and confirmed execution plans. None of these modules
//! starts inference; the CLI/application layer owns execution and must require
//! explicit confirmation before applying a plan.

pub mod capability;
pub mod engine;
pub mod environment;
pub mod plan;
pub mod resource;

pub use capability::{
    resolve_compatibility, resolve_compatibility_with_evidence, CompatibilityReport,
    CompatibilityRequest, EvidenceSource, ExecutionFeature, FeatureSupport, SupportClass,
    CAPABILITY_SCHEMA_VERSION,
};
pub use engine::{
    baseline_strategy, build_calibration_report, compare_experiments, generate_candidates,
    select_best, AdaptiveStrategy, BestSelection, CalibrationExperiment, CalibrationReport,
    CandidateComparison, CandidateLimits, CandidateSet, CandidateStatus, ExperimentStatus,
    MeasurementSummary, ObjectivePreference, StrategyAxis, StrategyCandidate,
    ADAPTIVE_SCHEMA_VERSION,
};
pub use environment::{
    environment_changes, record_invalidation, ExecutionEnvironment, Invalidation,
    InvalidationReason, EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
};
pub use plan::{
    applied_report_from_observation, build_plan, revalidate_plan, AdaptivePlan, AppliedPlanReport,
    Evidence, FieldObservation, FieldState, PlanDecision, PlanStatus, PlanValidation,
    ResourceComparison, RuntimeObservation, ADAPTIVE_PLAN_SCHEMA_VERSION,
};
pub use resource::{
    assess_budget, BudgetClass, BudgetComponent, ResourceBudget, DEFAULT_RUNTIME_OVERHEAD_BYTES,
    DEFAULT_VRAM_OVERHEAD_BYTES, RESOURCE_BUDGET_SCHEMA_VERSION,
};
