//! Versioned, observation-only contract for future scheduler consumers.
//!
//! The contract deliberately contains no eviction, release, prefetch, or
//! final-use authorization. It records what was observed, how it was obtained,
//! and what remains unknown. Consumers must not derive physical access or
//! reclamation permission from these values.

use serde::{Deserialize, Serialize};

use crate::residency_engine::NodeDispatchObservation;
use crate::{NodeDispatchEvent, NodeDispatchSource, PageSample, PageSampleStatus};

/// Current scheduler-facing observation schema.
pub const SCHEDULER_OBSERVATION_SCHEMA_VERSION: u32 = 1;
/// Bound on dispatch evidence embedded in one contract.
pub const MAX_CONTRACT_DISPATCHES: usize = 512;

/// How page evidence was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationStrategy {
    Disabled,
    Systematic,
    Targeted,
    Adaptive,
    Unknown,
}

/// Provenance categories present in one observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationProvenance {
    /// Fresh native page-residency observation.
    FreshMincore,
    /// Cached observation reused within its freshness window.
    ReusedCache,
    /// Fresh cached range intentionally skipped by Targeted.
    SkippedFreshCache,
    /// Not observed because the execution budget was exhausted.
    BudgetExcluded,
    /// Not observed because of a range/strategy cap or mapping gap.
    Unsampled,
    /// A page-residency state could not be determined.
    Unknown,
    /// The platform or backend does not support the observation.
    Unsupported,
    /// The observation failed.
    Failed,
    /// No page observation was requested.
    Disabled,
}

/// Deduplicated page accounting. Every invariant is validated explicitly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageAccount {
    pub requested_pages: u64,
    pub sampled_pages: u64,
    pub fresh_pages: u64,
    pub reused_pages: u64,
    pub skipped_pages: u64,
    pub budget_excluded_pages: u64,
    pub unsampled_pages: u64,
    pub resident_pages: u64,
    pub nonresident_pages: u64,
    pub unknown_pages: u64,
    pub observation_error: u32,
    /// Event sequence at which the dispatch sample was taken or reused.
    pub sampled_at_sequence: u64,
}

impl PageAccount {
    pub fn from_sample(sample: PageSample) -> Self {
        Self {
            requested_pages: sample.requested_pages,
            sampled_pages: sample.sampled_pages,
            fresh_pages: sample.fresh_sampled_pages,
            reused_pages: sample.reused_pages,
            skipped_pages: sample.skipped_pages,
            budget_excluded_pages: sample.budget_excluded_pages,
            unsampled_pages: sample.unsampled_pages,
            resident_pages: sample.resident_pages,
            nonresident_pages: sample.nonresident_pages,
            unknown_pages: sample.unknown_pages,
            observation_error: sample.error,
            sampled_at_sequence: sample.sampled_at_sequence,
        }
    }

    /// Returns the first violated accounting invariant, if any.
    pub fn validate(&self) -> Result<(), ContractError> {
        let accounted = self
            .sampled_pages
            .checked_add(self.skipped_pages)
            .and_then(|value| value.checked_add(self.budget_excluded_pages))
            .and_then(|value| value.checked_add(self.unsampled_pages))
            .ok_or(ContractError::Overflow("page accounting"))?;
        if accounted != self.requested_pages {
            return Err(ContractError::Invariant(format!(
                "requested {} != sampled {} + skipped {} + budget-excluded {} + unsampled {}",
                self.requested_pages,
                self.sampled_pages,
                self.skipped_pages,
                self.budget_excluded_pages,
                self.unsampled_pages
            )));
        }
        let observed = self
            .resident_pages
            .checked_add(self.nonresident_pages)
            .and_then(|value| value.checked_add(self.unknown_pages))
            .ok_or(ContractError::Overflow("observed page states"))?;
        if observed != self.sampled_pages {
            return Err(ContractError::Invariant(format!(
                "sampled {} != resident {} + nonresident {} + unknown {}",
                self.sampled_pages, self.resident_pages, self.nonresident_pages, self.unknown_pages
            )));
        }
        let provenance = self
            .fresh_pages
            .checked_add(self.reused_pages)
            .ok_or(ContractError::Overflow("provenance pages"))?;
        if provenance != self.sampled_pages {
            return Err(ContractError::Invariant(format!(
                "sampled {} != fresh {} + reused {}",
                self.sampled_pages, self.fresh_pages, self.reused_pages
            )));
        }
        Ok(())
    }

    pub fn coverage(&self) -> f64 {
        if self.requested_pages == 0 {
            0.0
        } else {
            self.sampled_pages as f64 / self.requested_pages as f64
        }
    }
}

/// Per-source attribution derived from the same deduplicated page observation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAttribution {
    pub name: String,
    pub mapping_id: Option<u64>,
    pub mapping_generation: Option<u64>,
    pub page_start: Option<u64>,
    pub page_end: Option<u64>,
    pub requested_pages: u64,
    pub exclusive_pages: u64,
    pub shared_pages: u64,
    pub sampled_pages: u64,
    pub fresh_pages: u64,
    pub reused_pages: u64,
    pub resident_pages: u64,
    pub nonresident_pages: u64,
    pub unknown_pages: u64,
    pub skipped_pages: u64,
    pub budget_excluded_pages: u64,
    pub unsampled_pages: u64,
}

impl SourceAttribution {
    pub fn from_source(source: &NodeDispatchSource) -> Self {
        Self {
            name: source.name.clone(),
            mapping_id: source.mapping_id,
            mapping_generation: source.mapping_generation,
            page_start: source.page_start,
            page_end: source.page_end,
            requested_pages: source.requested_pages,
            exclusive_pages: source.exclusive_pages,
            shared_pages: source.shared_pages,
            sampled_pages: source.sampled_pages,
            fresh_pages: source.fresh_pages,
            reused_pages: source.reused_pages,
            resident_pages: source.resident_pages,
            nonresident_pages: source.nonresident_pages,
            unknown_pages: source.unknown_pages,
            skipped_pages: source.skipped_pages,
            budget_excluded_pages: source.budget_excluded_pages,
            unsampled_pages: source.unsampled_pages,
        }
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.exclusive_pages + self.shared_pages != self.requested_pages {
            return Err(ContractError::Invariant(format!(
                "source '{}' exclusive {} + shared {} != requested {}",
                self.name, self.exclusive_pages, self.shared_pages, self.requested_pages
            )));
        }
        let accounted = self
            .sampled_pages
            .checked_add(self.skipped_pages)
            .and_then(|value| value.checked_add(self.budget_excluded_pages))
            .and_then(|value| value.checked_add(self.unsampled_pages))
            .ok_or(ContractError::Overflow("source page accounting"))?;
        if accounted != self.requested_pages {
            return Err(ContractError::Invariant(format!(
                "source '{}' accounting does not match requested pages",
                self.name
            )));
        }
        if self.resident_pages + self.nonresident_pages + self.unknown_pages != self.sampled_pages {
            return Err(ContractError::Invariant(format!(
                "source '{}' observed states do not match sampled pages",
                self.name
            )));
        }
        if self.fresh_pages + self.reused_pages != self.sampled_pages {
            return Err(ContractError::Invariant(format!(
                "source '{}' provenance does not match sampled pages",
                self.name
            )));
        }
        Ok(())
    }
}

/// One dispatch's evidence. `sample` is `None` when sampling was disabled or
/// the dispatch had no validated file-backed mapping range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchEvidence {
    pub execution_sequence: u64,
    pub event_sequence: u64,
    pub node_index: u32,
    pub node_count: u32,
    pub node_name: Option<String>,
    pub op_name: Option<String>,
    pub source_references: usize,
    pub sample: Option<PageAccount>,
    pub sources: Vec<SourceAttribution>,
}

impl DispatchEvidence {
    fn from_event(event: &NodeDispatchEvent) -> Self {
        Self {
            execution_sequence: event.execution_sequence,
            event_sequence: event.event_sequence,
            node_index: event.node_index,
            node_count: event.node_count,
            node_name: event.node_name.clone(),
            op_name: event.op_name.clone(),
            source_references: event.source_references,
            sample: event.dispatch_sample.map(PageAccount::from_sample),
            sources: event
                .source_samples
                .iter()
                .map(SourceAttribution::from_source)
                .collect(),
        }
    }
}

/// Observation-only scheduler contract. It has no eviction or release method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerObservation {
    pub schema_version: u32,
    pub observation_only: bool,
    pub model_instance_id: Option<u64>,
    pub context_id: Option<u64>,
    pub strategy: ObservationStrategy,
    pub sampling_enabled: bool,
    pub budget_total: u64,
    pub budget_remaining: u64,
    pub executions: u64,
    pub events_retained: usize,
    pub events_dropped: u64,
    pub global: PageAccount,
    pub provenance: Vec<ObservationProvenance>,
    pub dispatches: Vec<DispatchEvidence>,
    pub dispatches_truncated: usize,
    pub limitations: Vec<String>,
}

impl SchedulerObservation {
    /// Build a contract from the existing dispatch report. This does not
    /// change the observation and does not perform additional sampling.
    pub fn from_observation(observation: &NodeDispatchObservation) -> Self {
        let sampling_enabled = observation.sampling_enabled;
        let strategy = if !sampling_enabled {
            ObservationStrategy::Disabled
        } else {
            match observation.sampling_strategy.as_str() {
                "systematic" => ObservationStrategy::Systematic,
                "targeted" => ObservationStrategy::Targeted,
                "adaptive" => ObservationStrategy::Adaptive,
                _ => ObservationStrategy::Unknown,
            }
        };
        let global = PageAccount {
            requested_pages: observation.requested_pages,
            sampled_pages: observation.sampled_pages,
            fresh_pages: observation.fresh_sampled_pages,
            reused_pages: observation.reused_pages,
            skipped_pages: observation.skipped_pages,
            budget_excluded_pages: observation.budget_excluded_pages,
            unsampled_pages: observation.unsampled_pages,
            resident_pages: observation.resident_pages,
            nonresident_pages: observation.nonresident_pages,
            unknown_pages: observation.unknown_pages,
            observation_error: 0,
            sampled_at_sequence: observation
                .events
                .iter()
                .filter_map(|event| event.dispatch_sample)
                .map(|sample| sample.sampled_at_sequence)
                .max()
                .unwrap_or(0),
        };
        let mut provenance = Vec::new();
        if !sampling_enabled {
            provenance.push(ObservationProvenance::Disabled);
        }
        if observation.fresh_sampled_pages > 0 {
            provenance.push(ObservationProvenance::FreshMincore);
        }
        if observation.reused_pages > 0 {
            provenance.push(ObservationProvenance::ReusedCache);
        }
        if observation.skipped_pages > 0 {
            provenance.push(ObservationProvenance::SkippedFreshCache);
        }
        if observation.budget_excluded_pages > 0 {
            provenance.push(ObservationProvenance::BudgetExcluded);
        }
        if observation.unsampled_pages > 0 {
            provenance.push(ObservationProvenance::Unsampled);
        }
        if observation.unknown_pages > 0 {
            provenance.push(ObservationProvenance::Unknown);
        }
        if observation.sampling_errors > 0
            || observation.requested_pages > 0
                && observation.sampled_pages == 0
                && observation.budget_excluded_pages == 0
                && observation.skipped_pages == 0
        {
            provenance.push(ObservationProvenance::Failed);
        }
        let limitation = |text: &str| text.to_string();
        let limitations = vec![
            limitation("Residency is not physical access, page-fault, I/O, or final-use evidence."),
            limitation("Observation does not authorize eviction or resource release."),
            limitation("Warm-cache results are not cold/nonresident validation."),
            limitation("Per-source counts are derived from one deduplicated dispatch observation."),
            limitation("Unknown and budget-excluded pages are not nonresident pages."),
        ];
        let dispatches: Vec<_> = observation
            .events
            .iter()
            .take(MAX_CONTRACT_DISPATCHES)
            .map(DispatchEvidence::from_event)
            .collect();
        let dispatches_truncated = observation.events.len().saturating_sub(dispatches.len());
        Self {
            schema_version: SCHEDULER_OBSERVATION_SCHEMA_VERSION,
            observation_only: true,
            model_instance_id: observation
                .events
                .first()
                .map(|event| event.model_instance_id),
            context_id: observation.events.first().map(|event| event.context_id),
            strategy,
            sampling_enabled,
            budget_total: observation.budget_total,
            budget_remaining: observation.budget_remaining,
            executions: observation.summary.executions,
            events_retained: observation.summary.retained_events,
            events_dropped: observation.summary.dropped_events,
            global,
            provenance,
            dispatches,
            dispatches_truncated,
            limitations,
        }
    }

    /// Validate schema and accounting invariants. A consumer should call this
    /// after deserialization before using the observation.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != SCHEDULER_OBSERVATION_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema(self.schema_version));
        }
        if !self.observation_only {
            return Err(ContractError::Invariant(
                "scheduler observation must remain observation-only".to_string(),
            ));
        }
        self.global.validate()?;
        if !self.sampling_enabled
            && (self.global.sampled_pages != 0
                || self.global.fresh_pages != 0
                || self.global.reused_pages != 0
                || self.global.skipped_pages != 0
                || self.global.budget_excluded_pages != 0)
        {
            return Err(ContractError::Invariant(
                "disabled sampling must not report page observations".to_string(),
            ));
        }
        if self.budget_remaining > self.budget_total {
            return Err(ContractError::Invariant(
                "remaining budget exceeds total budget".to_string(),
            ));
        }
        for dispatch in &self.dispatches {
            if let Some(sample) = &dispatch.sample {
                sample.validate()?;
            }
            for source in &dispatch.sources {
                source.validate()?;
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<serde_json::Value, ContractError> {
        serde_json::to_value(self).map_err(|error| ContractError::Serialization(error.to_string()))
    }
}

/// Conversion and validation failures. These are data errors, never
/// authorization to reclaim memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    UnsupportedSchema(u32),
    Invariant(String),
    Overflow(&'static str),
    Serialization(String),
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(version) => {
                write!(
                    formatter,
                    "unsupported scheduler observation schema {version}"
                )
            }
            Self::Invariant(message) => {
                write!(formatter, "observation invariant violated: {message}")
            }
            Self::Overflow(field) => write!(formatter, "observation overflow in {field}"),
            Self::Serialization(message) => {
                write!(formatter, "observation serialization failed: {message}")
            }
        }
    }
}

impl std::error::Error for ContractError {}

/// Map a page sample status to contract provenance without treating unknown
/// as nonresident.
pub fn provenance_for_status(status: PageSampleStatus) -> ObservationProvenance {
    match status {
        PageSampleStatus::Complete => ObservationProvenance::FreshMincore,
        PageSampleStatus::Partial => ObservationProvenance::Unsampled,
        PageSampleStatus::Unsupported => ObservationProvenance::Disabled,
        PageSampleStatus::Stale | PageSampleStatus::Missing | PageSampleStatus::Error => {
            ObservationProvenance::Failed
        }
        PageSampleStatus::Unknown => ObservationProvenance::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn account(requested: u64) -> PageAccount {
        PageAccount {
            requested_pages: requested,
            ..PageAccount::default()
        }
    }

    fn simple_contract() -> SchedulerObservation {
        SchedulerObservation {
            schema_version: SCHEDULER_OBSERVATION_SCHEMA_VERSION,
            observation_only: true,
            model_instance_id: Some(1),
            context_id: Some(2),
            strategy: ObservationStrategy::Adaptive,
            sampling_enabled: true,
            budget_total: 8,
            budget_remaining: 4,
            executions: 1,
            events_retained: 0,
            events_dropped: 0,
            global: PageAccount {
                requested_pages: 4,
                sampled_pages: 4,
                fresh_pages: 2,
                reused_pages: 2,
                resident_pages: 3,
                nonresident_pages: 1,
                ..PageAccount::default()
            },
            provenance: vec![
                ObservationProvenance::FreshMincore,
                ObservationProvenance::ReusedCache,
            ],
            dispatches: Vec::new(),
            dispatches_truncated: 0,
            limitations: vec!["observation only".to_string()],
        }
    }

    #[test]
    fn account_rejects_unknown_promoted_to_nonresident() {
        let mut value = account(2);
        value.sampled_pages = 2;
        value.fresh_pages = 2;
        value.resident_pages = 1;
        value.nonresident_pages = 1;
        assert!(value.validate().is_ok());
        value.unknown_pages = 1;
        assert!(value.validate().is_err());
    }

    #[test]
    fn account_separates_budget_skip_and_unsampled() {
        let value = PageAccount {
            requested_pages: 10,
            sampled_pages: 4,
            fresh_pages: 4,
            skipped_pages: 2,
            budget_excluded_pages: 3,
            unsampled_pages: 1,
            resident_pages: 4,
            ..PageAccount::default()
        };
        assert!(value.validate().is_ok());
        assert!((value.coverage() - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn contract_round_trips_and_validates() {
        let contract = simple_contract();
        contract.validate().expect("contract");
        let json = contract.to_json().expect("json");
        let decoded: SchedulerObservation = serde_json::from_value(json).expect("decode");
        assert_eq!(contract, decoded);
        decoded.validate().expect("decoded contract");
    }

    #[test]
    fn contract_rejects_unknown_schema_and_inexact_accounting() {
        let mut contract = simple_contract();
        contract.schema_version = 99;
        assert!(contract.validate().is_err());

        let mut contract = simple_contract();
        contract.global.unsampled_pages = 1;
        assert!(contract.validate().is_err());

        let mut contract = simple_contract();
        contract.observation_only = false;
        assert!(contract.validate().is_err());
    }

    #[test]
    fn disabled_contract_reports_no_page_observation() {
        let contract = SchedulerObservation {
            schema_version: SCHEDULER_OBSERVATION_SCHEMA_VERSION,
            observation_only: true,
            model_instance_id: None,
            context_id: None,
            strategy: ObservationStrategy::Disabled,
            sampling_enabled: false,
            budget_total: 0,
            budget_remaining: 0,
            executions: 1,
            events_retained: 0,
            events_dropped: 0,
            global: PageAccount::default(),
            provenance: vec![ObservationProvenance::Disabled],
            dispatches: Vec::new(),
            dispatches_truncated: 0,
            limitations: Vec::new(),
        };
        contract.validate().expect("disabled contract");
    }

    #[test]
    fn status_mapping_keeps_unknown_explicit() {
        assert_eq!(
            provenance_for_status(PageSampleStatus::Unknown),
            ObservationProvenance::Unknown
        );
        assert_eq!(
            provenance_for_status(PageSampleStatus::Error),
            ObservationProvenance::Failed
        );
        assert_eq!(
            provenance_for_status(PageSampleStatus::Complete),
            ObservationProvenance::FreshMincore
        );
    }
}
