//! Deterministic, explainable predictive residency support.
//!
//! Predictions are advisory. They consume stable execution-profile access
//! history and never bypass admission, native lifetime, or release gates.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::execution_profile::{ExecutionProfile, ProfileIdentity};
use crate::residency_engine::{ModelGeneration, ModelId, ResourceInventory, TensorResource};

pub const PREDICTION_HISTORY_SCHEMA_VERSION: u32 = 1;
pub const PREDICTION_HISTORY_KIND: &str = "ramforge.prediction_history";
pub const MAX_PREDICTION_HISTORY_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_PREDICTION_HISTORY_RECORDS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionConfidence {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionSource {
    Profile,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictionConfig {
    pub enabled: bool,
    pub min_confidence_bps: u16,
    pub staging_window_bytes: u64,
    pub speculative_budget_bytes: u64,
    pub max_in_flight: u16,
    pub lookahead: usize,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_confidence_bps: 6_000,
            staging_window_bytes: 64 * 1024 * 1024,
            speculative_budget_bytes: 16 * 1024 * 1024,
            max_in_flight: 1,
            lookahead: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionError {
    InvalidConfig(String),
    Io(String),
    Json(String),
    TooLarge { bytes: u64, limit: u64 },
    Schema(u32),
    Kind(String),
    Identity(String),
    TooManyRecords(usize),
    Incomplete,
}

impl std::fmt::Display for PredictionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid prediction config: {message}")
            }
            Self::Io(message) => write!(formatter, "prediction history I/O: {message}"),
            Self::Json(message) => write!(formatter, "prediction history JSON: {message}"),
            Self::TooLarge { bytes, limit } => {
                write!(
                    formatter,
                    "prediction history is {bytes} bytes; limit is {limit}"
                )
            }
            Self::Schema(version) => write!(formatter, "unsupported prediction schema {version}"),
            Self::Kind(kind) => write!(formatter, "unexpected prediction kind {kind:?}"),
            Self::Identity(message) => write!(formatter, "prediction identity mismatch: {message}"),
            Self::TooManyRecords(records) => {
                write!(
                    formatter,
                    "prediction history has too many records: {records}"
                )
            }
            Self::Incomplete => formatter.write_str("prediction history is incomplete"),
        }
    }
}

impl std::error::Error for PredictionError {}

impl PredictionConfig {
    pub fn validate(&self) -> Result<(), PredictionError> {
        if !self.enabled {
            return Ok(());
        }
        if self.min_confidence_bps > 10_000 {
            return Err(PredictionError::InvalidConfig(
                "min confidence must be between 0 and 10000 basis points".to_string(),
            ));
        }
        if self.staging_window_bytes == 0 {
            return Err(PredictionError::InvalidConfig(
                "staging window must be greater than zero".to_string(),
            ));
        }
        if self.speculative_budget_bytes > self.staging_window_bytes {
            return Err(PredictionError::InvalidConfig(
                "speculative budget cannot exceed the staging window".to_string(),
            ));
        }
        if self.max_in_flight == 0 || self.max_in_flight > 16 {
            return Err(PredictionError::InvalidConfig(
                "max in-flight must be between 1 and 16".to_string(),
            ));
        }
        if self.lookahead == 0 || self.lookahead > 4096 {
            return Err(PredictionError::InvalidConfig(
                "lookahead must be between 1 and 4096".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictedAccess {
    pub key: crate::residency_engine::ResourceKey,
    pub name: String,
    pub predicted_order: usize,
    pub expected_distance: usize,
    pub confidence_bps: u16,
    pub confidence: PredictionConfidence,
    pub size_bytes: u64,
    pub known_size: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictionSet {
    pub model_id: ModelId,
    pub generation: ModelGeneration,
    pub source: PredictionSource,
    pub profile_version: Option<u32>,
    pub complete_observations: u64,
    pub predictions: Vec<PredictedAccess>,
    pub rejected: BTreeMap<String, usize>,
    pub high_confidence: usize,
    pub medium_confidence: usize,
    pub low_confidence: usize,
    pub history_valid: bool,
    pub warnings: Vec<String>,
}

impl PredictionSet {
    pub fn empty(model_id: ModelId, generation: ModelGeneration, reason: &str) -> Self {
        let mut rejected = BTreeMap::new();
        rejected.insert(reason.to_string(), 1);
        Self {
            model_id,
            generation,
            source: PredictionSource::None,
            profile_version: None,
            complete_observations: 0,
            predictions: Vec::new(),
            rejected,
            high_confidence: 0,
            medium_confidence: 0,
            low_confidence: 0,
            history_valid: false,
            warnings: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PredictionFeedbackRecord {
    pub predictions: u64,
    pub hits: u64,
    pub misses: u64,
    pub staged_requests: u64,
    pub staged_bytes: u64,
    pub wasted_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictionHistorySummary {
    pub records: usize,
    pub total_predictions: u64,
    pub hits: u64,
    pub misses: u64,
    pub staged_requests: u64,
    pub staged_bytes: u64,
    pub wasted_bytes: u64,
    pub accuracy_bps: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionHistory {
    pub schema_version: u32,
    pub kind: String,
    pub identity: ProfileIdentity,
    pub records: BTreeMap<String, PredictionFeedbackRecord>,
    pub total_predictions: u64,
    pub hits: u64,
    pub misses: u64,
    pub staged_requests: u64,
    pub staged_bytes: u64,
    pub wasted_bytes: u64,
}

impl PredictionHistory {
    pub fn new(identity: ProfileIdentity) -> Self {
        Self {
            schema_version: PREDICTION_HISTORY_SCHEMA_VERSION,
            kind: PREDICTION_HISTORY_KIND.to_string(),
            identity,
            records: BTreeMap::new(),
            total_predictions: 0,
            hits: 0,
            misses: 0,
            staged_requests: 0,
            staged_bytes: 0,
            wasted_bytes: 0,
        }
    }

    pub fn accuracy_bps(&self, tensor: &str) -> Option<u16> {
        let record = self.records.get(tensor)?;
        let samples = record.hits + record.misses;
        if samples == 0 {
            return None;
        }
        Some(((record.hits.saturating_mul(10_000)) / samples) as u16)
    }

    pub fn summary(&self) -> PredictionHistorySummary {
        let samples = self.hits.saturating_add(self.misses);
        PredictionHistorySummary {
            records: self.records.len(),
            total_predictions: self.total_predictions,
            hits: self.hits,
            misses: self.misses,
            staged_requests: self.staged_requests,
            staged_bytes: self.staged_bytes,
            wasted_bytes: self.wasted_bytes,
            accuracy_bps: (samples > 0)
                .then_some(((self.hits.saturating_mul(10_000)) / samples) as u16),
        }
    }

    pub fn reset(&mut self) {
        self.records.clear();
        self.total_predictions = 0;
        self.hits = 0;
        self.misses = 0;
        self.staged_requests = 0;
        self.staged_bytes = 0;
        self.wasted_bytes = 0;
    }

    pub fn record_predictions(&mut self, predictions: &PredictionSet) {
        for prediction in &predictions.predictions {
            self.total_predictions = self.total_predictions.saturating_add(1);
            let record = self.records.entry(prediction.name.clone()).or_default();
            record.predictions = record.predictions.saturating_add(1);
        }
    }

    pub fn record_feedback(
        &mut self,
        predictions: &PredictionSet,
        observed: &ExecutionProfile,
        staged_bytes: &BTreeMap<String, u64>,
    ) {
        let observed_names: BTreeSet<&str> = observed.access.keys().map(String::as_str).collect();
        for prediction in &predictions.predictions {
            let hit = observed_names.contains(prediction.name.as_str());
            let record = self.records.entry(prediction.name.clone()).or_default();
            if hit {
                record.hits = record.hits.saturating_add(1);
                self.hits = self.hits.saturating_add(1);
            } else {
                record.misses = record.misses.saturating_add(1);
                self.misses = self.misses.saturating_add(1);
            }
            if let Some(bytes) = staged_bytes.get(&prediction.name) {
                record.staged_requests = record.staged_requests.saturating_add(1);
                record.staged_bytes = record.staged_bytes.saturating_add(*bytes);
                self.staged_requests = self.staged_requests.saturating_add(1);
                self.staged_bytes = self.staged_bytes.saturating_add(*bytes);
                if !hit {
                    record.wasted_bytes = record.wasted_bytes.saturating_add(*bytes);
                    self.wasted_bytes = self.wasted_bytes.saturating_add(*bytes);
                }
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), PredictionError> {
        if self.records.len() > MAX_PREDICTION_HISTORY_RECORDS {
            return Err(PredictionError::TooManyRecords(self.records.len()));
        }
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| PredictionError::Json(error.to_string()))?;
        let length = bytes.len() as u64;
        if length > MAX_PREDICTION_HISTORY_BYTES {
            return Err(PredictionError::TooLarge {
                bytes: length,
                limit: MAX_PREDICTION_HISTORY_BYTES,
            });
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err(PredictionError::Io(format!(
                "parent directory does not exist: {}",
                parent.display()
            )));
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("prediction-history.json");
        let temp = parent.join(format!(".{name}.tmp.{}", std::process::id()));
        std::fs::write(&temp, &bytes).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            PredictionError::Io(format!("write temporary history: {error}"))
        })?;
        std::fs::rename(&temp, path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            PredictionError::Io(format!("atomic history rename: {error}"))
        })
    }

    pub fn validate(&self, expected: &ProfileIdentity) -> Result<(), PredictionError> {
        if self.schema_version != PREDICTION_HISTORY_SCHEMA_VERSION {
            return Err(PredictionError::Schema(self.schema_version));
        }
        if self.kind != PREDICTION_HISTORY_KIND {
            return Err(PredictionError::Kind(self.kind.clone()));
        }
        if self.identity != *expected {
            return Err(PredictionError::Identity(
                "model/runtime identity does not match".to_string(),
            ));
        }
        if self.records.len() > MAX_PREDICTION_HISTORY_RECORDS {
            return Err(PredictionError::TooManyRecords(self.records.len()));
        }
        Ok(())
    }
}

pub fn load_prediction_history(
    path: &Path,
    expected: &ProfileIdentity,
) -> Result<PredictionHistory, PredictionError> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| PredictionError::Io(format!("metadata: {error}")))?;
    if metadata.len() > MAX_PREDICTION_HISTORY_BYTES {
        return Err(PredictionError::TooLarge {
            bytes: metadata.len(),
            limit: MAX_PREDICTION_HISTORY_BYTES,
        });
    }
    let bytes =
        std::fs::read(path).map_err(|error| PredictionError::Io(format!("read: {error}")))?;
    let history: PredictionHistory =
        serde_json::from_slice(&bytes).map_err(|error| PredictionError::Json(error.to_string()))?;
    history.validate(expected)?;
    Ok(history)
}

/// Build a deterministic prediction set from a complete execution profile.
/// Stable tensor names are matched to the current inventory; ambiguous or
/// unknown names are rejected rather than guessed.
pub fn predict_from_profile(
    config: &PredictionConfig,
    profile: Option<&ExecutionProfile>,
    inventory: &ResourceInventory,
    history: Option<&PredictionHistory>,
) -> PredictionSet {
    let mut set = PredictionSet {
        model_id: inventory.model_id.clone(),
        generation: inventory.generation,
        source: PredictionSource::None,
        profile_version: None,
        complete_observations: 0,
        predictions: Vec::new(),
        rejected: BTreeMap::new(),
        high_confidence: 0,
        medium_confidence: 0,
        low_confidence: 0,
        history_valid: false,
        warnings: Vec::new(),
    };
    if !config.enabled {
        set.rejected.insert("prediction disabled".to_string(), 1);
        return set;
    }
    let Some(profile) = profile else {
        set.rejected.insert("no compatible profile".to_string(), 1);
        return set;
    };
    if !profile.is_valid_for(&inventory.model_id, inventory.generation) {
        set.rejected
            .insert("stale or incomplete profile".to_string(), 1);
        return set;
    }
    set.source = PredictionSource::Profile;
    set.profile_version = Some(profile.version);
    set.complete_observations = profile.complete_observations;
    set.history_valid = history.is_some();
    if let Some(history) = history {
        if history.identity.inventory_generation != inventory.generation.0 {
            set.history_valid = false;
            set.warnings.push(
                "prediction history inventory generation is incompatible; using profile-only confidence"
                    .to_string(),
            );
        }
    }

    let mut by_name: BTreeMap<&str, Vec<&TensorResource>> = BTreeMap::new();
    for resource in &inventory.resources {
        by_name
            .entry(resource.name.as_str())
            .or_default()
            .push(resource);
    }

    let observation_confidence = 5_500u32
        .saturating_add(profile.complete_observations.min(4) as u32 * 1_000)
        .min(10_000) as u16;
    let mut predictions = Vec::new();
    for (name, access) in &profile.access {
        let Some(resources) = by_name.get(name.as_str()) else {
            *set.rejected
                .entry("unknown resource name".to_string())
                .or_default() += 1;
            continue;
        };
        if resources.len() != 1 {
            *set.rejected
                .entry("ambiguous resource name".to_string())
                .or_default() += 1;
            continue;
        }
        let resource = resources[0];
        let history_confidence = history
            .filter(|_| set.history_valid)
            .and_then(|history| history.accuracy_bps(name));
        let confidence_bps = match history_confidence {
            Some(accuracy) => ((observation_confidence as u32 * 2 + accuracy as u32) / 3) as u16,
            None => observation_confidence,
        };
        let confidence = classify(confidence_bps);
        predictions.push(PredictedAccess {
            key: resource.key,
            name: name.clone(),
            predicted_order: access.last_order,
            expected_distance: access.last_order.saturating_sub(access.first_order),
            confidence_bps,
            confidence,
            size_bytes: resource.logical_bytes,
            known_size: resource.logical_bytes > 0,
            reason: format!(
                "profile first_order={} last_order={} consumers={} observations={}",
                access.first_order,
                access.last_order,
                access.observed_consumers,
                profile.complete_observations
            ),
        });
    }
    predictions.sort_by(|a, b| {
        b.confidence_bps
            .cmp(&a.confidence_bps)
            .then_with(|| a.predicted_order.cmp(&b.predicted_order))
            .then_with(|| b.size_bytes.cmp(&a.size_bytes))
            .then_with(|| a.key.tensor_index.cmp(&b.key.tensor_index))
    });
    predictions.truncate(config.lookahead);
    for prediction in &predictions {
        match prediction.confidence {
            PredictionConfidence::High => set.high_confidence += 1,
            PredictionConfidence::Medium => set.medium_confidence += 1,
            PredictionConfidence::Low => set.low_confidence += 1,
        }
    }
    set.predictions = predictions;
    set
}

fn classify(confidence_bps: u16) -> PredictionConfidence {
    if confidence_bps >= 7_500 {
        PredictionConfidence::High
    } else if confidence_bps >= 5_000 {
        PredictionConfidence::Medium
    } else {
        PredictionConfidence::Low
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_profile::AccessPattern;
    use crate::residency_engine::{ModelGeneration, ModelId};
    use forge_core::TensorMappingInfo;

    fn inventory() -> ResourceInventory {
        ResourceInventory::from_tensor_mappings(
            ModelId("m".to_string()),
            ModelGeneration(1),
            &[
                TensorMappingInfo {
                    name: "blk.0.a".to_string(),
                    file_backed: true,
                    mapping_index: Some(0),
                    mapping_size: 1 << 20,
                    offset: 0,
                    size: 4096,
                    page_size: 4096,
                    page_start: 0,
                    page_end: 4096,
                },
                TensorMappingInfo {
                    name: "blk.0.b".to_string(),
                    file_backed: true,
                    mapping_index: Some(0),
                    mapping_size: 1 << 20,
                    offset: 4096,
                    size: 4096,
                    page_size: 4096,
                    page_start: 4096,
                    page_end: 8192,
                },
            ],
        )
        .expect("inventory")
    }

    fn profile() -> ExecutionProfile {
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.complete_observations = 2;
        profile.access.insert(
            "blk.0.a".to_string(),
            AccessPattern {
                first_order: 0,
                last_order: 1,
                observed_consumers: 1,
            },
        );
        profile
    }

    #[test]
    fn config_validation_rejects_invalid_bounds() {
        let mut config = PredictionConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        config.max_in_flight = 0;
        assert!(config.validate().is_err());
        config.max_in_flight = 1;
        config.staging_window_bytes = 0;
        assert!(config.validate().is_err());
        config.staging_window_bytes = 100;
        config.speculative_budget_bytes = 101;
        assert!(config.validate().is_err());
    }

    #[test]
    fn prediction_is_deterministic_and_profile_bound() {
        let config = PredictionConfig {
            enabled: true,
            ..Default::default()
        };
        let first = predict_from_profile(&config, Some(&profile()), &inventory(), None);
        let second = predict_from_profile(&config, Some(&profile()), &inventory(), None);
        assert_eq!(first, second);
        assert_eq!(first.source, PredictionSource::Profile);
        assert_eq!(first.predictions.len(), 1);
        assert_eq!(first.predictions[0].name, "blk.0.a");
        assert!(first.predictions[0].known_size);
    }

    #[test]
    fn stale_or_disabled_profile_produces_no_prediction() {
        let config = PredictionConfig::default();
        let disabled = predict_from_profile(&config, Some(&profile()), &inventory(), None);
        assert!(disabled.predictions.is_empty());
        let enabled = PredictionConfig {
            enabled: true,
            ..Default::default()
        };
        let stale = predict_from_profile(&enabled, None, &inventory(), None);
        assert!(stale.predictions.is_empty());
        assert!(stale.rejected.contains_key("no compatible profile"));
        let mut mismatched = profile();
        mismatched.generation = ModelGeneration(2);
        let stale = predict_from_profile(&enabled, Some(&mismatched), &inventory(), None);
        assert!(stale.predictions.is_empty());
        assert!(stale.rejected.contains_key("stale or incomplete profile"));
        let mut incomplete = profile();
        incomplete.complete_observations = 0;
        let stale = predict_from_profile(&enabled, Some(&incomplete), &inventory(), None);
        assert!(stale.predictions.is_empty());
    }

    #[test]
    fn history_rejects_identity_and_schema_mismatch() {
        let identity = ProfileIdentity {
            model_path: "m".to_string(),
            model_size_bytes: 1,
            model_modified_unix_ns: None,
            inventory_generation: 1,
            native_contract_version: 1,
        };
        let mut history = PredictionHistory::new(identity.clone());
        history.records.insert(
            "blk.0.a".to_string(),
            PredictionFeedbackRecord {
                predictions: 1,
                ..PredictionFeedbackRecord::default()
            },
        );
        assert!(history.validate(&identity).is_ok());
        let output = std::env::temp_dir().join(format!(
            "ramforge-prediction-{}-identity.json",
            std::process::id()
        ));
        history.save(&output).expect("save");
        let mut wrong = identity.clone();
        wrong.inventory_generation = 2;
        assert!(matches!(
            load_prediction_history(&output, &wrong),
            Err(PredictionError::Identity(_))
        ));
        let mut schema = history.clone();
        schema.schema_version = 999;
        schema.save(&output).expect("save schema");
        assert!(matches!(
            load_prediction_history(&output, &identity),
            Err(PredictionError::Schema(999))
        ));
        history.reset();
        assert!(history.records.is_empty());
        assert_eq!(history.hits, 0);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn history_feedback_requires_stable_identity() {
        let identity = ProfileIdentity {
            model_path: "m".to_string(),
            model_size_bytes: 1,
            model_modified_unix_ns: None,
            inventory_generation: 1,
            native_contract_version: 1,
        };
        let mut history = PredictionHistory::new(identity.clone());
        let config = PredictionConfig {
            enabled: true,
            ..Default::default()
        };
        let predictions =
            predict_from_profile(&config, Some(&profile()), &inventory(), Some(&history));
        history.record_predictions(&predictions);
        history.record_feedback(&predictions, &profile(), &BTreeMap::new());
        assert_eq!(history.hits, 1);
        assert_eq!(history.misses, 0);
        let output = std::env::temp_dir().join(format!(
            "ramforge-prediction-{}-roundtrip.json",
            std::process::id()
        ));
        let output2 = std::env::temp_dir().join(format!(
            "ramforge-prediction-{}-roundtrip2.json",
            std::process::id()
        ));
        history.save(&output).expect("save");
        history.save(&output2).expect("save2");
        assert_eq!(
            std::fs::read(&output).expect("first bytes"),
            std::fs::read(&output2).expect("second bytes")
        );
        let loaded = load_prediction_history(&output, &identity).expect("load");
        assert_eq!(loaded.hits, 1);
        let _ = std::fs::remove_file(&output);
        let _ = std::fs::remove_file(&output2);
    }
}
