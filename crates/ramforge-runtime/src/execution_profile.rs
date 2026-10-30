//! Bounded, in-memory execution profiles derived from graph manifests.
//!
//! Profiles contain stable tensor names and observation counters only. Native
//! node identities are never persisted.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::UNIX_EPOCH;

use forge_core::TraceEvent;
use serde::{Deserialize, Serialize};

use crate::graph_manifest::GraphManifest;
use crate::residency_engine::{ModelGeneration, ModelId};

pub const EXECUTION_PROFILE_VERSION: u32 = 1;
pub const PROFILE_SCHEMA_VERSION: u32 = 1;
pub const PROFILE_KIND: &str = "ramforge.execution_profile";
pub const MAX_PROFILE_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_PROFILE_ENTRIES: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessPattern {
    pub first_order: usize,
    pub last_order: usize,
    pub observed_consumers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProfile {
    pub version: u32,
    pub model_id: ModelId,
    pub generation: ModelGeneration,
    pub complete_observations: u64,
    pub incomplete_observations: u64,
    pub last_node_count: usize,
    pub last_tensor_count: usize,
    pub access: BTreeMap<String, AccessPattern>,
    pub invalidated_reason: Option<String>,
}

impl ExecutionProfile {
    pub fn new(model_id: ModelId, generation: ModelGeneration) -> Self {
        Self {
            version: EXECUTION_PROFILE_VERSION,
            model_id,
            generation,
            complete_observations: 0,
            incomplete_observations: 0,
            last_node_count: 0,
            last_tensor_count: 0,
            access: BTreeMap::new(),
            invalidated_reason: None,
        }
    }

    /// Update from bounded trace events. A capture can contain several graph
    /// executions; each execution is analyzed separately so transient native
    /// node IDs are never merged across executions.
    pub fn update_from_trace(&mut self, events: &[TraceEvent], trace_complete: bool) {
        let mut execution_ids = std::collections::BTreeSet::new();
        for event in events {
            execution_ids.insert(event.execution_id);
        }
        if execution_ids.is_empty() {
            self.incomplete_observations += 1;
            return;
        }
        for execution_id in execution_ids {
            let execution_events: Vec<TraceEvent> = events
                .iter()
                .filter(|event| event.execution_id == execution_id)
                .cloned()
                .collect();
            let manifest = GraphManifest::from_trace(&execution_events);
            self.update(&manifest, trace_complete);
        }
    }

    /// Update from a manifest. Incomplete traces are counted but do not
    /// overwrite a previously complete access map.
    pub fn update(&mut self, manifest: &GraphManifest, trace_complete: bool) {
        if manifest.execution_id.is_none() {
            self.incomplete_observations += 1;
            return;
        }
        let usable = trace_complete
            && manifest.edges_known
            && manifest.completion_proven
            && manifest.dependency_cycles() == 0
            && !manifest.intra_decode_safe;
        self.last_node_count = manifest.nodes.len();
        self.last_tensor_count = manifest.tensor_to_nodes.len();
        if !usable {
            self.incomplete_observations += 1;
            return;
        }
        self.complete_observations += 1;
        let mut access = BTreeMap::new();
        for (tensor, consumers) in &manifest.tensor_to_nodes {
            let first = consumers.iter().copied().min().unwrap_or(0);
            let last = consumers.iter().copied().max().unwrap_or(0);
            access.insert(
                tensor.clone(),
                AccessPattern {
                    first_order: first,
                    last_order: last,
                    observed_consumers: consumers.len(),
                },
            );
        }
        self.access = access;
        self.invalidated_reason = None;
    }

    pub fn invalidate(&mut self, reason: impl Into<String>) {
        self.access.clear();
        self.last_node_count = 0;
        self.last_tensor_count = 0;
        self.invalidated_reason = Some(reason.into());
    }

    pub fn is_valid_for(&self, model_id: &ModelId, generation: ModelGeneration) -> bool {
        self.model_id == *model_id
            && self.generation == generation
            && self.complete_observations > 0
            && self.invalidated_reason.is_none()
    }

    pub fn last_order(&self, tensor_name: &str) -> Option<usize> {
        self.access
            .get(tensor_name)
            .map(|pattern| pattern.last_order)
    }

    pub fn is_complete(&self) -> bool {
        self.complete_observations > 0 && self.invalidated_reason.is_none()
    }
}

/// Stable file/model identity stored with a profile. Transient native node IDs
/// are deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileIdentity {
    pub model_path: String,
    pub model_size_bytes: u64,
    pub model_modified_unix_ns: Option<u64>,
    pub inventory_generation: u64,
    pub native_contract_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProfileFile {
    pub schema_version: u32,
    pub kind: String,
    pub identity: ProfileIdentity,
    pub profile: ExecutionProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileInspection {
    pub schema_version: u32,
    pub kind: String,
    pub identity: ProfileIdentity,
    pub profile_version: u32,
    pub complete_observations: u64,
    pub incomplete_observations: u64,
    pub entries: usize,
    pub invalidated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    Io(String),
    Json(String),
    TooLarge {
        bytes: u64,
        limit: u64,
    },
    Schema(u32),
    Kind(String),
    ProfileVersion(u32),
    NativeContract(u32),
    ModelPath(String),
    ModelSize {
        expected: u64,
        actual: u64,
    },
    ModelModified {
        expected: Option<u64>,
        actual: Option<u64>,
    },
    InventoryGeneration {
        expected: u64,
        actual: u64,
    },
    Incomplete,
    Invalidated(String),
    TooManyEntries(usize),
    InvalidEntry(String),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "profile I/O: {message}"),
            Self::Json(message) => write!(formatter, "profile JSON: {message}"),
            Self::TooLarge { bytes, limit } => {
                write!(formatter, "profile is {bytes} bytes; limit is {limit}")
            }
            Self::Schema(version) => write!(formatter, "unsupported profile schema {version}"),
            Self::Kind(kind) => write!(formatter, "unexpected profile kind {kind:?}"),
            Self::ProfileVersion(version) => {
                write!(formatter, "unsupported profile version {version}")
            }
            Self::NativeContract(version) => {
                write!(formatter, "native residency contract mismatch: {version}")
            }
            Self::ModelPath(path) => write!(formatter, "profile model path mismatch: {path}"),
            Self::ModelSize { expected, actual } => {
                write!(
                    formatter,
                    "profile model size mismatch: {actual} != {expected}"
                )
            }
            Self::ModelModified { expected, actual } => write!(
                formatter,
                "profile model modification mismatch: {actual:?} != {expected:?}"
            ),
            Self::InventoryGeneration { expected, actual } => write!(
                formatter,
                "profile inventory generation mismatch: {actual} != {expected}"
            ),
            Self::Incomplete => formatter.write_str("profile has no complete observation"),
            Self::Invalidated(reason) => write!(formatter, "profile is invalidated: {reason}"),
            Self::TooManyEntries(entries) => {
                write!(formatter, "profile has too many entries: {entries}")
            }
            Self::InvalidEntry(entry) => write!(formatter, "invalid profile entry: {entry}"),
        }
    }
}

impl std::error::Error for ProfileError {}

impl ProfileIdentity {
    pub fn from_model_path(
        model_path: &Path,
        model_size_bytes: u64,
        inventory_generation: u64,
        native_contract_version: u32,
    ) -> Result<Self, ProfileError> {
        let canonical = model_path
            .canonicalize()
            .map_err(|error| ProfileError::Io(format!("canonicalize model path: {error}")))?;
        let metadata = std::fs::metadata(&canonical)
            .map_err(|error| ProfileError::Io(format!("model metadata: {error}")))?;
        let modified = metadata.modified().ok().and_then(|time| {
            time.duration_since(UNIX_EPOCH)
                .ok()
                .map(|duration| duration.as_nanos().min(u64::MAX as u128) as u64)
        });
        Ok(Self {
            model_path: canonical.to_string_lossy().into_owned(),
            model_size_bytes,
            model_modified_unix_ns: modified,
            inventory_generation,
            native_contract_version,
        })
    }
}

impl ExecutionProfile {
    /// Deterministic, bounded, validated profile serialization.
    pub fn to_json_file(&self, identity: &ProfileIdentity) -> Result<Vec<u8>, ProfileError> {
        validate_for_persistence(self)?;
        if self.model_id.0 != identity.model_path {
            return Err(ProfileError::ModelPath(self.model_id.0.clone()));
        }
        if self.generation.0 != identity.inventory_generation {
            return Err(ProfileError::InventoryGeneration {
                expected: identity.inventory_generation,
                actual: self.generation.0,
            });
        }
        let file = ExecutionProfileFile {
            schema_version: PROFILE_SCHEMA_VERSION,
            kind: PROFILE_KIND.to_string(),
            identity: identity.clone(),
            profile: self.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&file)
            .map_err(|error| ProfileError::Json(error.to_string()))?;
        let length = bytes.len() as u64;
        if length > MAX_PROFILE_BYTES {
            return Err(ProfileError::TooLarge {
                bytes: length,
                limit: MAX_PROFILE_BYTES,
            });
        }
        Ok(bytes)
    }

    pub fn save_json_file(
        &self,
        identity: &ProfileIdentity,
        path: &Path,
    ) -> Result<(), ProfileError> {
        let bytes = self.to_json_file(identity)?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        if !parent.is_dir() {
            return Err(ProfileError::Io(format!(
                "profile parent directory does not exist: {}",
                parent.display()
            )));
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("profile.json");
        let temp = parent.join(format!(".{file_name}.tmp.{}", std::process::id()));
        std::fs::write(&temp, &bytes).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            ProfileError::Io(format!("write temporary profile: {error}"))
        })?;
        if let Ok(file) = std::fs::File::open(&temp) {
            let _ = file.sync_all();
        }
        std::fs::rename(&temp, path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            ProfileError::Io(format!("atomic profile rename: {error}"))
        })
    }
}

pub fn load_profile(
    path: &Path,
    expected: &ProfileIdentity,
) -> Result<ExecutionProfile, ProfileError> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| ProfileError::Io(format!("profile metadata: {error}")))?;
    if metadata.len() > MAX_PROFILE_BYTES {
        return Err(ProfileError::TooLarge {
            bytes: metadata.len(),
            limit: MAX_PROFILE_BYTES,
        });
    }
    let bytes =
        std::fs::read(path).map_err(|error| ProfileError::Io(format!("read profile: {error}")))?;
    let file: ExecutionProfileFile =
        serde_json::from_slice(&bytes).map_err(|error| ProfileError::Json(error.to_string()))?;
    if file.schema_version != PROFILE_SCHEMA_VERSION {
        return Err(ProfileError::Schema(file.schema_version));
    }
    if file.kind != PROFILE_KIND {
        return Err(ProfileError::Kind(file.kind));
    }
    if file.profile.version != EXECUTION_PROFILE_VERSION {
        return Err(ProfileError::ProfileVersion(file.profile.version));
    }
    if file.identity.native_contract_version != expected.native_contract_version {
        return Err(ProfileError::NativeContract(
            file.identity.native_contract_version,
        ));
    }
    if file.identity.model_path != expected.model_path {
        return Err(ProfileError::ModelPath(file.identity.model_path));
    }
    if file.identity.model_size_bytes != expected.model_size_bytes {
        return Err(ProfileError::ModelSize {
            expected: expected.model_size_bytes,
            actual: file.identity.model_size_bytes,
        });
    }
    if file.identity.model_modified_unix_ns != expected.model_modified_unix_ns {
        return Err(ProfileError::ModelModified {
            expected: expected.model_modified_unix_ns,
            actual: file.identity.model_modified_unix_ns,
        });
    }
    if file.identity.inventory_generation != expected.inventory_generation {
        return Err(ProfileError::InventoryGeneration {
            expected: expected.inventory_generation,
            actual: file.identity.inventory_generation,
        });
    }
    if file.profile.model_id.0 != file.identity.model_path {
        return Err(ProfileError::ModelPath(file.profile.model_id.0.clone()));
    }
    if file.profile.generation.0 != file.identity.inventory_generation {
        return Err(ProfileError::InventoryGeneration {
            expected: file.identity.inventory_generation,
            actual: file.profile.generation.0,
        });
    }
    validate_for_persistence(&file.profile)?;
    if !file
        .profile
        .is_valid_for(&file.profile.model_id, file.profile.generation)
    {
        return Err(ProfileError::Incomplete);
    }
    Ok(file.profile)
}

pub fn inspect_profile(path: &Path) -> Result<ProfileInspection, ProfileError> {
    let metadata = std::fs::metadata(path)
        .map_err(|error| ProfileError::Io(format!("profile metadata: {error}")))?;
    if metadata.len() > MAX_PROFILE_BYTES {
        return Err(ProfileError::TooLarge {
            bytes: metadata.len(),
            limit: MAX_PROFILE_BYTES,
        });
    }
    let bytes =
        std::fs::read(path).map_err(|error| ProfileError::Io(format!("read profile: {error}")))?;
    let file: ExecutionProfileFile =
        serde_json::from_slice(&bytes).map_err(|error| ProfileError::Json(error.to_string()))?;
    if file.schema_version != PROFILE_SCHEMA_VERSION {
        return Err(ProfileError::Schema(file.schema_version));
    }
    if file.kind != PROFILE_KIND {
        return Err(ProfileError::Kind(file.kind));
    }
    if file.profile.version != EXECUTION_PROFILE_VERSION {
        return Err(ProfileError::ProfileVersion(file.profile.version));
    }
    validate_for_persistence(&file.profile)?;
    Ok(ProfileInspection {
        schema_version: file.schema_version,
        kind: file.kind,
        identity: file.identity,
        profile_version: file.profile.version,
        complete_observations: file.profile.complete_observations,
        incomplete_observations: file.profile.incomplete_observations,
        entries: file.profile.access.len(),
        invalidated: file.profile.invalidated_reason.is_some(),
    })
}

fn validate_for_persistence(profile: &ExecutionProfile) -> Result<(), ProfileError> {
    if profile.invalidated_reason.is_some() {
        return Err(ProfileError::Invalidated(
            profile.invalidated_reason.clone().unwrap_or_default(),
        ));
    }
    if profile.complete_observations == 0 || profile.access.is_empty() {
        return Err(ProfileError::Incomplete);
    }
    if profile.access.len() > MAX_PROFILE_ENTRIES {
        return Err(ProfileError::TooManyEntries(profile.access.len()));
    }
    for (name, pattern) in &profile.access {
        if name.is_empty() || name.len() > 4096 {
            return Err(ProfileError::InvalidEntry(format!(
                "tensor name length {}",
                name.len()
            )));
        }
        if pattern.first_order > pattern.last_order || pattern.observed_consumers == 0 {
            return Err(ProfileError::InvalidEntry(format!(
                "tensor {name} has inconsistent access range"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_manifest::{GraphManifest, LastUse};
    use forge_core::{TraceCompletion, TracePhase};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn manifest() -> GraphManifest {
        GraphManifest {
            execution_id: Some(1),
            nodes: Vec::new(),
            tensor_to_nodes: BTreeMap::from([
                ("blk.0.weight".to_string(), vec![0, 3]),
                ("output.weight".to_string(), vec![5]),
            ]),
            last_use: BTreeMap::<String, LastUse>::new(),
            edges_known: true,
            completion_proven: true,
            intra_decode_safe: false,
        }
    }

    #[test]
    fn complete_profile_records_names_and_orders() {
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.update(&manifest(), true);
        assert_eq!(profile.complete_observations, 1);
        assert_eq!(profile.last_order("blk.0.weight"), Some(3));
        assert_eq!(profile.last_order("output.weight"), Some(5));
        assert!(profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(1)));
        assert_eq!(profile.access.len(), 2);
    }

    #[test]
    fn incomplete_trace_does_not_overwrite_profile() {
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.update(&manifest(), true);
        let mut incomplete = manifest();
        incomplete.edges_known = false;
        profile.update(&incomplete, false);
        assert_eq!(profile.complete_observations, 1);
        assert_eq!(profile.incomplete_observations, 1);
        assert_eq!(profile.last_order("blk.0.weight"), Some(3));
    }

    #[test]
    fn multi_execution_trace_is_not_merged_by_native_node_id() {
        let event = |execution_id, sequence, name: &str| TraceEvent {
            execution_id,
            sequence,
            node_id: 1,
            node_name: Some("node".to_string()),
            op_name: None,
            bytes: 4096,
            dependencies: vec![name.to_string()],
            dependency_ids: vec![2],
            phase: TracePhase::Requested,
            completion: TraceCompletion::NotCompleted,
        };
        let completed = |execution_id, sequence| TraceEvent {
            execution_id,
            sequence,
            node_id: 1,
            node_name: Some("node".to_string()),
            op_name: None,
            bytes: 4096,
            dependencies: Vec::new(),
            dependency_ids: Vec::new(),
            phase: TracePhase::CompletedSynchronized,
            completion: TraceCompletion::BackendSynchronized,
        };
        let events = vec![
            event(1, 0, "blk.0.a"),
            completed(1, 1),
            event(2, 0, "blk.0.b"),
            completed(2, 1),
        ];
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.update_from_trace(&events, true);
        assert_eq!(profile.complete_observations, 2);
        assert_eq!(profile.last_order("blk.0.a"), None);
        assert_eq!(profile.last_order("blk.0.b"), Some(0));
        assert!(profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(1)));
    }

    #[test]
    fn invalidation_and_generation_mismatch_disable_profile() {
        let mut profile = ExecutionProfile::new(ModelId("m".to_string()), ModelGeneration(1));
        profile.update(&manifest(), true);
        assert!(profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(1)));
        assert!(!profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(2)));
        profile.invalidate("reload");
        assert!(!profile.is_valid_for(&ModelId("m".to_string()), ModelGeneration(1)));
    }

    fn sample_profile(path: &str, generation: u64) -> ExecutionProfile {
        let mut profile =
            ExecutionProfile::new(ModelId(path.to_string()), ModelGeneration(generation));
        profile.complete_observations = 2;
        profile.last_node_count = 3;
        profile.last_tensor_count = 1;
        profile.access.insert(
            "blk.0.attn.weight".to_string(),
            AccessPattern {
                first_order: 0,
                last_order: 2,
                observed_consumers: 1,
            },
        );
        profile
    }

    fn temp_model(tag: &str) -> (PathBuf, ProfileIdentity) {
        let path = std::env::temp_dir().join(format!(
            "ramforge-profile-{tag}-{}-{}.gguf",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"profile identity fixture").expect("model fixture");
        let canonical = path.canonicalize().expect("canonical");
        let identity = ProfileIdentity::from_model_path(&canonical, 22, 1, 1).expect("identity");
        (canonical, identity)
    }

    #[test]
    fn profile_roundtrip_is_deterministic_and_atomic() {
        let (path, identity) = temp_model("roundtrip");
        let profile = sample_profile(&identity.model_path, 1);
        let first = profile.to_json_file(&identity).expect("json");
        let second = profile.to_json_file(&identity).expect("json");
        assert_eq!(first, second);
        let output = path.with_extension("profile.json");
        profile.save_json_file(&identity, &output).expect("save");
        let loaded = load_profile(&output, &identity).expect("load");
        assert_eq!(loaded.access, profile.access);
        assert_eq!(loaded.complete_observations, 2);
        let inspection = inspect_profile(&output).expect("inspect");
        assert_eq!(inspection.entries, 1);
        assert!(!inspection.invalidated);
        let _ = std::fs::remove_file(&output);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn profile_rejects_model_identity_mismatch() {
        let (path, identity) = temp_model("mismatch");
        let profile = sample_profile(&identity.model_path, 1);
        let output = path.with_extension("profile.json");
        profile.save_json_file(&identity, &output).expect("save");
        let mut other = identity.clone();
        other.model_size_bytes += 1;
        assert!(matches!(
            load_profile(&output, &other),
            Err(ProfileError::ModelSize { .. })
        ));
        let mut stale = identity.clone();
        stale.inventory_generation = 2;
        assert!(matches!(
            load_profile(&output, &stale),
            Err(ProfileError::InventoryGeneration { .. })
        ));
        let _ = std::fs::remove_file(&output);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn profile_rejects_incomplete_corrupt_and_schema_mismatch() {
        let (path, identity) = temp_model("invalid");
        let output = path.with_extension("profile.json");
        std::fs::write(&output, b"{ not json").expect("corrupt");
        assert!(matches!(
            load_profile(&output, &identity),
            Err(ProfileError::Json(_))
        ));
        let mut incomplete = sample_profile(&identity.model_path, 1);
        incomplete.complete_observations = 0;
        assert!(matches!(
            incomplete.to_json_file(&identity),
            Err(ProfileError::Incomplete)
        ));
        let schema = serde_json::json!({
            "schema_version": 999,
            "kind": PROFILE_KIND,
            "identity": identity,
            "profile": sample_profile(&identity.model_path, 1),
        });
        std::fs::write(&output, serde_json::to_vec(&schema).unwrap()).expect("schema");
        assert!(matches!(
            load_profile(&output, &identity),
            Err(ProfileError::Schema(999))
        ));
        let _ = std::fs::remove_file(&output);
        let _ = std::fs::remove_file(&path);
    }
}
