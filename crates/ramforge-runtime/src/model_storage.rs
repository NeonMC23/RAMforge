//! Model storage assessment and explicit verified copy.
//!
//! Copy is never automatic and never deletes the source. The caller must
//! explicitly authorize it. A faster device is not claimed to improve tokens/s.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::machine_profile::storage::collect_storage_profile;
use crate::machine_profile::StorageProfile;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestinationAssessment {
    pub path: String,
    pub filesystem: Option<String>,
    pub available_bytes: Option<u64>,
    pub required_bytes: u64,
    pub sufficient_space: Option<bool>,
    pub read_only: Option<bool>,
    pub recommended: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageRecommendation {
    pub schema_version: u32,
    pub model_path: String,
    pub model_bytes: u64,
    pub source: StorageProfile,
    pub destinations: Vec<DestinationAssessment>,
    pub recommended_destination: Option<String>,
    pub limitations: Vec<String>,
}

impl StorageRecommendation {
    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CopyReport {
    pub schema_version: u32,
    pub source_path: String,
    pub destination_path: String,
    pub source_bytes: u64,
    pub copied_bytes: u64,
    pub fingerprint: String,
    pub verified: bool,
    pub source_preserved: bool,
    pub notes: Vec<String>,
}

impl CopyReport {
    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|error| error.to_string())
    }
}

pub const MODEL_STORAGE_SCHEMA_VERSION: u32 = 1;

/// Assess destinations without modifying anything.
pub fn assess_storage(
    model_path: &Path,
    destinations: &[PathBuf],
    sequence: u64,
) -> Result<StorageRecommendation, String> {
    let model_bytes = std::fs::metadata(model_path)
        .map_err(|error| error.to_string())?
        .len();
    let source = collect_storage_profile(model_path, sequence);
    let mut assessments = Vec::new();
    for destination in destinations {
        let profile = collect_storage_profile(destination, sequence);
        let available = profile.available_bytes.as_ref().copied();
        let sufficient = available.map(|available| available >= model_bytes);
        let read_only = profile.read_only.as_ref().copied();
        let mut reasons = Vec::new();
        if sufficient == Some(false) {
            reasons.push("destination has insufficient free space".to_string());
        }
        if read_only == Some(true) {
            reasons.push("destination filesystem is read-only".to_string());
        }
        if sufficient.is_none() {
            reasons.push("destination free space is unavailable".to_string());
        }
        let recommended = sufficient == Some(true) && read_only != Some(true);
        assessments.push(DestinationAssessment {
            path: destination.to_string_lossy().to_string(),
            filesystem: profile.filesystem.as_ref().cloned(),
            available_bytes: available,
            required_bytes: model_bytes,
            sufficient_space: sufficient,
            read_only,
            recommended,
            reasons,
        });
    }
    let recommended_destination = assessments
        .iter()
        .find(|assessment| assessment.recommended)
        .map(|assessment| assessment.path.clone());
    Ok(StorageRecommendation {
        schema_version: MODEL_STORAGE_SCHEMA_VERSION,
        model_path: model_path.to_string_lossy().to_string(),
        model_bytes,
        source,
        destinations: assessments,
        recommended_destination,
        limitations: vec![
            "Storage recommendations are capacity/read-only checks, not throughput guarantees."
                .to_string(),
            "A faster device does not automatically increase token generation throughput."
                .to_string(),
            "No copy or deletion is performed by this assessment.".to_string(),
        ],
    })
}

/// Explicitly copy and verify a model. The source is never removed or modified.
pub fn copy_model_verified(
    source: &Path,
    destination: &Path,
    overwrite: bool,
) -> Result<CopyReport, String> {
    if source == destination {
        return Err("source and destination are the same path".to_string());
    }
    if destination.exists() && !overwrite {
        return Err(format!(
            "destination already exists: {} (explicit overwrite is required)",
            destination.display()
        ));
    }
    let source_bytes = std::fs::metadata(source)
        .map_err(|error| error.to_string())?
        .len();
    let fingerprint = file_fingerprint(source)?;
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temporary = destination.with_extension(format!("ramforge-tmp-{}", std::process::id()));
    let result = (|| {
        let mut input = std::fs::File::open(source).map_err(|error| error.to_string())?;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        let copied = std::io::copy(&mut input, &mut output).map_err(|error| error.to_string())?;
        output.sync_all().map_err(|error| error.to_string())?;
        if copied != source_bytes {
            return Err(format!(
                "copied byte count {copied} does not match source {source_bytes}"
            ));
        }
        let copied_fingerprint = file_fingerprint(&temporary)?;
        if copied_fingerprint != fingerprint {
            return Err("copied file fingerprint does not match source".to_string());
        }
        if destination.exists() {
            std::fs::remove_file(destination).map_err(|error| error.to_string())?;
        }
        std::fs::rename(&temporary, destination).map_err(|error| error.to_string())?;
        Ok::<_, String>((copied, copied_fingerprint))
    })();
    match result {
        Ok((copied, copied_fingerprint)) => Ok(CopyReport {
            schema_version: MODEL_STORAGE_SCHEMA_VERSION,
            source_path: source.to_string_lossy().to_string(),
            destination_path: destination.to_string_lossy().to_string(),
            source_bytes,
            copied_bytes: copied,
            fingerprint: copied_fingerprint,
            verified: true,
            source_preserved: true,
            notes: vec![
                "Source file was preserved unchanged.".to_string(),
                "Destination was verified by byte count and bounded fingerprint.".to_string(),
                "No performance benefit is implied by copying.".to_string(),
            ],
        }),
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            Err(error)
        }
    }
}

fn file_fingerprint(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let size = file.metadata().map_err(|error| error.to_string())?.len();
    let mut header = vec![0u8; 64 * 1024];
    let read = file.read(&mut header).map_err(|error| error.to_string())?;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in &header[..read] {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= size;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    Ok(format!("fnv1a64:{hash:016x}:size={size}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn copy_is_verified_and_preserves_source() {
        let directory =
            std::env::temp_dir().join(format!("ramforge-copy-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("source.gguf");
        let destination = directory.join("dest.gguf");
        let mut file = std::fs::File::create(&source).unwrap();
        file.write_all(&[1u8, 2, 3, 4, 5]).unwrap();
        drop(file);
        let report = copy_model_verified(&source, &destination, false).unwrap();
        assert!(report.verified);
        assert!(report.source_preserved);
        assert_eq!(report.copied_bytes, 5);
        assert_eq!(
            std::fs::read(&source).unwrap(),
            std::fs::read(&destination).unwrap()
        );
        assert!(copy_model_verified(&source, &destination, false).is_err());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn assessment_is_non_destructive() {
        let directory =
            std::env::temp_dir().join(format!("ramforge-storage-assess-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let model = directory.join("model.gguf");
        std::fs::write(&model, vec![0u8; 1024]).unwrap();
        let report = assess_storage(&model, std::slice::from_ref(&directory), 1).unwrap();
        assert_eq!(report.model_bytes, 1024);
        assert!(!report.destinations.is_empty());
        let _ = std::fs::remove_dir_all(directory);
    }
}
