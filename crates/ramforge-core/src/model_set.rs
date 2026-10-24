//! GGUF model-set discovery and split-shard validation.
//!
//! A model is no longer required to be a single `.gguf` file. This module
//! resolves either a single file, a standard split GGUF shard, or a directory
//! containing one model's shard set. Parsing remains header-only: discovery
//! reads metadata and tensor descriptors, never tensor payloads.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{GgufError, Result};
use crate::gguf::parse_gguf_file;
use crate::model::GgufModel;

pub const SPLIT_COUNT_KEY: &str = "split.count";
pub const SPLIT_NO_KEY: &str = "split.no";
pub const SPLIT_TENSOR_COUNT_KEY: &str = "split.tensors.count";

/// Physical shard layout used by the file-backed data source.
///
/// For a non-sharded file, the virtual range is the whole file and
/// `physical_data_start` is zero, preserving absolute GGUF offsets. For a
/// sharded model, tensor data sections are concatenated into a virtual range;
/// `physical_data_start` is the shard's real data-section offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufShardLayout {
    pub path: PathBuf,
    pub file_size: u64,
    pub data_start_offset: u64,
    pub data_length: u64,
    pub virtual_start: u64,
    pub virtual_end: u64,
    pub physical_data_start: u64,
}

/// One resolved GGUF model, possibly composed of multiple shards.
#[derive(Debug, Clone)]
pub struct GgufModelSet {
    /// Path supplied by the caller (file or directory).
    pub path: PathBuf,
    /// Shards ordered by `split.no` (or one entry for a single-file model).
    pub shards: Vec<GgufModel>,
}

impl GgufModelSet {
    pub fn is_sharded(&self) -> bool {
        self.shards.len() > 1
    }

    /// Path of shard zero, suitable for native loaders that resolve siblings.
    pub fn primary_path(&self) -> &Path {
        &self.shards[0].path
    }

    pub fn total_file_bytes(&self) -> u64 {
        self.shards
            .iter()
            .fold(0u64, |total, shard| total.saturating_add(shard.file_size))
    }

    pub fn total_tensor_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        for shard in &self.shards {
            total = total.checked_add(shard.total_tensor_bytes()?)?;
        }
        Some(total)
    }

    /// Size of the virtual data space consumed by the data source.
    pub fn virtual_size(&self) -> u64 {
        if self.is_sharded() {
            self.shards.iter().fold(0u64, |total, shard| {
                total.saturating_add(shard.file_size.saturating_sub(shard.data_start_offset))
            })
        } else {
            self.shards[0].file_size
        }
    }

    /// Physical-to-virtual shard mapping in shard order.
    pub fn shard_layouts(&self) -> Vec<GgufShardLayout> {
        if !self.is_sharded() {
            let shard = &self.shards[0];
            return vec![GgufShardLayout {
                path: shard.path.clone(),
                file_size: shard.file_size,
                data_start_offset: shard.data_start_offset,
                data_length: shard.file_size,
                virtual_start: 0,
                virtual_end: shard.file_size,
                physical_data_start: 0,
            }];
        }

        let mut virtual_start = 0u64;
        self.shards
            .iter()
            .map(|shard| {
                let data_length = shard.file_size.saturating_sub(shard.data_start_offset);
                let layout = GgufShardLayout {
                    path: shard.path.clone(),
                    file_size: shard.file_size,
                    data_start_offset: shard.data_start_offset,
                    data_length,
                    virtual_start,
                    virtual_end: virtual_start.saturating_add(data_length),
                    physical_data_start: shard.data_start_offset,
                };
                virtual_start = layout.virtual_end;
                layout
            })
            .collect()
    }

    /// Merge shard metadata and descriptors into a single execution view.
    ///
    /// Model-level metadata comes from shard zero, which is where GGUF split
    /// files carry architecture and tokenizer keys. Tensor descriptors are
    /// remapped to the virtual data space described by [`Self::shard_layouts`].
    pub fn merged_model(&self) -> GgufModel {
        let mut merged = self.shards[0].clone();
        merged.path = self.shards[0].path.clone();
        merged.file_size = self.total_file_bytes();

        if !self.is_sharded() {
            return merged;
        }

        let mut tensors = Vec::new();
        let mut virtual_start = 0u64;
        for shard in &self.shards {
            let data_length = shard.file_size.saturating_sub(shard.data_start_offset);
            for descriptor in &shard.tensors {
                let mut descriptor = descriptor.clone();
                let relative = descriptor
                    .file_offset
                    .saturating_sub(shard.data_start_offset);
                descriptor.file_offset = virtual_start.saturating_add(relative);
                descriptor.offset = descriptor.file_offset;
                tensors.push(descriptor);
            }
            virtual_start = virtual_start.saturating_add(data_length);
        }

        merged.tensors = tensors;
        merged.data_start_offset = 0;
        merged
    }
}

/// Resolve a path to one complete GGUF model set.
///
/// Accepted inputs:
/// - a single non-split `.gguf` file;
/// - any shard of a standard split set named
///   `<prefix>-00001-of-00002.gguf`;
/// - a directory containing exactly one model's `.gguf` file(s).
pub fn discover_model_set<P: AsRef<Path>>(path: P) -> Result<GgufModelSet> {
    let path = path.as_ref();
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        discover_from_file(path)
    } else if metadata.is_dir() {
        discover_from_dir(path)
    } else {
        Err(GgufError::General(format!(
            "model path is neither a file nor a directory: {}",
            path.display()
        )))
    }
}

fn discover_from_file(path: &Path) -> Result<GgufModelSet> {
    let model = parse_gguf_file(path)?;
    let split_count = split_count(&model)?;
    if split_count <= 1 {
        return Ok(GgufModelSet {
            path: path.to_path_buf(),
            shards: vec![model],
        });
    }

    let (prefix, filename_count) = split_prefix(path).ok_or_else(|| {
        GgufError::General(format!(
            "{} declares split.count={} but does not use the standard \
             '<prefix>-00001-of-0000N.gguf' shard name",
            path.display(),
            split_count
        ))
    })?;
    if filename_count != split_count {
        return Err(GgufError::General(format!(
            "{} filename declares {} shards but split.count is {}",
            path.display(),
            filename_count,
            split_count
        )));
    }

    let mut candidates = Vec::with_capacity(split_count as usize);
    for index in 1..=split_count {
        let candidate_path = shard_path(&prefix, index, split_count);
        if !candidate_path.is_file() {
            return Err(GgufError::General(format!(
                "missing GGUF shard {} of {}: {}",
                index,
                split_count,
                candidate_path.display()
            )));
        }
        candidates.push(parse_gguf_file(&candidate_path)?);
    }

    assemble_shards(path, split_count, candidates)
}

fn discover_from_dir(path: &Path) -> Result<GgufModelSet> {
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if !entry.file_type()?.is_file() {
            continue;
        }
        if entry_path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            files.push(entry_path);
        }
    }
    files.sort();

    if files.is_empty() {
        return Err(GgufError::General(format!(
            "no .gguf model file found in directory {}",
            path.display()
        )));
    }

    let mut parsed = Vec::with_capacity(files.len());
    for file in &files {
        let model = parse_gguf_file(file)?;
        let count = split_count(&model)?;
        parsed.push((file.clone(), model, count));
    }

    let split_files: Vec<_> = parsed.iter().filter(|(_, _, count)| *count > 1).collect();
    if split_files.is_empty() {
        if parsed.len() != 1 {
            return Err(GgufError::General(format!(
                "directory {} contains {} independent GGUF files; pass one file or a split shard explicitly: {}",
                path.display(),
                parsed.len(),
                files
                    .iter()
                    .map(|file| file.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let (_file, model, _) = parsed.pop().expect("one parsed model");
        return Ok(GgufModelSet {
            path: path.to_path_buf(),
            shards: vec![model],
        });
    }

    if parsed.len() != split_files.len() {
        return Err(GgufError::General(format!(
            "directory {} contains a split GGUF set plus independent GGUF files; pass the first shard explicitly",
            path.display()
        )));
    }

    let mut groups: BTreeMap<(PathBuf, u32), Vec<GgufModel>> = BTreeMap::new();
    for (file, model, count) in parsed {
        let (prefix, filename_count) = split_prefix(&file).ok_or_else(|| {
            GgufError::General(format!(
                "{} declares split.count={} but does not use the standard shard name",
                file.display(),
                count
            ))
        })?;
        if filename_count != count {
            return Err(GgufError::General(format!(
                "{} filename declares {} shards but split.count is {}",
                file.display(),
                filename_count,
                count
            )));
        }
        groups.entry((prefix, count)).or_default().push(model);
    }

    if groups.len() != 1 {
        return Err(GgufError::General(format!(
            "directory {} contains multiple split GGUF model sets; pass the first shard of one set explicitly",
            path.display()
        )));
    }

    let ((_, count), candidates) = groups.into_iter().next().expect("one split group");
    assemble_shards(path, count, candidates)
}

fn assemble_shards(
    source_path: &Path,
    expected_count: u32,
    candidates: Vec<GgufModel>,
) -> Result<GgufModelSet> {
    if candidates.len() != expected_count as usize {
        return Err(GgufError::General(format!(
            "expected {} GGUF shards, found {}",
            expected_count,
            candidates.len()
        )));
    }

    let mut split_tensor_count: Option<u64> = None;
    for model in &candidates {
        if let Some(value) = metadata_u64(model, SPLIT_TENSOR_COUNT_KEY)? {
            match split_tensor_count {
                Some(existing) if existing != value => {
                    return Err(GgufError::General(format!(
                        "inconsistent {} across shards: {} vs {}",
                        SPLIT_TENSOR_COUNT_KEY, existing, value
                    )))
                }
                None => split_tensor_count = Some(value),
                _ => {}
            }
        }
    }

    let mut shards: Vec<Option<GgufModel>> = (0..expected_count).map(|_| None).collect();
    let mut seen_tensors = BTreeSet::new();
    let mut total_tensors = 0u64;
    let mut reference_name: Option<String> = None;
    let mut reference_architecture: Option<String> = None;

    for model in candidates {
        let count = split_count(&model)?;
        if count != expected_count {
            return Err(GgufError::General(format!(
                "{} declares split.count={} but the set requires {}",
                model.path.display(),
                count,
                expected_count
            )));
        }
        let index = metadata_u64(&model, SPLIT_NO_KEY)?.ok_or_else(|| {
            GgufError::General(format!(
                "{} is missing required {} metadata",
                model.path.display(),
                SPLIT_NO_KEY
            ))
        })?;
        if index >= expected_count as u64 {
            return Err(GgufError::General(format!(
                "{} declares split.no={} outside 0..{}",
                model.path.display(),
                index,
                expected_count
            )));
        }
        let slot = &mut shards[index as usize];
        if slot.is_some() {
            return Err(GgufError::General(format!(
                "duplicate GGUF shard index {} in model set",
                index
            )));
        }

        if let Some(name) = model
            .get_metadata("general.name")
            .and_then(|value| value.as_string())
        {
            match &reference_name {
                Some(reference) if reference != name => {
                    return Err(GgufError::General(format!(
                        "inconsistent general.name in shard {}: '{}' vs '{}'",
                        index, name, reference
                    )))
                }
                None => reference_name = Some(name.to_string()),
                _ => {}
            }
        }
        if let Some(architecture) = model
            .get_metadata("general.architecture")
            .and_then(|value| value.as_string())
        {
            match &reference_architecture {
                Some(reference) if reference != architecture => {
                    return Err(GgufError::General(format!(
                        "inconsistent general.architecture in shard {}: '{}' vs '{}'",
                        index, architecture, reference
                    )))
                }
                None => reference_architecture = Some(architecture.to_string()),
                _ => {}
            }
        }

        for tensor in &model.tensors {
            if !seen_tensors.insert(tensor.name.clone()) {
                return Err(GgufError::General(format!(
                    "duplicate tensor '{}' across GGUF shards",
                    tensor.name
                )));
            }
            if let Some(byte_length) = tensor.byte_length {
                let end = tensor.file_offset.checked_add(byte_length).ok_or_else(|| {
                    GgufError::General(format!(
                        "tensor '{}' range overflows in {}",
                        tensor.name,
                        model.path.display()
                    ))
                })?;
                if end > model.file_size {
                    return Err(GgufError::General(format!(
                        "tensor '{}' extends beyond shard {}: {} + {} > {}",
                        tensor.name,
                        model.path.display(),
                        tensor.file_offset,
                        byte_length,
                        model.file_size
                    )));
                }
            }
        }
        total_tensors = total_tensors
            .checked_add(model.tensors.len() as u64)
            .ok_or_else(|| GgufError::General("GGUF tensor count overflow".to_string()))?;
        *slot = Some(model);
    }

    let mut ordered = Vec::with_capacity(expected_count as usize);
    for (index, model) in shards.into_iter().enumerate() {
        ordered.push(model.ok_or_else(|| {
            GgufError::General(format!(
                "missing GGUF shard index {} of {}",
                index, expected_count
            ))
        })?);
    }

    if let Some(expected) = split_tensor_count {
        if total_tensors != expected {
            return Err(GgufError::General(format!(
                "split tensor count mismatch: shards contain {} tensors, metadata declares {}",
                total_tensors, expected
            )));
        }
    }
    if ordered[0]
        .get_metadata("general.architecture")
        .and_then(|value| value.as_string())
        .is_none()
    {
        return Err(GgufError::General(format!(
            "shard zero {} is missing general.architecture metadata",
            ordered[0].path.display()
        )));
    }

    Ok(GgufModelSet {
        path: source_path.to_path_buf(),
        shards: ordered,
    })
}

fn split_count(model: &GgufModel) -> Result<u32> {
    let Some(value) = metadata_u64(model, SPLIT_COUNT_KEY)? else {
        return Ok(1);
    };
    u32::try_from(value)
        .map_err(|_| GgufError::General(format!("split.count {} exceeds u32", value)))
}

fn metadata_u64(model: &GgufModel, key: &str) -> Result<Option<u64>> {
    let Some(value) = model.get_metadata(key) else {
        return Ok(None);
    };
    value.as_u64().map(Some).ok_or_else(|| {
        GgufError::General(format!(
            "metadata '{}' in {} is not an unsigned integer",
            key,
            model.path.display()
        ))
    })
}

/// Parse the standard GGUF split suffix into `(prefix, shard_count)`.
///
/// `foo-00001-of-00002.gguf` -> `(parent/foo, 2)`.
fn split_prefix(path: &Path) -> Option<(PathBuf, u32)> {
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".gguf")?;
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    if parts.len() != 4 || parts[1] != "of" || parts[0].len() != 5 || parts[2].len() != 5 {
        return None;
    }
    let count = parts[0].parse::<u32>().ok()?;
    let index = parts[2].parse::<u32>().ok()?;
    if count == 0 || index == 0 || index > count {
        return None;
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    Some((parent.join(parts[3]), count))
}

fn shard_path(prefix: &Path, index: u32, count: u32) -> PathBuf {
    let file_name = format!(
        "{}-{:05}-of-{:05}.gguf",
        prefix
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model"),
        index,
        count
    );
    prefix
        .parent()
        .map(|parent| parent.join(&file_name))
        .unwrap_or_else(|| PathBuf::from(file_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_string<W: Write>(writer: &mut W, value: &str) {
        let bytes = value.as_bytes();
        writer
            .write_all(&(bytes.len() as u64).to_le_bytes())
            .unwrap();
        writer.write_all(bytes).unwrap();
    }

    fn write_metadata_string<W: Write>(writer: &mut W, key: &str, value: &str) {
        write_string(writer, key);
        writer.write_all(&8u32.to_le_bytes()).unwrap();
        write_string(writer, value);
    }

    fn write_metadata_u32<W: Write>(writer: &mut W, key: &str, value: u32) {
        write_string(writer, key);
        writer.write_all(&4u32.to_le_bytes()).unwrap();
        writer.write_all(&value.to_le_bytes()).unwrap();
    }

    fn write_split_gguf(
        path: &Path,
        index: u32,
        count: u32,
        tensor_count_total: u64,
        tensor_name: &str,
        payload: &[u8],
    ) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let kv_count = if index == 0 { 7u64 } else { 3 };
        bytes.extend_from_slice(&kv_count.to_le_bytes());
        if index == 0 {
            write_metadata_string(&mut bytes, "general.architecture", "llama");
            write_metadata_string(&mut bytes, "general.name", "synthetic-split");
            write_metadata_u32(&mut bytes, "llama.block_count", 1);
            write_metadata_u32(&mut bytes, "llama.attention.head_count", 1);
        }
        write_metadata_u32(&mut bytes, SPLIT_COUNT_KEY, count);
        write_metadata_u32(&mut bytes, SPLIT_NO_KEY, index);
        write_string(&mut bytes, SPLIT_TENSOR_COUNT_KEY);
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&tensor_count_total.to_le_bytes());

        write_string(&mut bytes, tensor_name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64 / 4).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(payload);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn discovers_standard_split_set_from_any_shard() {
        let dir = TempDir::new().unwrap();
        let prefix = dir.path().join("synthetic-00001-of-00002");
        let first = dir.path().join("synthetic-00001-of-00002.gguf");
        let second = dir.path().join("synthetic-00002-of-00002.gguf");
        write_split_gguf(&first, 0, 2, 2, "token_embd.weight", &[1, 2, 3, 4]);
        write_split_gguf(&second, 1, 2, 2, "output.weight", &[5, 6, 7, 8]);
        assert_eq!(prefix.file_stem().unwrap(), "synthetic-00001-of-00002");

        let set = discover_model_set(&second).unwrap();
        assert!(set.is_sharded());
        assert_eq!(set.shards.len(), 2);
        assert!(set
            .primary_path()
            .ends_with("synthetic-00001-of-00002.gguf"));
        assert_eq!(set.total_tensor_bytes(), Some(8));

        let merged = set.merged_model();
        assert_eq!(merged.tensors.len(), 2);
        assert_eq!(merged.data_start_offset, 0);
        let second_tensor = merged
            .tensors
            .iter()
            .find(|tensor| tensor.name == "output.weight")
            .unwrap();
        assert_eq!(second_tensor.file_offset, 4);

        let layouts = set.shard_layouts();
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0].virtual_start, 0);
        assert_eq!(layouts[1].virtual_start, 4);
        assert_eq!(set.virtual_size(), 8);
    }

    #[test]
    fn data_source_reads_across_shard_boundary() {
        let dir = TempDir::new().unwrap();
        let first = dir.path().join("synthetic-00001-of-00002.gguf");
        let second = dir.path().join("synthetic-00002-of-00002.gguf");
        write_split_gguf(&first, 0, 2, 2, "token_embd.weight", &[1, 2, 3, 4]);
        write_split_gguf(&second, 1, 2, 2, "output.weight", &[5, 6, 7, 8]);

        let source = crate::GgufDataSource::open(dir.path()).unwrap();
        assert_eq!(
            source.read_tensor("token_embd.weight").unwrap(),
            [1, 2, 3, 4]
        );
        assert_eq!(source.read_tensor("output.weight").unwrap(), [5, 6, 7, 8]);

        let descriptors: Vec<_> = source.model().tensors.iter().collect();
        let coalesced = source
            .read_coalesced_tensor_range(&descriptors, 0, 8)
            .unwrap();
        assert_eq!(coalesced, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn directory_discovery_rejects_missing_shard() {
        let dir = TempDir::new().unwrap();
        let first = dir.path().join("model-00001-of-00002.gguf");
        write_split_gguf(&first, 0, 2, 2, "token_embd.weight", &[1, 2, 3, 4]);
        let error = discover_model_set(dir.path()).unwrap_err().to_string();
        assert!(error.contains("expected 2 GGUF shards"), "{error}");
    }

    #[test]
    fn directory_discovery_rejects_multiple_independent_models() {
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("a.gguf");
        let b = dir.path().join("b.gguf");
        write_split_gguf(&a, 0, 1, 1, "a.weight", &[1, 2, 3, 4]);
        write_split_gguf(&b, 0, 1, 1, "b.weight", &[5, 6, 7, 8]);
        let error = discover_model_set(dir.path()).unwrap_err().to_string();
        assert!(error.contains("independent GGUF files"), "{error}");
    }

    #[test]
    fn rejects_tensor_count_mismatch() {
        let dir = TempDir::new().unwrap();
        let first = dir.path().join("model-00001-of-00002.gguf");
        let second = dir.path().join("model-00002-of-00002.gguf");
        write_split_gguf(&first, 0, 2, 3, "token_embd.weight", &[1, 2, 3, 4]);
        write_split_gguf(&second, 1, 2, 3, "output.weight", &[5, 6, 7, 8]);
        let error = discover_model_set(&first).unwrap_err().to_string();
        assert!(error.contains("split tensor count mismatch"), "{error}");
    }

    #[test]
    fn duplicate_tensor_names_across_shards_are_rejected() {
        let dir = TempDir::new().unwrap();
        let first = dir.path().join("model-00001-of-00002.gguf");
        let second = dir.path().join("model-00002-of-00002.gguf");
        write_split_gguf(&first, 0, 2, 2, "same.weight", &[1, 2, 3, 4]);
        write_split_gguf(&second, 1, 2, 2, "same.weight", &[5, 6, 7, 8]);
        let error = discover_model_set(&first).unwrap_err().to_string();
        assert!(error.contains("duplicate tensor"), "{error}");
    }

    #[test]
    fn single_file_still_resolves_as_one_shard() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("single.gguf");
        write_split_gguf(&file, 0, 1, 1, "token_embd.weight", &[1, 2, 3, 4]);
        let set = discover_model_set(&file).unwrap();
        assert!(!set.is_sharded());
        assert_eq!(set.shards.len(), 1);
        assert_eq!(set.merged_model().tensors.len(), 1);
        assert_eq!(set.virtual_size(), std::fs::metadata(&file).unwrap().len());
    }
}
