//! File-backed tensor access
//!
//! This module builds directly on tensor descriptors from Milestone 1.
//! It provides explicit access to tensor data from the original GGUF file
//! without loading the entire model into RAM.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::DataSourceError;
use crate::model::{GgufModel, TensorDescriptor};
use crate::gguf::parse_gguf_file;

/// A file-backed data source for GGUF tensor data
///
/// Holds the parsed model (metadata + descriptors) and provides methods to
/// read tensor payloads on demand. The entire model is never loaded into RAM
/// at once; only requested tensors or byte ranges are read.
#[derive(Debug)]
pub struct GgufDataSource {
    model: GgufModel,
    path: PathBuf,
    file_size: u64,
}

impl GgufDataSource {
    /// Open a GGUF file as a data source
    ///
    /// Parses header, metadata, and tensor descriptors (file-backed), but does
    /// not load tensor payloads.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DataSourceError> {
        let path_buf = path.as_ref().to_path_buf();
        let model = parse_gguf_file(&path_buf).map_err(|e| DataSourceError::General(e.to_string()))?;
        let file_size = model.file_size;
        Ok(Self {
            model,
            path: path_buf,
            file_size,
        })
    }

    /// Get reference to parsed model
    pub fn model(&self) -> &GgufModel {
        &self.model
    }

    /// Get tensor descriptor by name
    pub fn get_descriptor(&self, name: &str) -> Result<&TensorDescriptor, DataSourceError> {
        self.model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| DataSourceError::TensorNotFound(name.to_string()))
    }

    /// Read full tensor data by descriptor
    pub fn read_tensor_by_descriptor(
        &self,
        desc: &TensorDescriptor,
    ) -> Result<Vec<u8>, DataSourceError> {
        let byte_length = desc.byte_length.ok_or_else(|| {
            DataSourceError::UnknownByteLength(desc.name.clone(), desc.ggml_type.name())
        })?;

        self.validate_bounds(desc, byte_length)?;

        self.read_range(desc.file_offset, byte_length)
    }

    /// Read full tensor data by name
    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, DataSourceError> {
        let desc = self.get_descriptor(name)?;
        // Clone descriptor to avoid borrow issues
        let desc_clone = desc.clone();
        self.read_tensor_by_descriptor(&desc_clone)
    }

    /// Read a byte range within a tensor
    ///
    /// `offset` and `length` are relative to the start of the tensor data,
    /// not absolute file offset.
    pub fn read_tensor_range(
        &self,
        name: &str,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, DataSourceError> {
        let desc = self.get_descriptor(name)?.clone();
        self.read_tensor_range_by_descriptor(&desc, offset, length)
    }

    /// Read a byte range within a tensor by descriptor
    pub fn read_tensor_range_by_descriptor(
        &self,
        desc: &TensorDescriptor,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, DataSourceError> {
        let byte_length = desc.byte_length.ok_or_else(|| {
            DataSourceError::UnknownByteLength(desc.name.clone(), desc.ggml_type.name())
        })?;

        // Validate range within tensor
        if offset > byte_length {
            return Err(DataSourceError::InvalidRange(format!(
                "offset {} beyond tensor '{}' length {}",
                offset, desc.name, byte_length
            )));
        }
        if offset + length > byte_length {
            return Err(DataSourceError::InvalidRange(format!(
                "range offset {} + length {} exceeds tensor '{}' length {}",
                offset, length, desc.name, byte_length
            )));
        }

        let file_offset = desc.file_offset + offset;
        self.validate_bounds_at(file_offset, length, desc)?;

        self.read_range(file_offset, length)
    }

    fn validate_bounds(
        &self,
        desc: &TensorDescriptor,
        byte_length: u64,
    ) -> Result<(), DataSourceError> {
        self.validate_bounds_at(desc.file_offset, byte_length, desc)
    }

    fn validate_bounds_at(
        &self,
        file_offset: u64,
        length: u64,
        desc: &TensorDescriptor,
    ) -> Result<(), DataSourceError> {
        if file_offset < self.model.data_start_offset {
            return Err(DataSourceError::InvalidOffset(format!(
                "tensor '{}' file_offset {} is before data_start {}",
                desc.name, file_offset, self.model.data_start_offset
            )));
        }
        if file_offset + length > self.file_size {
            return Err(DataSourceError::OutOfBounds {
                name: desc.name.clone(),
                file_offset,
                byte_length: length,
                file_size: self.file_size,
            });
        }
        Ok(())
    }

    fn read_range(&self, file_offset: u64, length: u64) -> Result<Vec<u8>, DataSourceError> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(file_offset))?;
        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.model.tensors.iter().map(|t| t.name.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::align_offset;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_string<W: Write>(w: &mut W, s: &str) {
        let bytes = s.as_bytes();
        w.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        w.write_all(bytes).unwrap();
    }

    fn write_u32<W: Write>(w: &mut W, v: u32) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }

    fn write_u64<W: Write>(w: &mut W, v: u64) {
        w.write_all(&v.to_le_bytes()).unwrap();
    }

    fn create_test_gguf_with_data() -> NamedTempFile {
        let mut buf = Vec::new();
        // magic
        buf.extend_from_slice(b"GGUF");
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 2); // tensor count
        write_u64(&mut buf, 1); // kv count
        write_string(&mut buf, "general.architecture");
        write_u32(&mut buf, 8);
        write_string(&mut buf, "llama");

        // tensor 1: a.weight, 4 elements F32
        write_string(&mut buf, "a.weight");
        write_u32(&mut buf, 1);
        write_u64(&mut buf, 4);
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 0); // offset 0

        // tensor 2: b.weight, 8 elements F32
        write_string(&mut buf, "b.weight");
        write_u32(&mut buf, 1);
        write_u64(&mut buf, 8);
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 16); // offset 16 (after first tensor 4*4=16)

        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        let pad = (aligned - pos) as usize;
        buf.extend(vec![0u8; pad]);

        // tensor data: a = [1.0, 2.0, 3.0, 4.0] f32 LE
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // b = [5.0 .. 12.0]
        for v in [5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn test_valid_tensor_read() {
        let tmp = create_test_gguf_with_data();
        let ds = GgufDataSource::open(tmp.path()).unwrap();
        assert_eq!(ds.model().tensors.len(), 2);
        let data = ds.read_tensor("a.weight").unwrap();
        assert_eq!(data.len(), 16);
        // Check first float is 1.0
        let f = f32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(f, 1.0);
    }

    #[test]
    fn test_valid_range_read() {
        let tmp = create_test_gguf_with_data();
        let ds = GgufDataSource::open(tmp.path()).unwrap();
        // Read second element of a.weight: offset 4, length 4
        let data = ds.read_tensor_range("a.weight", 4, 4).unwrap();
        let f = f32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(f, 2.0);
    }

    #[test]
    fn test_invalid_range_rejection() {
        let tmp = create_test_gguf_with_data();
        let ds = GgufDataSource::open(tmp.path()).unwrap();
        // Offset beyond tensor length
        let err = ds.read_tensor_range("a.weight", 100, 4).unwrap_err();
        match err {
            DataSourceError::InvalidRange(_) => {}
            _ => panic!("expected InvalidRange, got {:?}", err),
        }
        // Length exceeds
        let err = ds.read_tensor_range("a.weight", 12, 8).unwrap_err();
        match err {
            DataSourceError::InvalidRange(_) => {}
            _ => panic!("expected InvalidRange, got {:?}", err),
        }
    }

    #[test]
    fn test_invalid_file_offset_rejection() {
        // Create a GGUF where tensor claims to be beyond file
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 1);
        write_u64(&mut buf, 0);
        write_string(&mut buf, "bad.weight");
        write_u32(&mut buf, 1);
        write_u64(&mut buf, 4);
        write_u32(&mut buf, 0);
        write_u64(&mut buf, 999999); // huge offset
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        // No data
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        let ds = GgufDataSource::open(tmp.path()).unwrap();
        let err = ds.read_tensor("bad.weight").unwrap_err();
        match err {
            DataSourceError::OutOfBounds { .. } => {}
            _ => panic!("expected OutOfBounds, got {:?}", err),
        }
    }

    #[test]
    fn test_large_file_incremental_access() {
        // Simulate large file larger than cache: create file with 1M elements but only read chunks
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 1);
        write_u64(&mut buf, 0);
        write_string(&mut buf, "large.weight");
        write_u32(&mut buf, 1);
        write_u64(&mut buf, 1_000_000);
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 0);
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        // Write 4MB of data (1M * 4)
        buf.extend(vec![0xAB; 4_000_000]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();

        let ds = GgufDataSource::open(tmp.path()).unwrap();
        // File size is ~4MB + header, but we can read incrementally without loading all
        let chunk1 = ds.read_tensor_range("large.weight", 0, 1024).unwrap();
        assert_eq!(chunk1.len(), 1024);
        let chunk2 = ds.read_tensor_range("large.weight", 1024, 1024).unwrap();
        assert_eq!(chunk2.len(), 1024);
        // Ensure we didn't load entire file into memory at once (by design, we only read requested range)
    }
}
