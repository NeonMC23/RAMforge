//! File-backed tensor access
//!
//! This module builds directly on tensor descriptors from Milestone 1.
//! It provides explicit access to tensor data from the original GGUF file
//! without loading the entire model into RAM.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::DataSourceError;
use crate::gguf::parse_gguf_file;
use crate::model::{GgufModel, TensorDescriptor};
use crate::types::GgmlType;

/// Optional datasource read-path counters. `elapsed` includes destination
/// allocation plus file open/seek/read, not just kernel read syscall time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoProfile {
    pub bytes_read: u64,
    pub read_operations: u64,
    pub read_failures: u64,
    pub elapsed: Duration,
}

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
    profiling_enabled: AtomicBool,
    profile_bytes_read: AtomicU64,
    profile_read_operations: AtomicU64,
    profile_read_failures: AtomicU64,
    profile_read_nanos: AtomicU64,
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
            profiling_enabled: AtomicBool::new(false),
            profile_bytes_read: AtomicU64::new(0),
            profile_read_operations: AtomicU64::new(0),
            profile_read_failures: AtomicU64::new(0),
            profile_read_nanos: AtomicU64::new(0),
        })
    }

    /// Get reference to parsed model
    pub fn model(&self) -> &GgufModel {
        &self.model
    }

    pub fn set_profiling(&self, enabled: bool) {
        self.profiling_enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn reset_io_profile(&self) {
        self.profile_bytes_read.store(0, Ordering::Relaxed);
        self.profile_read_operations.store(0, Ordering::Relaxed);
        self.profile_read_failures.store(0, Ordering::Relaxed);
        self.profile_read_nanos.store(0, Ordering::Relaxed);
    }

    pub fn io_profile(&self) -> IoProfile {
        IoProfile {
            bytes_read: self.profile_bytes_read.load(Ordering::Relaxed),
            read_operations: self.profile_read_operations.load(Ordering::Relaxed),
            read_failures: self.profile_read_failures.load(Ordering::Relaxed),
            elapsed: Duration::from_nanos(self.profile_read_nanos.load(Ordering::Relaxed)),
        }
    }

    fn profile_start(&self) -> Option<Instant> {
        self.profiling_enabled
            .load(Ordering::Relaxed)
            .then(Instant::now)
    }

    fn record_read(&self, started: Option<Instant>, bytes: u64, failed: bool) {
        let Some(started) = started else {
            return;
        };
        self.profile_bytes_read.fetch_add(bytes, Ordering::Relaxed);
        self.profile_read_operations.fetch_add(1, Ordering::Relaxed);
        if failed {
            self.profile_read_failures.fetch_add(1, Ordering::Relaxed);
        }
        let nanos = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.profile_read_nanos.fetch_add(nanos, Ordering::Relaxed);
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

    /// Read an F32 tensor directly into its final `Vec<f32>` representation.
    ///
    /// The destination allocation is not zero-filled and no intermediate raw
    /// byte vector is created. GGUF little-endian bytes are converted in place
    /// only on big-endian hosts; little-endian hosts perform no element-wise
    /// decode pass.
    pub fn read_f32_tensor_by_descriptor(
        &self,
        desc: &TensorDescriptor,
    ) -> Result<Vec<f32>, DataSourceError> {
        self.read_f32_tensor_range_by_descriptor(desc, 0, desc.num_elements)
    }

    /// Read a contiguous element range of an F32 tensor directly into final
    /// F32 storage. `element_offset` and `element_count` are measured in f32
    /// elements, not bytes.
    pub fn read_f32_tensor_range_by_descriptor(
        &self,
        desc: &TensorDescriptor,
        element_offset: u64,
        element_count: u64,
    ) -> Result<Vec<f32>, DataSourceError> {
        if desc.ggml_type != GgmlType::F32 {
            return Err(DataSourceError::General(format!(
                "direct F32 read requires F32 tensor '{}', got {}",
                desc.name,
                desc.ggml_type.name()
            )));
        }

        let byte_length = desc.byte_length.ok_or_else(|| {
            DataSourceError::UnknownByteLength(desc.name.clone(), desc.ggml_type.name())
        })?;
        let expected_byte_length = desc
            .num_elements
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                DataSourceError::InvalidRange(format!(
                    "F32 tensor '{}' byte length overflows",
                    desc.name
                ))
            })?;
        if byte_length != expected_byte_length {
            return Err(DataSourceError::InvalidRange(format!(
                "F32 tensor '{}' descriptor length {} does not match {} elements ({} bytes)",
                desc.name, byte_length, desc.num_elements, expected_byte_length
            )));
        }

        let element_end = element_offset.checked_add(element_count).ok_or_else(|| {
            DataSourceError::InvalidRange(format!(
                "F32 element range overflow for tensor '{}': {} + {}",
                desc.name, element_offset, element_count
            ))
        })?;
        if element_end > desc.num_elements {
            return Err(DataSourceError::InvalidRange(format!(
                "F32 element range {}..{} exceeds tensor '{}' element count {}",
                element_offset, element_end, desc.name, desc.num_elements
            )));
        }

        let relative_byte_offset = element_offset
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                DataSourceError::InvalidRange(format!(
                    "F32 byte offset overflow for tensor '{}'",
                    desc.name
                ))
            })?;
        let range_byte_length = element_count
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| {
                DataSourceError::InvalidRange(format!(
                    "F32 byte length overflow for tensor '{}'",
                    desc.name
                ))
            })?;
        let file_offset = desc
            .file_offset
            .checked_add(relative_byte_offset)
            .ok_or_else(|| {
                DataSourceError::InvalidRange(format!(
                    "F32 file offset overflow for tensor '{}'",
                    desc.name
                ))
            })?;
        self.validate_bounds_at(file_offset, range_byte_length, desc)?;
        self.read_f32_range(file_offset, element_count)
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
        let range_end = offset.checked_add(length).ok_or_else(|| {
            DataSourceError::InvalidRange(format!(
                "range offset {} + length {} overflows for tensor '{}'",
                offset, length, desc.name
            ))
        })?;
        if range_end > byte_length {
            return Err(DataSourceError::InvalidRange(format!(
                "range offset {} + length {} exceeds tensor '{}' length {}",
                offset, length, desc.name, byte_length
            )));
        }

        let file_offset = desc.file_offset.checked_add(offset).ok_or_else(|| {
            DataSourceError::InvalidRange(format!(
                "file offset overflow for tensor '{}': {} + {}",
                desc.name, desc.file_offset, offset
            ))
        })?;
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
        let end = file_offset.checked_add(length).ok_or_else(|| {
            DataSourceError::InvalidRange(format!(
                "file range overflow for tensor '{}': {} + {}",
                desc.name, file_offset, length
            ))
        })?;
        if end > self.file_size {
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
        let started = self.profile_start();
        let result: Result<Vec<u8>, DataSourceError> = (|| {
            let length_usize = usize::try_from(length).map_err(|_| {
                DataSourceError::InvalidRange(format!(
                    "requested byte range length {} does not fit this platform",
                    length
                ))
            })?;
            let mut file = File::open(&self.path)?;
            file.seek(SeekFrom::Start(file_offset))?;

            // `read_to_end` appends into spare capacity rather than requiring a
            // zero-filled destination that `read_exact` immediately overwrites.
            // `Take` prevents reading beyond the validated range; the explicit
            // length check preserves exact/short-read behavior.
            let mut buf = Vec::new();
            buf.try_reserve_exact(length_usize).map_err(|e| {
                DataSourceError::General(format!(
                    "failed to reserve {}-byte read buffer: {}",
                    length_usize, e
                ))
            })?;
            let mut limited = file.take(length);
            limited.read_to_end(&mut buf)?;
            if buf.len() != length_usize {
                return Err(DataSourceError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                )));
            }
            Ok(buf)
        })();
        let bytes = result.as_ref().map(|buffer| buffer.len() as u64).unwrap_or(0);
        self.record_read(started, bytes, result.is_err());
        result
    }

    /// Read exact little-endian F32 data into uninitialized final storage.
    fn read_f32_range(
        &self,
        file_offset: u64,
        element_count: u64,
    ) -> Result<Vec<f32>, DataSourceError> {
        let started = self.profile_start();
        let result = self.read_f32_range_unprofiled(file_offset, element_count);
        let bytes = result
            .as_ref()
            .map(|values| std::mem::size_of_val(values.as_slice()) as u64)
            .unwrap_or(0);
        self.record_read(started, bytes, result.is_err());
        result
    }

    fn read_f32_range_unprofiled(
        &self,
        file_offset: u64,
        element_count: u64,
    ) -> Result<Vec<f32>, DataSourceError> {
        let element_count = usize::try_from(element_count).map_err(|_| {
            DataSourceError::InvalidRange(format!(
                "F32 element count {} does not fit this platform",
                element_count
            ))
        })?;
        let byte_length = element_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                DataSourceError::InvalidRange(format!(
                    "F32 read size overflows for {} elements",
                    element_count
                ))
            })?;
        if element_count == 0 {
            return Ok(Vec::new());
        }

        let mut values = Vec::<f32>::new();
        values.try_reserve_exact(element_count).map_err(|e| {
            DataSourceError::General(format!(
                "failed to reserve {}-element F32 read buffer: {}",
                element_count, e
            ))
        })?;

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(file_offset))?;
        {
            let spare = &mut values.spare_capacity_mut()[..element_count];
            // SAFETY:
            // - `spare` refers to `element_count` properly aligned f32 slots;
            // - their total size is exactly `byte_length` bytes;
            // - the vector length remains zero while `read_exact` runs, so an
            //   error drops no partially initialized f32 values;
            // - every possible 32-bit pattern is a valid Rust f32 value;
            // - on success every byte is initialized before `set_len` below.
            let destination = unsafe {
                std::slice::from_raw_parts_mut(
                    spare.as_mut_ptr().cast::<u8>(),
                    byte_length,
                )
            };
            file.read_exact(destination)?;
        }

        // SAFETY: `read_exact` above initialized every byte of every reserved
        // f32 slot, and all f32 bit patterns are valid.
        unsafe {
            values.set_len(element_count);
        }

        // GGUF is little-endian. The direct byte load already has native F32
        // layout on little-endian hosts; big-endian hosts normalize in place.
        #[cfg(target_endian = "big")]
        for value in &mut values {
            *value = f32::from_bits(u32::from_le(value.to_bits()));
        }

        Ok(values)
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

    fn create_single_f32_gguf(dims: &[u64], bits: &[u32]) -> NamedTempFile {
        assert_eq!(dims.iter().product::<u64>(), bits.len() as u64);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 1); // tensor count
        write_u64(&mut buf, 0); // metadata count
        write_string(&mut buf, "test.weight");
        write_u32(&mut buf, dims.len() as u32);
        for &dim in dims {
            write_u64(&mut buf, dim);
        }
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 0);
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        for &value_bits in bits {
            buf.extend_from_slice(&value_bits.to_le_bytes());
        }

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
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
    fn test_io_profile_counts_reads_and_bytes_when_enabled() {
        let tmp = create_test_gguf_with_data();
        let ds = GgufDataSource::open(tmp.path()).unwrap();
        ds.set_profiling(true);
        ds.reset_io_profile();
        let data = ds.read_tensor("a.weight").unwrap();
        assert_eq!(data.len(), 16);
        let profile = ds.io_profile();
        assert_eq!(profile.bytes_read, 16);
        assert_eq!(profile.read_operations, 1);
        assert_eq!(profile.read_failures, 0);
        ds.reset_io_profile();
        assert_eq!(ds.io_profile(), IoProfile::default());
    }

    #[test]
    fn test_direct_f32_read_matches_logical_decode_and_layout() {
        // GGML shape [in=3, out=2], file rows [1,2,3] and [4,5,6].
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bits: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        let tmp = create_single_f32_gguf(&[3, 2], &bits);
        let ds = GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("test.weight").unwrap();

        let direct = ds.read_f32_tensor_by_descriptor(desc).unwrap();
        let raw = ds.read_tensor_by_descriptor(desc).unwrap();
        let reference: Vec<f32> = raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(
            direct.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            reference
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        let tensor = crate::tensor::TensorData::from_f32_vec(
            desc.dimensions.clone(),
            desc.num_elements,
            direct,
        )
        .unwrap();
        assert_eq!(tensor.shape(), &[3, 2]);
        let mut output = [0.0f32; 2];
        tensor.matvec(&[1.0, 10.0, 100.0], &mut output).unwrap();
        assert_eq!(output, [321.0, 654.0]);
    }

    #[test]
    fn test_direct_f32_read_preserves_edge_bit_patterns() {
        let bits = [
            0x0000_0000, // +0
            0x8000_0000, // -0
            0x3fc0_0000, // +1.5
            0xc010_0000, // -2.25
            0x0000_0001, // smallest positive subnormal
            0x8000_0001, // smallest negative subnormal
            0x7f80_0000, // +Inf
            0xff80_0000, // -Inf
            0x7fc1_2345, // quiet NaN payload
            0x7fa1_2345, // signaling NaN payload
        ];
        let tmp = create_single_f32_gguf(&[bits.len() as u64], &bits);
        let ds = GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("test.weight").unwrap();
        let direct = ds.read_f32_tensor_by_descriptor(desc).unwrap();
        let actual: Vec<u32> = direct.iter().map(|value| value.to_bits()).collect();
        assert_eq!(actual.as_slice(), &bits);
    }

    #[test]
    fn test_direct_f32_short_read_is_error() {
        let bits = [1.0f32.to_bits(), 2.0f32.to_bits(), 3.0f32.to_bits()];
        let tmp = create_single_f32_gguf(&[3], &bits);
        let ds = GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("test.weight").unwrap();
        tmp.as_file()
            .set_len(desc.file_offset + desc.byte_length.unwrap() - 1)
            .unwrap();

        let error = ds.read_f32_tensor_by_descriptor(desc).unwrap_err();
        assert!(
            matches!(&error, DataSourceError::Io(io) if io.kind() == std::io::ErrorKind::UnexpectedEof),
            "unexpected error: {:?}",
            error
        );
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
        // Overflow must be rejected rather than wrapping into an in-bounds read.
        let err = ds
            .read_tensor_range("a.weight", 1, u64::MAX)
            .unwrap_err();
        assert!(matches!(err, DataSourceError::InvalidRange(_)));
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
