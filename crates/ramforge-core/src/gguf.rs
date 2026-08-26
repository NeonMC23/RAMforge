use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use crate::error::{GgufError, Result};
use crate::model::{align_offset, GgufModel, TensorDescriptor};
use crate::types::{ArrayValue, GgmlType, GgufValueType, MetadataValue};

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_STRING_LEN: u64 = 10 * 1024 * 1024; // 10 MB limit for sanity
const MAX_DIMENSIONS: u32 = 8; // sanity limit
const MAX_KV_COUNT: u64 = 100_000;
const MAX_TENSOR_COUNT: u64 = 10_000_000;

struct Reader<R: Read + Seek> {
    inner: R,
    // For debugging, track position
}

impl<R: Read + Seek> Reader<R> {
    fn new(inner: R) -> Self {
        Self { inner }
    }

    fn position(&mut self) -> Result<u64> {
        Ok(self.inner.stream_position()?)
    }

    fn read_exact_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::Truncated(format!("expected {} bytes, got EOF", len))
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(buf)
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.inner.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::Truncated("unexpected EOF while reading u8".to_string())
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(buf[0])
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.inner.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::Truncated("unexpected EOF while reading u16".to_string())
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::Truncated("unexpected EOF while reading u32".to_string())
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64_le(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::Truncated("unexpected EOF while reading u64".to_string())
            } else {
                GgufError::Io(e)
            }
        })?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16_le(&mut self) -> Result<i16> {
        Ok(self.read_u16_le()? as i16)
    }

    fn read_i32_le(&mut self) -> Result<i32> {
        Ok(self.read_u32_le()? as i32)
    }

    fn read_i64_le(&mut self) -> Result<i64> {
        Ok(self.read_u64_le()? as i64)
    }

    fn read_f32_le(&mut self) -> Result<f32> {
        let bits = self.read_u32_le()?;
        Ok(f32::from_bits(bits))
    }

    fn read_f64_le(&mut self) -> Result<f64> {
        let bits = self.read_u64_le()?;
        Ok(f64::from_bits(bits))
    }

    fn read_bool(&mut self) -> Result<bool> {
        let b = self.read_u8()?;
        Ok(b != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64_le()?;
        if len > MAX_STRING_LEN {
            return Err(GgufError::InvalidStringLength(len));
        }
        let bytes = self.read_exact_bytes(len as usize)?;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidStringUtf8)
    }

    fn read_metadata_value(&mut self, value_type: GgufValueType) -> Result<MetadataValue> {
        let val = match value_type {
            GgufValueType::UInt8 => MetadataValue::UInt8(self.read_u8()?),
            GgufValueType::Int8 => MetadataValue::Int8(self.read_i8()?),
            GgufValueType::UInt16 => MetadataValue::UInt16(self.read_u16_le()?),
            GgufValueType::Int16 => MetadataValue::Int16(self.read_i16_le()?),
            GgufValueType::UInt32 => MetadataValue::UInt32(self.read_u32_le()?),
            GgufValueType::Int32 => MetadataValue::Int32(self.read_i32_le()?),
            GgufValueType::Float32 => MetadataValue::Float32(self.read_f32_le()?),
            GgufValueType::Bool => MetadataValue::Bool(self.read_bool()?),
            GgufValueType::String => MetadataValue::String(self.read_string()?),
            GgufValueType::UInt64 => MetadataValue::UInt64(self.read_u64_le()?),
            GgufValueType::Int64 => MetadataValue::Int64(self.read_i64_le()?),
            GgufValueType::Float64 => MetadataValue::Float64(self.read_f64_le()?),
            GgufValueType::Array => {
                return Err(GgufError::General(
                    "nested array type not expected in scalar value path".to_string(),
                ))
            }
        };
        Ok(val)
    }

    fn read_array_value(&mut self) -> Result<MetadataValue> {
        let array_type_u32 = self.read_u32_le()?;
        let array_type = GgufValueType::from_u32(array_type_u32)
            .ok_or(GgufError::InvalidMetadataType(array_type_u32))?;
        if array_type == GgufValueType::Array {
            return Err(GgufError::General(
                "array of arrays is not supported".to_string(),
            ));
        }
        let array_len = self.read_u64_le()?;
        // Sanity check: prevent huge allocations
        if array_len > 10_000_000 {
            return Err(GgufError::General(format!(
                "array length too large: {}",
                array_len
            )));
        }
        let mut values = Vec::with_capacity(array_len as usize);
        for _ in 0..array_len {
            // For array elements, strings are handled specially
            let v = if array_type == GgufValueType::String {
                MetadataValue::String(self.read_string()?)
            } else {
                self.read_metadata_value(array_type)?
            };
            values.push(v);
        }
        Ok(MetadataValue::Array(ArrayValue {
            element_type: array_type,
            values,
        }))
    }
}

/// Parse a GGUF file without loading tensor payloads
///
/// This function implements the core memory-efficiency guarantee of RAMforge:
/// - Only header (24 bytes), metadata KV pairs, and tensor descriptors are read
/// - Tensor data is NOT copied into RAM; only file offsets and byte lengths are recorded
/// - The resulting `GgufModel` is file-backed and supports future out-of-core access
///   via mmap or streaming reads
///
/// The parser validates magic, version, and structure, and returns clear errors
/// for invalid or truncated files.
pub fn parse_gguf_file<P: AsRef<Path>>(path: P) -> Result<GgufModel> {
    let path_buf = path.as_ref().to_path_buf();
    let file = File::open(&path_buf)?;
    let file_size = file.metadata()?.len();
    let reader = BufReader::new(file);
    let mut r = Reader::new(reader);

    // Header
    let magic_bytes = r.read_exact_bytes(4)?;
    let magic_arr: [u8; 4] = magic_bytes.try_into().unwrap();
    if magic_arr != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(magic_arr));
    }

    let version = r.read_u32_le()?;
    if version != 1 && version != 2 && version != 3 {
        // Allow but warn? For milestone we error on unsupported version
        // but spec says current is 3, older versions also exist. Let's allow 1,2,3 only.
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = r.read_u64_le()?;
    let kv_count = r.read_u64_le()?;

    if tensor_count > MAX_TENSOR_COUNT {
        return Err(GgufError::General(format!(
            "tensor count too large: {}",
            tensor_count
        )));
    }
    if kv_count > MAX_KV_COUNT {
        return Err(GgufError::General(format!(
            "metadata kv count too large: {}",
            kv_count
        )));
    }

    // Metadata
    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.read_string()?;
        let value_type_u32 = r.read_u32_le()?;
        let value_type = GgufValueType::from_u32(value_type_u32)
            .ok_or(GgufError::InvalidMetadataType(value_type_u32))?;

        let value = if value_type == GgufValueType::Array {
            r.read_array_value()?
        } else if value_type == GgufValueType::String {
            MetadataValue::String(r.read_string()?)
        } else {
            r.read_metadata_value(value_type)?
        };

        metadata.insert(key, value);
    }

    // Determine alignment
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_ALIGNMENT);

    // Tensor infos
    let mut tensor_infos_raw = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.read_string()?;
        let n_dims = r.read_u32_le()?;
        if n_dims > MAX_DIMENSIONS {
            return Err(GgufError::InvalidDimensionsCount(n_dims));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.read_u64_le()?);
        }
        let ty = r.read_u32_le()?;
        let offset = r.read_u64_le()?;
        tensor_infos_raw.push((name, dims, ty, offset));
    }

    let after_tensor_info_pos = r.position()?;
    let data_start = align_offset(after_tensor_info_pos, alignment);

    // Build TensorDescriptors
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for (name, dims, ty_u32, offset) in tensor_infos_raw {
        let ggml_type = GgmlType::from_u32(ty_u32);
        let num_elements = if dims.is_empty() {
            1
        } else {
            dims.iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .unwrap_or(u64::MAX)
        };

        let byte_length = if let Some((block_size, type_size)) = ggml_type.type_info() {
            if num_elements == u64::MAX {
                None
            } else {
                // ceil division for block quantized types
                let n_blocks = num_elements.div_ceil(block_size);
                n_blocks.checked_mul(type_size)
            }
        } else {
            // For unquantized but unknown, try to estimate if block size 1?
            // We already handled common unquantized types via type_info
            None
        };

        let file_offset = data_start + offset;

        // Basic validation: file_offset should not exceed file_size (but tensor data may be at end)
        // We don't error, just preserve.

        tensors.push(TensorDescriptor {
            name,
            dimensions: dims,
            ggml_type,
            offset,
            file_offset,
            byte_length,
            num_elements,
        });
    }

    Ok(GgufModel {
        path: path_buf,
        file_size,
        version,
        metadata,
        tensors,
        alignment,
        data_start_offset: data_start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[allow(dead_code)]
    fn make_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        // magic
        buf.extend_from_slice(&GGUF_MAGIC);
        // version 3
        write_u32(&mut buf, 3);
        // tensor_count 1
        write_u64(&mut buf, 1);
        // kv_count 2
        write_u64(&mut buf, 2);
        // kv 1: general.architecture = "llama"
        write_string(&mut buf, "general.architecture");
        write_u32(&mut buf, 8); // string
        write_string(&mut buf, "llama");
        // kv 2: llama.context_length = 2048 uint32
        write_string(&mut buf, "llama.context_length");
        write_u32(&mut buf, 4); // uint32
        write_u32(&mut buf, 2048);
        // tensor info
        write_string(&mut buf, "token_embd.weight");
        write_u32(&mut buf, 2); // n_dims
        write_u64(&mut buf, 4096);
        write_u64(&mut buf, 32000);
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 0); // offset
                                // padding to alignment 32
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        let pad = (aligned - pos) as usize;
        buf.extend(vec![0u8; pad]);
        // tensor data: 4096*32000*4 bytes would be huge, but we don't need full data for test; we just write small dummy
        // For test we use small dims earlier? Let's use small dims for test; but this is minimal file with dummy data.
        // We'll not write full data; just enough to not be truncated for header parsing.
        // Actually we need to adjust dims to small for test.
        // This function is not used for file size validation, so we can leave data empty.
        buf
    }

    fn make_gguf_with_small_tensor() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 1);
        write_u64(&mut buf, 1);
        write_string(&mut buf, "general.architecture");
        write_u32(&mut buf, 8);
        write_string(&mut buf, "llama");
        write_string(&mut buf, "test.weight");
        write_u32(&mut buf, 1);
        write_u64(&mut buf, 4);
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 0);
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        let pad = (aligned - pos) as usize;
        buf.extend(vec![0u8; pad]);
        // 4 * f32 = 16 bytes
        buf.extend(vec![0u8; 16]);
        buf
    }

    #[test]
    fn test_parse_valid_minimal() {
        let data = make_gguf_with_small_tensor();
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&data).unwrap();
        let model = parse_gguf_file(tmp.path()).unwrap();
        assert_eq!(model.version, 3);
        assert_eq!(model.tensors.len(), 1);
        assert_eq!(model.tensors[0].name, "test.weight");
        assert_eq!(model.tensors[0].dimensions, vec![4]);
        assert_eq!(model.tensors[0].ggml_type, GgmlType::F32);
        assert_eq!(model.tensors[0].byte_length, Some(16));
        assert_eq!(
            model
                .metadata
                .get("general.architecture")
                .unwrap()
                .as_string(),
            Some("llama")
        );
    }

    #[test]
    fn test_invalid_magic() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BAD!");
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 0);
        write_u64(&mut buf, 0);
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        let err = parse_gguf_file(tmp.path()).unwrap_err();
        match err {
            GgufError::InvalidMagic(_) => {}
            _ => panic!("expected InvalidMagic, got {:?}", err),
        }
    }

    #[test]
    fn test_truncated_file() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        write_u32(&mut buf, 3);
        // truncated after version
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        let err = parse_gguf_file(tmp.path()).unwrap_err();
        match err {
            GgufError::Truncated(_) => {}
            _ => panic!("expected Truncated, got {:?}", err),
        }
    }

    #[test]
    fn test_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        write_u32(&mut buf, 99);
        write_u64(&mut buf, 0);
        write_u64(&mut buf, 0);
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        let err = parse_gguf_file(tmp.path()).unwrap_err();
        match err {
            GgufError::UnsupportedVersion(99) => {}
            _ => panic!("expected UnsupportedVersion, got {:?}", err),
        }
    }

    #[test]
    fn test_metadata_types() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 0);
        write_u64(&mut buf, 5);
        // uint8
        write_string(&mut buf, "test.uint8");
        write_u32(&mut buf, 0);
        buf.push(42);
        // bool
        write_string(&mut buf, "test.bool");
        write_u32(&mut buf, 7);
        buf.push(1);
        // float32
        write_string(&mut buf, "test.f32");
        write_u32(&mut buf, 6);
        write_u32(&mut buf, f32::to_bits(2.5));
        // string
        write_string(&mut buf, "test.string");
        write_u32(&mut buf, 8);
        write_string(&mut buf, "hello");
        // array of int32
        write_string(&mut buf, "test.array");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 5); // int32
        write_u64(&mut buf, 3);
        write_u32(&mut buf, 1);
        write_u32(&mut buf, 2);
        write_u32(&mut buf, 3);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        let model = parse_gguf_file(tmp.path()).unwrap();
        assert_eq!(model.metadata.len(), 5);
        assert_eq!(model.metadata.get("test.uint8").unwrap().as_u64(), Some(42));
        assert_eq!(
            model.metadata.get("test.bool").unwrap().as_bool(),
            Some(true)
        );
        assert_eq!(
            model.metadata.get("test.string").unwrap().as_string(),
            Some("hello")
        );
        let arr = model
            .metadata
            .get("test.array")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(arr.values.len(), 3);
    }

    #[test]
    fn test_quantized_byte_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 1);
        write_u64(&mut buf, 0);
        write_string(&mut buf, "weight.q4_0");
        write_u32(&mut buf, 1);
        write_u64(&mut buf, 32);
        write_u32(&mut buf, 2); // Q4_0
        write_u64(&mut buf, 0);
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        let pad = (aligned - pos) as usize;
        buf.extend(vec![0u8; pad]);
        buf.extend(vec![0u8; 18]); // one block
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        let model = parse_gguf_file(tmp.path()).unwrap();
        assert_eq!(model.tensors[0].byte_length, Some(18));
    }

    #[test]
    fn test_alignment_override() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 0);
        write_u64(&mut buf, 1);
        write_string(&mut buf, "general.alignment");
        write_u32(&mut buf, 4); // uint32
        write_u32(&mut buf, 64);
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        let model = parse_gguf_file(tmp.path()).unwrap();
        assert_eq!(model.alignment, 64);
    }
}
