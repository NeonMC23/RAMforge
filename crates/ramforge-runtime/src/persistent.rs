//! Persistent weight streaming – token_embd, output_norm, output
//!
//! Policy: small persistent tensor (<25% budget) → keep resident, large → stream on demand

#![allow(clippy::needless_range_loop)]

use ramforge_core::{
    cache::BoundedCache,
    datasource::GgufDataSource,
    memory::MemoryBudget,
    model::TensorDescriptor,
    tensor::TensorData,
};

use crate::residency::ResidencyStats;

#[derive(Debug, Clone)]
pub enum PersistentWeight {
    Resident(TensorData),
    Streamed(TensorDescriptor),
}

impl PersistentWeight {
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Resident(td) => td.resident_bytes(),
            Self::Streamed(_) => 0,
        }
    }

    pub fn is_resident(&self) -> bool {
        matches!(self, Self::Resident(_))
    }

    pub fn is_streamed(&self) -> bool {
        matches!(self, Self::Streamed(_))
    }

    pub fn get_embedding(
        &self,
        token_id: usize,
        n_embd: usize,
        data_source: &GgufDataSource,
        _cache: &mut BoundedCache,
        budget: &mut MemoryBudget,
        _stats: &mut ResidencyStats,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Resident(td) => td
                .get_embedding(token_id, n_embd)
                .map_err(|e| format!("embedding lookup failed: {}", e)),
            Self::Streamed(desc) => {
                let row_bytes = Self::row_bytes_for_token(desc, n_embd)?;
                let raw = data_source
                    .read_tensor_range(&desc.name, (token_id as u64) * (row_bytes as u64), row_bytes as u64)
                    .map_err(|e| format!("failed to read embedding row: {}", e))?;

                let tmp_bytes = n_embd * 4;
                let alloc_name = format!("tmp:embd_row:{}", token_id);
                if budget.get(&alloc_name).is_none() {
                    budget
                        .allocate(alloc_name.clone(), tmp_bytes as u64)
                        .map_err(|e| format!("budget too small for embedding row: {}", e))?;
                }

                let td = TensorData::from_bytes(
                    desc.ggml_type,
                    vec![n_embd as u64],
                    n_embd as u64,
                    raw,
                )
                .map_err(|e| format!("failed to create TensorData for embedding row: {}", e))?;

                let result = td.to_f32_vec().map_err(|e| format!("failed to dequantize embedding row: {}", e))?;
                let _ = budget.release(&alloc_name);
                Ok(result)
            }
        }
    }

    fn row_bytes_for_token(desc: &TensorDescriptor, n_embd: usize) -> Result<usize, String> {
        let elems = n_embd;
        let bytes = match desc.ggml_type {
            ramforge_core::types::GgmlType::F32 => elems * 4,
            ramforge_core::types::GgmlType::F16 | ramforge_core::types::GgmlType::BF16 => elems * 2,
            ramforge_core::types::GgmlType::Q4_0 => (elems / 32) * 18,
            ramforge_core::types::GgmlType::Q8_0 => (elems / 32) * 34,
            ramforge_core::types::GgmlType::Q4_K => (elems / 256) * 144,
            ramforge_core::types::GgmlType::Q5_K => (elems / 256) * 176,
            ramforge_core::types::GgmlType::Q6_K => (elems / 256) * 210,
            ramforge_core::types::GgmlType::Q2_K => (elems / 256) * 84,
            ramforge_core::types::GgmlType::Q3_K => (elems / 256) * 110,
            ramforge_core::types::GgmlType::Q8_K => (elems / 256) * 292,
            _ => {
                return Err(format!(
                    "unsupported persistent weight type for streaming: {}",
                    desc.ggml_type.name()
                ))
            }
        };
        Ok(bytes)
    }

    pub fn compute_logits(
        &self,
        hidden: &[f32],
        data_source: &GgufDataSource,
        _cache: &mut BoundedCache,
        _budget: &mut MemoryBudget,
        _stats: &mut ResidencyStats,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Resident(td) => {
                let n_embd = hidden.len();
                let vocab = td.num_elements() / n_embd;
                let mut logits = vec![0.0f32; vocab];
                td.matvec(hidden, &mut logits)
                    .map_err(|e| format!("output matvec failed: {}", e))?;
                Ok(logits)
            }
            Self::Streamed(desc) => {
                let n_embd = hidden.len();
                let vocab = if desc.dimensions.len() == 2 {
                    let d0 = desc.dimensions[0] as usize;
                    let d1 = desc.dimensions[1] as usize;
                    if d0 == n_embd { d1 } else { d0 }
                } else {
                    return Err("output weight should be 2D".to_string());
                };
                let row_bytes = Self::row_bytes_for_token(desc, n_embd)?;
                let mut logits = vec![0.0f32; vocab];
                for token_id in 0..vocab {
                    let raw = data_source
                        .read_tensor_range(&desc.name, (token_id as u64) * (row_bytes as u64), row_bytes as u64)
                        .map_err(|e| format!("failed to read output row {}: {}", token_id, e))?;
                    let td = TensorData::from_bytes(
                        desc.ggml_type,
                        vec![n_embd as u64],
                        n_embd as u64,
                        raw,
                    )
                    .map_err(|e| format!("failed to create TensorData for output row: {}", e))?;
                    let row_f32 = td.to_f32_vec().map_err(|e| e.to_string())?;
                    let mut sum = 0.0f32;
                    for (a, b) in row_f32.iter().zip(hidden.iter()) {
                        sum += a * b;
                    }
                    logits[token_id] = sum;
                }
                Ok(logits)
            }
        }
    }

    pub fn to_f32_vec(&self, data_source: &GgufDataSource) -> Result<Vec<f32>, String> {
        match self {
            Self::Resident(td) => td.to_f32_vec().map_err(|e| e.to_string()),
            Self::Streamed(desc) => {
                let raw = data_source
                    .read_tensor(&desc.name)
                    .map_err(|e| format!("failed to read persistent tensor '{}': {}", desc.name, e))?;
                let td = TensorData::from_bytes(
                    desc.ggml_type,
                    desc.dimensions.clone(),
                    desc.num_elements,
                    raw,
                )
                .map_err(|e| format!("failed to create TensorData: {}", e))?;
                td.to_f32_vec().map_err(|e| e.to_string())
            }
        }
    }

    pub fn matvec(
        &self,
        x: &[f32],
        y: &mut [f32],
        data_source: &GgufDataSource,
    ) -> Result<(), String> {
        match self {
            Self::Resident(td) => td.matvec(x, y).map_err(|e| e.to_string()),
            Self::Streamed(_) => {
                let f32_vec = self.to_f32_vec(data_source)?;
                if f32_vec.len() == x.len() * y.len() {
                    for i in 0..y.len() {
                        let mut sum = 0.0;
                        for j in 0..x.len() {
                            sum += f32_vec[i * x.len() + j] * x[j];
                        }
                        y[i] = sum;
                    }
                    Ok(())
                } else {
                    Err("streamed matvec shape mismatch".to_string())
                }
            }
        }
    }
}

pub fn should_keep_resident(tensor_bytes: u64, budget_total: u64) -> bool {
    tensor_bytes * 4 <= budget_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use ramforge_core::model::align_offset;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_string<W: Write>(w: &mut W, s: &str) {
        w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }
    fn write_u32<W: Write>(w: &mut W, v: u32) { w.write_all(&v.to_le_bytes()).unwrap(); }
    fn write_u64<W: Write>(w: &mut W, v: u64) { w.write_all(&v.to_le_bytes()).unwrap(); }

    #[test]
    fn test_persistent_policy() {
        assert!(should_keep_resident(100, 1000));
        assert!(!should_keep_resident(300, 1000));
    }

    #[test]
    fn test_persistent_streaming_embedding() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        write_string(&mut buf, "token_embd.weight");
        write_u32(&mut buf, 2);
        write_u64(&mut buf, 8);
        write_u64(&mut buf, 4);
        write_u32(&mut buf, 0);
        write_u64(&mut buf, 0);
        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        for token_id in 0..4 {
            for _ in 0..8 {
                buf.extend_from_slice(&(token_id as f32).to_le_bytes());
            }
        }
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();

        let ds = ramforge_core::datasource::GgufDataSource::open(tmp.path()).unwrap();
        let desc = ds.get_descriptor("token_embd.weight").unwrap().clone();
        let streamed = PersistentWeight::Streamed(desc);

        let mut budget = ramforge_core::memory::MemoryBudget::new(1024).unwrap();
        let mut cache = ramforge_core::cache::BoundedCache::new(512).unwrap();
        let mut stats = crate::residency::ResidencyStats::new(0);

        let embd = streamed.get_embedding(2, 8, &ds, &mut cache, &mut budget, &mut stats).unwrap();
        assert_eq!(embd.len(), 8);
        assert!((embd[0] - 2.0).abs() < 1e-5);
    }
}
