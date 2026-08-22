//! Layer-oriented model representation for out-of-core streaming
//!
//! Groups required tensors by transformer layer to enable loading one layer at a time.

use ramforge_core::model::{GgufModel, TensorDescriptor};

#[derive(Debug, Clone)]
pub struct LayerDescriptor {
    pub layer_idx: usize,
    pub tensors: Vec<TensorDescriptor>,
    /// Total byte length of all tensors in this layer when determinable
    pub total_bytes: Option<u64>,
}

impl LayerDescriptor {
    pub fn new(layer_idx: usize, tensors: Vec<TensorDescriptor>) -> Self {
        let total_bytes = {
            let mut sum = 0u64;
            let mut all_known = true;
            for t in &tensors {
                if let Some(b) = t.byte_length {
                    if let Some(new_sum) = sum.checked_add(b) {
                        sum = new_sum;
                    } else {
                        all_known = false;
                        break;
                    }
                } else {
                    all_known = false;
                    break;
                }
            }
            if all_known { Some(sum) } else { None }
        };

        Self {
            layer_idx,
            tensors,
            total_bytes,
        }
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| t.name.clone()).collect()
    }
}

/// Group tensors by layer for LLaMA/Qwen2 architecture
///
/// Expected naming: `blk.{i}.attn_q.weight` etc.
/// Returns Vec<LayerDescriptor> sorted by layer_idx
pub fn group_layers(model: &GgufModel, block_count: usize) -> Vec<LayerDescriptor> {
    let mut layers: Vec<LayerDescriptor> = Vec::with_capacity(block_count);

    for i in 0..block_count {
        let prefix = format!("blk.{}.", i);
        let mut tensors = Vec::new();
        for t in &model.tensors {
            if t.name.starts_with(&prefix) {
                tensors.push(t.clone());
            }
        }
        layers.push(LayerDescriptor::new(i, tensors));
    }

    layers
}

/// Persistent (non-layer) tensors
#[derive(Debug, Clone)]
pub struct PersistentDescriptors {
    pub token_embd: Option<TensorDescriptor>,
    pub output_norm: Option<TensorDescriptor>,
    pub output: Option<TensorDescriptor>,
}

impl PersistentDescriptors {
    pub fn from_model(model: &GgufModel) -> Self {
        let token_embd = model.tensors.iter().find(|t| t.name == "token_embd.weight").cloned();
        let output_norm = model.tensors.iter().find(|t| t.name == "output_norm.weight").cloned();
        let output = model.tensors.iter().find(|t| t.name == "output.weight").cloned();
        Self {
            token_embd,
            output_norm,
            output,
        }
    }

    pub fn total_bytes(&self) -> Option<u64> {
        let mut sum = 0u64;
        let mut all_known = true;
        for t in [&self.token_embd, &self.output_norm, &self.output]
            .iter()
            .filter_map(|opt| opt.as_ref())
        {
            if let Some(b) = t.byte_length {
                sum = sum.saturating_add(b);
                if sum == u64::MAX {
                    all_known = false;
                    break;
                }
            } else {
                all_known = false;
                break;
            }
        }
        if all_known { Some(sum) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ramforge_core::model::{align_offset, GgufModel};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_string<W: Write>(w: &mut W, s: &str) {
        w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }
    fn write_u32<W: Write>(w: &mut W, v: u32) { w.write_all(&v.to_le_bytes()).unwrap(); }
    fn write_u64<W: Write>(w: &mut W, v: u64) { w.write_all(&v.to_le_bytes()).unwrap(); }

    fn make_model_with_layers(n_layers: usize) -> GgufModel {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        let tensor_count = 2 + n_layers * 9;
        buf.extend_from_slice(&(tensor_count as u64).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv count

        let mut offset = 0u64;
        for name in ["token_embd.weight", "output_norm.weight"] {
            write_string(&mut buf, name);
            write_u32(&mut buf, 1);
            write_u64(&mut buf, 8);
            write_u32(&mut buf, 0);
            write_u64(&mut buf, offset);
            offset += 32;
        }
        for i in 0..n_layers {
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                let name = format!("blk.{}.{}", i, suffix);
                write_string(&mut buf, &name);
                write_u32(&mut buf, 1);
                write_u64(&mut buf, 8);
                write_u32(&mut buf, 0);
                write_u64(&mut buf, offset);
                offset += 32;
            }
        }

        let pos = buf.len() as u64;
        let aligned = align_offset(pos, 32);
        buf.extend(vec![0u8; (aligned - pos) as usize]);
        buf.extend(vec![0u8; offset as usize]);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        ramforge_core::gguf::parse_gguf_file(tmp.path()).unwrap()
    }

    #[test]
    fn test_layer_grouping() {
        let model = make_model_with_layers(3);
        let layers = group_layers(&model, 3);
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].tensors.len(), 9);
        assert_eq!(layers[1].tensors.len(), 9);
        assert_eq!(layers[0].layer_idx, 0);
    }

    #[test]
    fn test_persistent_descriptors() {
        let model = make_model_with_layers(2);
        let pers = PersistentDescriptors::from_model(&model);
        assert!(pers.token_embd.is_some());
        assert!(pers.output_norm.is_some());
    }
}
