//! Tokenizer loading from GGUF metadata
//!
//! Supports LLaMA SentencePiece-style tokenizer with naive longest-match
//! encoding and simple detokenization. This is sufficient for Milestone 3
//! to demonstrate real inference with a small model.
//!
//! Documented supported tokenizer models:
//! - "llama" (SentencePiece, with scores)
//! - "gpt2" (BPE with merges, fallback to naive)
//!
//! For real LLaMA models, the tokenizer uses the vocab from
//! `tokenizer.ggml.tokens` and scores from `tokenizer.ggml.scores`.

use std::collections::HashMap;

use crate::model::GgufModel;
use crate::types::MetadataValue;

/// Token type as stored in GGUF
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub model: String,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub token_types: Vec<TokenType>,
    pub merges: Option<Vec<String>>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
    // Fast lookup
    token_to_id: HashMap<String, u32>,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufModel) -> Result<Self, String> {
        let model = gguf
            .get_metadata("tokenizer.ggml.model")
            .and_then(|v| v.as_string())
            .unwrap_or("llama")
            .to_string();

        // tokens
        let tokens_val = gguf
            .get_metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| "missing tokenizer.ggml.tokens".to_string())?;
        let tokens_arr = tokens_val
            .as_array()
            .ok_or_else(|| "tokenizer.ggml.tokens is not an array".to_string())?;

        let mut tokens = Vec::with_capacity(tokens_arr.values.len());
        for v in &tokens_arr.values {
            if let Some(s) = v.as_string() {
                tokens.push(s.to_string());
            } else {
                return Err(format!("token value not a string: {:?}", v));
            }
        }

        // scores (optional)
        let scores = if let Some(val) = gguf.get_metadata("tokenizer.ggml.scores") {
            if let Some(arr) = val.as_array() {
                let mut out = Vec::with_capacity(arr.values.len());
                for v in &arr.values {
                    match v {
                        MetadataValue::Float32(f) => out.push(*f),
                        MetadataValue::Float64(f) => out.push(*f as f32),
                        MetadataValue::Int32(i) => out.push(*i as f32),
                        MetadataValue::UInt32(u) => out.push(*u as f32),
                        _ => out.push(0.0),
                    }
                }
                out
            } else {
                vec![0.0; tokens.len()]
            }
        } else {
            vec![0.0; tokens.len()]
        };

        // token_type (optional)
        let token_types = if let Some(val) = gguf.get_metadata("tokenizer.ggml.token_type") {
            if let Some(arr) = val.as_array() {
                arr.values
                    .iter()
                    .map(|v| {
                        let i = match v {
                            MetadataValue::Int32(i) => *i,
                            MetadataValue::UInt32(u) => *u as i32,
                            MetadataValue::Int64(i) => *i as i32,
                            _ => 1,
                        };
                        TokenType::from_i32(i)
                    })
                    .collect()
            } else {
                vec![TokenType::Normal; tokens.len()]
            }
        } else {
            vec![TokenType::Normal; tokens.len()]
        };

        // merges (optional, for gpt2)
        let merges = if let Some(val) = gguf.get_metadata("tokenizer.ggml.merges") {
            if let Some(arr) = val.as_array() {
                let mut m = Vec::with_capacity(arr.values.len());
                for v in &arr.values {
                    if let Some(s) = v.as_string() {
                        m.push(s.to_string());
                    }
                }
                Some(m)
            } else {
                None
            }
        } else {
            None
        };

        let bos_id = gguf
            .get_metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let eos_id = gguf
            .get_metadata("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let unk_id = gguf
            .get_metadata("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let pad_id = gguf
            .get_metadata("tokenizer.ggml.padding_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let add_bos = gguf
            .get_metadata("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let add_eos = gguf
            .get_metadata("tokenizer.ggml.add_eos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut token_to_id = HashMap::new();
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        Ok(Self {
            model,
            tokens,
            scores,
            token_types,
            merges,
            bos_id,
            eos_id,
            unk_id,
            pad_id,
            add_bos,
            add_eos,
            token_to_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Encode text into token IDs
    ///
    /// Simple longest-match implementation:
    /// - Optionally prepend BOS
    /// - For llama model, replace spaces with "▁" (U+2581) for matching, but keep original text handling
    /// - Greedy longest prefix match in vocab
    /// - Fallback to byte tokens or UNK
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();

        if add_bos && self.add_bos {
            if let Some(bos) = self.bos_id {
                ids.push(bos);
            }
        }

        // For llama tokenizer, SentencePiece uses "▁" to represent space
        // We will attempt to match tokens directly, including "▁"
        // Simple approach: iterate over chars, but for longest match we need to consider byte string

        // Convert text to a string that uses "▁" for spaces if vocab contains "▁"
        let contains_underscore = self.tokens.iter().any(|t| t.contains('▁'));
        let processed = if contains_underscore && self.model == "llama" {
            // Replace spaces with "▁" and prepend "▁" if not starting with space? SentencePiece usually prepends ▁
            // For simplicity, replace " " with "▁" and keep as is
            // Also handle that first token often starts with ▁
            let mut s = text.replace(' ', "▁");
            // If text doesn't start with ▁ and vocab has tokens starting with ▁, prepend ▁ for first word
            // This is heuristic
            if !s.starts_with('▁') && !text.is_empty() {
                s = format!("▁{}", s);
            }
            s
        } else {
            text.to_string()
        };

        // Greedy longest match
        let chars: Vec<char> = processed.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut matched = false;
            // Try longest to shortest
            for j in (i + 1..=chars.len()).rev() {
                let substr: String = chars[i..j].iter().collect();
                if let Some(&id) = self.token_to_id.get(&substr) {
                    ids.push(id);
                    i = j;
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Try single char fallback
                let single: String = chars[i..i + 1].iter().collect();
                if let Some(&id) = self.token_to_id.get(&single) {
                    ids.push(id);
                } else {
                    // Try byte fallback: look for <0xNN> token or byte token
                    let byte = single.as_bytes().first().copied().unwrap_or(0);
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        ids.push(id);
                    } else if let Some(unk) = self.unk_id {
                        ids.push(unk);
                    } else {
                        // Last resort: skip
                    }
                }
                i += 1;
            }
        }

        if self.add_eos {
            if let Some(eos) = self.eos_id {
                ids.push(eos);
            }
        }

        ids
    }

    /// Decode token IDs into text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(tok) = self.tokens.get(id as usize) {
                // Skip control tokens like BOS, EOS for display? Keep them but they may be empty
                // For llama, BOS is often "<s>" which we may want to skip in final output
                // We'll include but handle ▁ -> space
                let mut s = tok.clone();
                // Handle byte tokens <0xNN>
                if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
                    if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
                        text.push(byte as char);
                        continue;
                    }
                }
                // Replace ▁ with space
                s = s.replace('▁', " ");
                text.push_str(&s);
            }
        }
        // Trim leading space that may come from initial ▁
        text.trim_start().to_string()
    }

    /// Decode single token
    pub fn decode_single(&self, id: u32) -> String {
        self.decode(&[id])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_gguf_with_tokenizer() -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&6u64.to_le_bytes()); // kv count

        fn write_string<W: Write>(w: &mut W, s: &str) {
            w.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            w.write_all(s.as_bytes()).unwrap();
        }
        fn write_u32<W: Write>(w: &mut W, v: u32) {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        fn write_u64<W: Write>(w: &mut W, v: u64) {
            w.write_all(&v.to_le_bytes()).unwrap();
        }

        // tokenizer.ggml.model = "llama"
        write_string(&mut buf, "tokenizer.ggml.model");
        write_u32(&mut buf, 8);
        write_string(&mut buf, "llama");

        // tokenizer.ggml.tokens = ["<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world"]
        write_string(&mut buf, "tokenizer.ggml.tokens");
        write_u32(&mut buf, 9); // array
        write_u32(&mut buf, 8); // string type
        write_u64(&mut buf, 7);
        for tok in ["<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world"] {
            write_string(&mut buf, tok);
        }

        // tokenizer.ggml.scores
        write_string(&mut buf, "tokenizer.ggml.scores");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 6); // float32
        write_u64(&mut buf, 7);
        for _ in 0..7 {
            write_u32(&mut buf, 0f32.to_bits());
        }

        // tokenizer.ggml.token_type
        write_string(&mut buf, "tokenizer.ggml.token_type");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 5); // int32
        write_u64(&mut buf, 7);
        for t in [2, 3, 3, 1, 1, 1, 1] {
            write_u32(&mut buf, t as u32);
        }

        // bos, eos
        write_string(&mut buf, "tokenizer.ggml.bos_token_id");
        write_u32(&mut buf, 4);
        write_u32(&mut buf, 1);
        write_string(&mut buf, "tokenizer.ggml.eos_token_id");
        write_u32(&mut buf, 4);
        write_u32(&mut buf, 2);

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(&buf).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn test_tokenizer_load() {
        let tmp = make_gguf_with_tokenizer();
        let model = crate::gguf::parse_gguf_file(tmp.path()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&model).unwrap();
        assert_eq!(tokenizer.vocab_size(), 7);
        assert_eq!(tokenizer.bos_id, Some(1));
        assert_eq!(tokenizer.eos_id, Some(2));
    }

    #[test]
    fn test_tokenizer_encode_decode() {
        let tmp = make_gguf_with_tokenizer();
        let model = crate::gguf::parse_gguf_file(tmp.path()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&model).unwrap();
        let ids = tokenizer.encode("hello world", false);
        // Should encode to [3,4] i.e. "▁hello", "▁world"
        assert_eq!(ids, vec![3, 4]);
        let text = tokenizer.decode(&ids);
        assert_eq!(text, "hello world");
    }
}
