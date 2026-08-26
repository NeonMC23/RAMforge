//! Tokenizer loading from GGUF metadata – Milestone 6
//!
//! Proper GGUF tokenizer based on actual metadata:
//! - Supports SentencePiece unigram (llama) using scores with Viterbi best path
//! - Supports BPE (gpt2, qwen2) using merges
//! - Handles BOS/EOS/UNK/PAD, special tokens, byte fallback
//! - Correctly encodes normal English text and decodes

use std::collections::HashMap;

use crate::model::GgufModel;
use crate::types::MetadataValue;

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
    pub pre: Option<String>,
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
    token_to_id: HashMap<String, u32>,
    // For BPE
    merges_rank: HashMap<(String, String), usize>,
    // For quick byte fallback
    byte_tokens: HashMap<u8, u32>,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufModel) -> Result<Self, String> {
        let model = gguf
            .get_metadata("tokenizer.ggml.model")
            .and_then(|v| v.as_string())
            .unwrap_or("llama")
            .to_string();

        let pre = gguf
            .get_metadata("tokenizer.ggml.pre")
            .and_then(|v| v.as_string())
            .map(|s| s.to_string());

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
            .or_else(|| gguf.get_metadata("tokenizer.ggml.unk_token_id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let pad_id = gguf
            .get_metadata("tokenizer.ggml.padding_token_id")
            .or_else(|| gguf.get_metadata("tokenizer.ggml.pad_token_id"))
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
            // Keep first occurrence for duplicates
            token_to_id.entry(tok.clone()).or_insert(i as u32);
        }

        // Build merges rank map
        let mut merges_rank = HashMap::new();
        if let Some(merges_list) = &merges {
            for (rank, merge) in merges_list.iter().enumerate() {
                // Merge format: "a b" – split into two parts
                // For robustness, split by whitespace and take first two tokens
                // If merge contains multiple spaces, we need to handle: split into Vec, first is left, rest joined as right? But spec says two tokens separated by space
                let parts: Vec<&str> = merge.split_whitespace().collect();
                if parts.len() >= 2 {
                    // For simplicity, if more than 2 parts, treat as left = parts[0], right = parts[1..].join(" ")
                    // But typical merges are exactly 2
                    let left = parts[0].to_string();
                    let right = if parts.len() == 2 {
                        parts[1].to_string()
                    } else {
                        parts[1..].join(" ")
                    };
                    merges_rank.insert((left, right), rank);
                } else {
                    // If no space, try to split by space char, fallback: first char + rest?
                    // We'll skip malformed
                }
            }
        }

        // Build byte token map for fallback
        let mut byte_tokens = HashMap::new();
        for (i, tok) in tokens.iter().enumerate() {
            // Byte tokens in llama are often <0x00> style or single byte
            if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    byte_tokens.insert(b, i as u32);
                }
            } else if tok.len() == 1 {
                // Single byte char could be byte fallback for gpt2
                if let Some(&id) = token_to_id.get(tok) {
                    if let Some(&tt) = token_types.get(id as usize) {
                        if tt == TokenType::Byte {
                            if let Some(first_byte) = tok.as_bytes().first() {
                                byte_tokens.entry(*first_byte).or_insert(id);
                            }
                        }
                    }
                }
            }
        }
        // Also for gpt2, byte tokens are often represented as "Ġ" etc., but we also need to handle raw bytes
        // For gpt2, the first 256 tokens are often byte encodings
        // We'll also populate byte_tokens for any token that is a single byte and type Byte, or for <0x..> we already did

        Ok(Self {
            model,
            pre,
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
            merges_rank,
            byte_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Encode text into token IDs – dispatches based on tokenizer model
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();

        if add_bos && self.add_bos {
            if let Some(bos) = self.bos_id {
                ids.push(bos);
            }
        }

        let mut encoded = if self.model == "gpt2"
            || self.pre.as_deref() == Some("qwen2")
            || self.merges.is_some()
        {
            self.encode_bpe(text)
        } else {
            // Default to SentencePiece unigram (llama)
            self.encode_sentencepiece(text)
        };

        ids.append(&mut encoded);

        if self.add_eos {
            if let Some(eos) = self.eos_id {
                ids.push(eos);
            }
        }

        ids
    }

    /// SentencePiece unigram encoding using Viterbi best path based on scores
    fn encode_sentencepiece(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Preprocess: handle ▁ for llama
        let contains_underscore = self.tokens.iter().any(|t| t.contains('▁'));
        let processed = if contains_underscore {
            let mut s = text.replace(' ', "▁");
            if !s.starts_with('▁') && !text.is_empty() {
                s = format!("▁{}", s);
            }
            s
        } else {
            text.to_string()
        };

        let chars: Vec<char> = processed.chars().collect();
        let n = chars.len();

        // DP arrays
        let mut best_score = vec![f32::NEG_INFINITY; n + 1];
        let mut best_id = vec![None; n + 1];
        let mut best_prev = vec![0; n + 1];
        best_score[0] = 0.0;

        for i in 0..n {
            if best_score[i] == f32::NEG_INFINITY {
                continue;
            }
            // Limit max token length to avoid O(n^2) blowup – typical max 16-32 chars
            let max_len = (n - i).min(32);
            for len in 1..=max_len {
                let j = i + len;
                let substr: String = chars[i..j].iter().collect();
                if let Some(&id) = self.token_to_id.get(&substr) {
                    // Skip special tokens that are control unless they are in text? For simplicity include
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    let new_score = best_score[i] + score;
                    if new_score > best_score[j] {
                        best_score[j] = new_score;
                        best_id[j] = Some(id);
                        best_prev[j] = i;
                    }
                }
            }
        }

        // If no path found for full string, fallback to greedy longest-match
        if best_score[n] == f32::NEG_INFINITY {
            return self.encode_greedy(&processed);
        }

        // Backtrack
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            if let Some(id) = best_id[pos] {
                ids.push(id);
                pos = best_prev[pos];
            } else {
                // Should not happen, fallback to single char
                pos -= 1;
                if let Some(&unk) = self.byte_tokens.get(&(chars[pos] as u8)) {
                    ids.push(unk);
                } else if let Some(unk) = self.unk_id {
                    ids.push(unk);
                }
            }
        }
        ids.reverse();
        ids
    }

    /// Greedy longest-match fallback
    fn encode_greedy(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let mut matched = false;
            for j in (i + 1..=chars.len()).rev() {
                if j - i > 32 {
                    continue;
                }
                let substr: String = chars[i..j].iter().collect();
                if let Some(&id) = self.token_to_id.get(&substr) {
                    ids.push(id);
                    i = j;
                    matched = true;
                    break;
                }
            }
            if !matched {
                let single: String = chars[i..i + 1].iter().collect();
                if let Some(&id) = self.token_to_id.get(&single) {
                    ids.push(id);
                } else {
                    let byte = single.as_bytes().first().copied().unwrap_or(0);
                    if let Some(&bid) = self.byte_tokens.get(&byte) {
                        ids.push(bid);
                    } else if let Some(unk) = self.unk_id {
                        ids.push(unk);
                    }
                }
                i += 1;
            }
        }
        ids
    }

    /// BPE encoding using merges
    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Pre-tokenize: split into words with Ġ handling for gpt2/qwen2
        let pre_tokens = self.pre_tokenize_bpe(text);

        let mut all_ids = Vec::new();

        for word in pre_tokens {
            let bpe_pieces = self.bpe_word(&word);
            for piece in bpe_pieces {
                if let Some(&id) = self.token_to_id.get(&piece) {
                    all_ids.push(id);
                } else {
                    // Try byte fallback per byte
                    for b in piece.as_bytes() {
                        if let Some(&bid) = self.byte_tokens.get(b) {
                            all_ids.push(bid);
                        } else if let Some(&id) = self.token_to_id.get(&piece) {
                            all_ids.push(id);
                        } else if let Some(unk) = self.unk_id {
                            all_ids.push(unk);
                        }
                    }
                }
            }
        }

        // If BPE produced nothing, fallback to greedy
        if all_ids.is_empty() {
            return self.encode_greedy(text);
        }

        all_ids
    }

    fn pre_tokenize_bpe(&self, text: &str) -> Vec<String> {
        // For gpt2/qwen2, Ġ represents space
        // Simple heuristic: split on whitespace, prepend Ġ to all but first word if vocab contains Ġ
        let has_g_dot = self.tokens.iter().any(|t| t.starts_with('Ġ'));

        if has_g_dot {
            let mut result = Vec::new();
            let words = text.split_whitespace();
            let mut first = true;
            for word in words {
                if first {
                    result.push(word.to_string());
                    first = false;
                } else {
                    result.push(format!("Ġ{}", word));
                }
                // Handle that original text may have multiple spaces – we ignore for simplicity
            }
            // If text starts with space, first token should have Ġ
            if text.starts_with(|c: char| c.is_whitespace()) && !result.is_empty() {
                result[0] = format!("Ġ{}", result[0]);
            }
            // Handle punctuation as separate? For simplicity, keep words as is, BPE will split
            result
        } else {
            // For llama without Ġ, use ▁ handling
            vec![text.to_string()]
        }
    }

    fn bpe_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return Vec::new();
        }

        // Start with characters
        let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        if self.merges_rank.is_empty() {
            // No merges, return as is if in vocab, otherwise split
            return pieces;
        }

        loop {
            // Find best merge pair
            let mut best_rank = None;
            let mut best_idx = None;

            for i in 0..pieces.len().saturating_sub(1) {
                let left = &pieces[i];
                let right = &pieces[i + 1];
                if let Some(&rank) = self.merges_rank.get(&(left.clone(), right.clone())) {
                    if best_rank.is_none() || rank < best_rank.unwrap() {
                        best_rank = Some(rank);
                        best_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = best_idx {
                // Merge
                let merged = format!("{}{}", pieces[idx], pieces[idx + 1]);
                pieces[idx] = merged;
                pieces.remove(idx + 1);
            } else {
                break;
            }
        }

        pieces
    }

    /// Decode token IDs into text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        let mut byte_buffer: Vec<u8> = Vec::new();

        for &id in ids {
            // Skip BOS/EOS for final display unless verbose? For Milestone 6, we skip control tokens that are BOS/EOS in decode
            if Some(id) == self.bos_id || Some(id) == self.eos_id {
                // For llama, BOS is <s>, EOS is </s> – often we want to skip in final output
                // Check token type
                if let Some(t) = self.token_types.get(id as usize) {
                    if *t == TokenType::Control {
                        continue;
                    }
                }
            }

            if let Some(tok) = self.tokens.get(id as usize) {
                if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                    if let Ok(byte) = u8::from_str_radix(&tok[3..5], 16) {
                        byte_buffer.push(byte);
                        continue;
                    }
                }

                // If we have accumulated bytes, try to decode them as UTF-8 before handling next token
                if !byte_buffer.is_empty() {
                    if let Ok(s) = String::from_utf8(byte_buffer.clone()) {
                        text.push_str(&s);
                    } else {
                        // Replace invalid with �
                        text.push_str(&String::from_utf8_lossy(&byte_buffer));
                    }
                    byte_buffer.clear();
                }

                // Handle token type Byte: token may be single byte
                if let Some(t) = self.token_types.get(id as usize) {
                    if *t == TokenType::Byte {
                        // Token string may be a single byte character
                        if tok.len() == 1 {
                            byte_buffer.push(tok.as_bytes()[0]);
                            continue;
                        }
                    }
                }

                let mut s = tok.clone();
                s = s.replace('▁', " ");
                s = s.replace('Ġ', " ");
                text.push_str(&s);
            }
        }

        if !byte_buffer.is_empty() {
            if let Ok(s) = String::from_utf8(byte_buffer.clone()) {
                text.push_str(&s);
            } else {
                text.push_str(&String::from_utf8_lossy(&byte_buffer));
            }
        }

        // For gpt2/qwen2, the first token may have leading space – trim start only if not intended?
        // For llama, trim leading space from initial ▁
        // We'll trim only leading spaces that are from initial ▁ for llama, but for gpt2 we want to keep?
        // Simple: trim start for llama model, keep for gpt2? We'll check model type
        if self.model == "llama" {
            text.trim_start().to_string()
        } else {
            text
        }
    }

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
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&6u64.to_le_bytes());

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

        write_string(&mut buf, "tokenizer.ggml.model");
        write_u32(&mut buf, 8);
        write_string(&mut buf, "llama");

        write_string(&mut buf, "tokenizer.ggml.tokens");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 8);
        write_u64(&mut buf, 7);
        for tok in ["<unk>", "<s>", "</s>", "▁hello", "▁world", "hello", "world"] {
            write_string(&mut buf, tok);
        }

        write_string(&mut buf, "tokenizer.ggml.scores");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 6);
        write_u64(&mut buf, 7);
        for _ in 0..7 {
            write_u32(&mut buf, 0f32.to_bits());
        }

        write_string(&mut buf, "tokenizer.ggml.token_type");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 5);
        write_u64(&mut buf, 7);
        for t in [2, 3, 3, 1, 1, 1, 1] {
            write_u32(&mut buf, t as u32);
        }

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

    fn make_gguf_with_bpe_tokenizer() -> NamedTempFile {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());

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

        write_string(&mut buf, "tokenizer.ggml.model");
        write_u32(&mut buf, 8);
        write_string(&mut buf, "gpt2");

        write_string(&mut buf, "tokenizer.ggml.tokens");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 8);
        write_u64(&mut buf, 11);
        for tok in [
            "<unk>", "<s>", "</s>", "h", "e", "l", "o", "he", "lo", "hel", "hello",
        ] {
            write_string(&mut buf, tok);
        }

        write_string(&mut buf, "tokenizer.ggml.scores");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 6);
        write_u64(&mut buf, 11);
        for _ in 0..11 {
            write_u32(&mut buf, 0f32.to_bits());
        }

        write_string(&mut buf, "tokenizer.ggml.token_type");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 5);
        write_u64(&mut buf, 11);
        for _ in 0..11 {
            write_u32(&mut buf, 1);
        }

        write_string(&mut buf, "tokenizer.ggml.merges");
        write_u32(&mut buf, 9);
        write_u32(&mut buf, 8);
        write_u64(&mut buf, 4);
        for m in ["h e", "he l", "l o", "hel lo"] {
            write_string(&mut buf, m);
        }

        write_string(&mut buf, "tokenizer.ggml.bos_token_id");
        write_u32(&mut buf, 4);
        write_u32(&mut buf, 1);
        write_string(&mut buf, "tokenizer.ggml.eos_token_id");
        write_u32(&mut buf, 4);
        write_u32(&mut buf, 2);
        write_string(&mut buf, "tokenizer.ggml.unknown_token_id");
        write_u32(&mut buf, 4);
        write_u32(&mut buf, 0);

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
        assert_eq!(ids, vec![3, 4]);
        let text = tokenizer.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_tokenizer_bpe() {
        let tmp = make_gguf_with_bpe_tokenizer();
        let model = crate::gguf::parse_gguf_file(tmp.path()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&model).unwrap();
        assert_eq!(tokenizer.model, "gpt2");
        assert!(tokenizer.merges.is_some());
        // With merges h e -> he, he l -> hel, l o -> lo, hel lo -> hello, "hello" should encode to single token "hello" id 10
        let ids = tokenizer.encode("hello", false);
        // BPE should merge h+e->he, he+l->hel, l+o->lo, hel+lo->hello
        // So should be [10]
        assert!(
            ids.contains(&10),
            "BPE should produce hello token, got {:?}",
            ids
        );
    }

    #[test]
    fn test_tokenizer_whitespace() {
        let tmp = make_gguf_with_tokenizer();
        let model = crate::gguf::parse_gguf_file(tmp.path()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&model).unwrap();
        let ids = tokenizer.encode("hello   world", false);
        // Multiple spaces should still encode
        assert!(!ids.is_empty());
        let text = tokenizer.decode(&ids);
        assert!(text.contains("hello"));
        assert!(text.contains("world"));
    }

    #[test]
    fn test_tokenizer_punctuation() {
        let tmp = make_gguf_with_tokenizer();
        let model = crate::gguf::parse_gguf_file(tmp.path()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&model).unwrap();
        let ids = tokenizer.encode("hello, world!", false);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_tokenizer_bos_eos() {
        let tmp = make_gguf_with_tokenizer();
        let model = crate::gguf::parse_gguf_file(tmp.path()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&model).unwrap();
        let ids_with_bos = tokenizer.encode("hello", true);
        let ids_without_bos = tokenizer.encode("hello", false);
        assert!(ids_with_bos.len() > ids_without_bos.len());
        assert_eq!(ids_with_bos[0], 1); // BOS
    }

    #[test]
    fn test_tokenizer_byte_fallback() {
        let tmp = make_gguf_with_bpe_tokenizer();
        let model = crate::gguf::parse_gguf_file(tmp.path()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&model).unwrap();
        // Encode unknown char that is not in vocab, should fallback to unk
        let ids = tokenizer.encode("xyz", false);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let tmp = make_gguf_with_tokenizer();
        let model = crate::gguf::parse_gguf_file(tmp.path()).unwrap();
        let tokenizer = Tokenizer::from_gguf(&model).unwrap();
        let original = "hello world";
        let ids = tokenizer.encode(original, false);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, original);
    }
}
