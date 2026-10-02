//! RAMforge's Phase-2 adapter for the pinned ForgeCore API.
//!
//! This module is deliberately small. RAMforge supplies the model path,
//! token IDs, positions, and sampling policy; ForgeCore owns model loading,
//! context state, batch construction, decode, and native logits extraction.
//! No llama.cpp/ggml symbols or numerical model operations belong here.

use std::fmt;
use std::path::{Path, PathBuf};

use forge_core::{BatchBuilder, Context, Model, ModelOptions};

/// Errors translated at the RAMforge/ForgeCore boundary.
///
/// ForgeCore currently exposes one string-backed error type. The adapter adds
/// the operation domain so callers can report model, context, batch, decode,
/// and logits failures without depending on ForgeCore's internal strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeCoreAdapterError {
    Configuration(String),
    Model(String),
    Context(String),
    Batch(String),
    Decode(String),
    Logits(String),
}

impl fmt::Display for ForgeCoreAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(message) => write!(formatter, "ForgeCore configuration: {message}"),
            Self::Model(message) => write!(formatter, "ForgeCore model: {message}"),
            Self::Context(message) => write!(formatter, "ForgeCore context: {message}"),
            Self::Batch(message) => write!(formatter, "ForgeCore batch: {message}"),
            Self::Decode(message) => write!(formatter, "ForgeCore decode: {message}"),
            Self::Logits(message) => write!(formatter, "ForgeCore logits: {message}"),
        }
    }
}

impl std::error::Error for ForgeCoreAdapterError {}

/// CPU model-loading options exposed by the RAMforge adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ForgeCoreModelOptions {
    /// Ask libllama to validate tensor payloads while loading.
    pub check_tensors: bool,
}

/// Metadata copied from the ForgeCore model handle for RAMforge diagnostics
/// and compatibility checks. The values are observations from ForgeCore, not
/// a second execution configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeCoreModelMetadata {
    pub n_params: u64,
    pub vocab_size: u32,
    pub size_bytes: u64,
    pub context_length: u32,
    pub embedding_length: u32,
    pub layer_count: u32,
    pub head_count: u32,
    pub kv_head_count: u32,
    pub description: String,
}

/// RAMforge-owned handle for one ForgeCore/libllama model.
///
/// The underlying ForgeCore model is CPU-only in this phase. Cloning this
/// wrapper only shares ForgeCore's native model; it does not copy weights.
#[derive(Debug, Clone)]
pub struct ForgeCoreModel {
    model: Model,
    path: PathBuf,
    metadata: ForgeCoreModelMetadata,
}

impl ForgeCoreModel {
    /// Load a model through ForgeCore/libllama using CPU defaults.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ForgeCoreAdapterError> {
        Self::load_with_options(path, ForgeCoreModelOptions::default())
    }

    /// Load a model through ForgeCore with the adapter's CPU-only options.
    pub fn load_with_options(
        path: impl AsRef<Path>,
        options: ForgeCoreModelOptions,
    ) -> Result<Self, ForgeCoreAdapterError> {
        let path = path.as_ref().to_path_buf();
        let mut model_options = ModelOptions::default();
        model_options.check_tensors = options.check_tensors;
        let model = Model::load_with_options(&path, &model_options)
            .map_err(|error| ForgeCoreAdapterError::Model(error.to_string()))?;

        let metadata = ForgeCoreModelMetadata {
            n_params: model.n_params(),
            vocab_size: model
                .vocab_size()
                .map_err(|error| ForgeCoreAdapterError::Model(error.to_string()))?,
            size_bytes: model.size_bytes(),
            context_length: model
                .n_ctx_train()
                .map_err(|error| ForgeCoreAdapterError::Model(error.to_string()))?,
            embedding_length: model
                .n_embd()
                .map_err(|error| ForgeCoreAdapterError::Model(error.to_string()))?,
            layer_count: model
                .n_layer()
                .map_err(|error| ForgeCoreAdapterError::Model(error.to_string()))?,
            head_count: model
                .n_head()
                .map_err(|error| ForgeCoreAdapterError::Model(error.to_string()))?,
            kv_head_count: model
                .n_head_kv()
                .map_err(|error| ForgeCoreAdapterError::Model(error.to_string()))?,
            description: model
                .description()
                .map_err(|error| ForgeCoreAdapterError::Model(error.to_string()))?,
        };

        Ok(Self {
            model,
            path,
            metadata,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn metadata(&self) -> &ForgeCoreModelMetadata {
        &self.metadata
    }

    /// Open a single CPU context. The returned session owns a shared model
    /// handle, so its lifetime is independent of this wrapper value.
    pub fn open_context(
        &self,
        options: ForgeCoreContextOptions,
    ) -> Result<ForgeCoreSession, ForgeCoreAdapterError> {
        options.validate()?;
        let mut context_options = forge_core::ContextOptions::default();
        context_options.n_ctx = options.n_ctx.unwrap_or(0);
        context_options.n_threads = u32::try_from(options.n_threads).map_err(|_| {
            ForgeCoreAdapterError::Configuration(format!(
                "CPU thread count {} exceeds ForgeCore's u32 limit",
                options.n_threads
            ))
        })?;
        let context = Context::open(&self.model, &context_options)
            .map_err(|error| ForgeCoreAdapterError::Context(error.to_string()))?;

        Ok(ForgeCoreSession {
            model: self.clone(),
            context,
            options,
        })
    }
}

/// Context options accepted by the Phase-2 CPU adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForgeCoreContextOptions {
    /// `None` selects the model's training context length. `Some(value)`
    /// requests that explicit length from ForgeCore.
    pub n_ctx: Option<u32>,
    /// CPU worker count forwarded to ForgeCore/libllama.
    pub n_threads: usize,
}

impl Default for ForgeCoreContextOptions {
    fn default() -> Self {
        Self {
            n_ctx: None,
            n_threads: 1,
        }
    }
}

impl ForgeCoreContextOptions {
    fn validate(self) -> Result<(), ForgeCoreAdapterError> {
        if self.n_threads == 0 {
            return Err(ForgeCoreAdapterError::Configuration(
                "CPU thread count must be greater than zero".to_string(),
            ));
        }
        if self.n_threads > u32::MAX as usize {
            return Err(ForgeCoreAdapterError::Configuration(format!(
                "CPU thread count {} exceeds ForgeCore's u32 limit",
                self.n_threads
            )));
        }
        if self.n_ctx == Some(0) {
            return Err(ForgeCoreAdapterError::Configuration(
                "explicit context length must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Owned logits returned by a ForgeCore decode.
///
/// `values` is copied from ForgeCore's owned `Logits` object before the
/// adapter returns. RAMforge's sampler may therefore read it without holding
/// a native context borrow.
#[derive(Debug, Clone, PartialEq)]
pub struct ForgeCoreLogits {
    token_index: usize,
    values: Vec<f32>,
}

impl ForgeCoreLogits {
    pub fn token_index(&self) -> usize {
        self.token_index
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn n_vocab(&self) -> usize {
        self.values.len()
    }
}

/// One RAMforge-owned session over ForgeCore's model/context/batch/decode API.
#[derive(Debug)]
pub struct ForgeCoreSession {
    model: ForgeCoreModel,
    context: Context,
    options: ForgeCoreContextOptions,
}

impl ForgeCoreSession {
    pub fn model(&self) -> &ForgeCoreModel {
        &self.model
    }

    pub fn context_options(&self) -> ForgeCoreContextOptions {
        self.options
    }

    pub fn context_length(&self) -> u32 {
        self.context.n_ctx()
    }

    pub fn vocab_size(&self) -> u32 {
        self.context.n_vocab()
    }

    /// Build one token batch, decode it through ForgeCore, and copy the
    /// logits for the final token in the batch back to RAMforge.
    ///
    /// `start_position` is a llama sequence position, not a token ID. The
    /// adapter intentionally does not maintain or invent KV/residency policy;
    /// callers provide positions and ForgeCore validates native context state.
    pub fn decode_tokens(
        &mut self,
        tokens: &[u32],
        start_position: u32,
    ) -> Result<ForgeCoreLogits, ForgeCoreAdapterError> {
        if tokens.is_empty() {
            return Err(ForgeCoreAdapterError::Batch(
                "at least one token is required".to_string(),
            ));
        }

        let mut builder = BatchBuilder::new(self.model.metadata.vocab_size);
        for (offset, &token) in tokens.iter().enumerate() {
            let offset = u32::try_from(offset).map_err(|_| {
                ForgeCoreAdapterError::Batch("batch offset exceeds u32".to_string())
            })?;
            let position = start_position.checked_add(offset).ok_or_else(|| {
                ForgeCoreAdapterError::Batch("token position overflows u32".to_string())
            })?;
            builder
                .push(token, position, offset as usize + 1 == tokens.len())
                .map_err(|error| ForgeCoreAdapterError::Batch(error.to_string()))?;
        }
        let token_index = u32::try_from(tokens.len() - 1).map_err(|_| {
            ForgeCoreAdapterError::Batch("batch logits index exceeds u32".to_string())
        })?;
        let batch = builder
            .build()
            .map_err(|error| ForgeCoreAdapterError::Batch(error.to_string()))?;

        self.context
            .decode(&batch)
            .map_err(|error| ForgeCoreAdapterError::Decode(error.to_string()))?;
        self.logits(token_index)
    }

    /// Copy logits for a position in the most recently decoded batch.
    ///
    /// This is intentionally a thin safe forwarding method: an unflagged or
    /// out-of-range index remains a ForgeCore logits error rather than being
    /// replaced with another position or an implicit fallback.
    pub fn logits(&mut self, token_index: u32) -> Result<ForgeCoreLogits, ForgeCoreAdapterError> {
        let logits = self
            .context
            .logits(token_index)
            .map_err(|error| ForgeCoreAdapterError::Logits(error.to_string()))?;
        Ok(ForgeCoreLogits {
            token_index: token_index as usize,
            values: logits.values().to_vec(),
        })
    }

    pub fn decode_token(
        &mut self,
        token: u32,
        position: u32,
    ) -> Result<ForgeCoreLogits, ForgeCoreAdapterError> {
        self.decode_tokens(&[token], position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_defaults_match_phase_two_cpu_contract() {
        let options = ForgeCoreContextOptions::default();
        assert_eq!(options.n_ctx, None);
        assert_eq!(options.n_threads, 1);
        assert!(options.validate().is_ok());
    }

    #[test]
    fn context_options_reject_invalid_values_before_native_calls() {
        assert_eq!(
            ForgeCoreContextOptions {
                n_ctx: None,
                n_threads: 0,
            }
            .validate()
            .unwrap_err(),
            ForgeCoreAdapterError::Configuration(
                "CPU thread count must be greater than zero".to_string()
            )
        );
        assert!(ForgeCoreContextOptions {
            n_ctx: Some(0),
            n_threads: 1,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn logits_are_owned_values() {
        let logits = ForgeCoreLogits {
            token_index: 2,
            values: vec![1.0, -2.0],
        };
        assert_eq!(logits.token_index(), 2);
        assert_eq!(logits.n_vocab(), 2);
        assert_eq!(logits.values(), &[1.0, -2.0]);
    }
}
