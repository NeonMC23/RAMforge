//! RAMforge Runtime – placeholder for future out-of-core inference
//!
//! This milestone focuses only on GGUF inspection. The runtime will be
//! implemented in later milestones to support hierarchical memory management
//! across RAM, VRAM, and storage.

use ramforge_core::GgufModel;

/// Placeholder struct for future runtime planning
#[derive(Debug)]
pub struct RuntimePlaceholder {
    pub model_path: std::path::PathBuf,
}

impl RuntimePlaceholder {
    pub fn new(model: &GgufModel) -> Self {
        Self {
            model_path: model.path.clone(),
        }
    }

    pub fn info(&self) -> &'static str {
        "RAMforge runtime not yet implemented – inspection only in milestone 1"
    }
}
