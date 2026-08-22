use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid GGUF magic: expected 'GGUF', found {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("unsupported GGUF version: {0} (expected 1, 2, or 3)")]
    UnsupportedVersion(u32),

    #[error("file is truncated: {0}")]
    Truncated(String),

    #[error("invalid UTF-8 string: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    #[error("string contains invalid UTF-8")]
    InvalidStringUtf8,

    #[error("invalid metadata value type: {0}")]
    InvalidMetadataType(u32),

    #[error("invalid string length: {0} (too large)")]
    InvalidStringLength(u64),

    #[error("invalid dimensions count: {0}")]
    InvalidDimensionsCount(u32),

    #[error("metadata key too long or invalid")]
    InvalidMetadataKey,

    #[error("tensor name too long or invalid")]
    InvalidTensorName,

    #[error("general error: {0}")]
    General(String),
}

pub type Result<T> = std::result::Result<T, GgufError>;
