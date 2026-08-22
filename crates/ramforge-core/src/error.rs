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

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ParseSizeError {
    #[error("empty size string")]
    Empty,

    #[error("invalid size format: {0}")]
    InvalidFormat(String),

    #[error("unknown unit: {0}")]
    UnknownUnit(String),

    #[error("size must be positive, got: {0}")]
    NonPositive(String),

    #[error("size overflow: {0}")]
    Overflow(String),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MemoryError {
    #[error("insufficient memory: requested {requested} bytes for '{name}', but only {available} bytes available (total {total}, used {used})")]
    Insufficient {
        name: String,
        requested: u64,
        available: u64,
        total: u64,
        used: u64,
    },

    #[error("allocation '{name}' already exists")]
    AlreadyExists { name: String },

    #[error("allocation '{name}' not found")]
    NotFound { name: String },

    #[error("invalid allocation size: {0} (must be > 0)")]
    InvalidSize(u64),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CacheError {
    #[error("entry too large: {size} bytes exceeds cache capacity {capacity} bytes")]
    TooLarge { size: u64, capacity: u64 },

    #[error("cache error: {0}")]
    General(String),
}

#[derive(Debug, Error)]
pub enum DataSourceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error("invalid tensor offset: {0}")]
    InvalidOffset(String),

    #[error("invalid range: {0}")]
    InvalidRange(String),

    #[error("tensor data extends beyond file: tensor '{name}' file_offset {file_offset} + byte_length {byte_length} > file_size {file_size}")]
    OutOfBounds {
        name: String,
        file_offset: u64,
        byte_length: u64,
        file_size: u64,
    },

    #[error("tensor byte length unknown: cannot determine size for tensor '{0}' (type {1})")]
    UnknownByteLength(String, String),

    #[error("general data source error: {0}")]
    General(String),
}
