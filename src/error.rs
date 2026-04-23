use thiserror::Error;

#[derive(Debug, Error)]
pub enum GrokOzempicError {
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("expert index {index} out of range (num_experts={num_experts})")]
    ExpertOutOfRange { index: usize, num_experts: usize },

    #[error("quantization error: {0}")]
    Quantization(String),

    #[error("GOZ1 pack write error: {0}")]
    PackWrite(String),

    #[error("manifest I/O error at {path}: {source}")]
    ManifestIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("manifest parse error at {path}: {source}")]
    ManifestParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("unsupported manifest schema_version: got {got}, expected {expected}")]
    ManifestSchemaVersion { got: u32, expected: u32 },

    #[error(
        "manifest tensor_name_convention mismatch: got {got:?}, expected {expected:?}"
    )]
    ManifestNameConventionMismatch { got: String, expected: String },

    #[error("unsupported manifest precision tier: {got:?}")]
    ManifestInvalidPrecision { got: String },
}

pub type Result<T> = std::result::Result<T, GrokOzempicError>;
