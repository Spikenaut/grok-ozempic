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
}

pub type Result<T> = std::result::Result<T, GrokOzempicError>;
