//! grok-ozempic — SNN-logic quantization for Grok models
//!
//! Turns Grok's massive MoE into a sparse, membrane-driven, ternary spiking system.
//! "Ozempic for Grok" — we keep the intelligence, drop the fat.

pub mod types;
pub mod core;
pub mod error;

pub use types::{
    HybridConfig, HybridOutput, QuantizationConfig, QuantizationInputFormat, TelemetrySnapshot,
    TensorPrecision, GROK1_HIDDEN_DIM,
};
pub use core::HybridModel;

// Re-export main types for convenience
pub use crate::core::projector::Projector;
pub use crate::core::olmoe::OLMoE;
pub use crate::core::stream::{append_grok1_arch_metadata, run_quantization, ShardStats};
pub use crate::core::gguf::{GgufMetaValue, GgufStreamWriter, TensorHeader};
pub use crate::core::gguf_read::{verify_gguf_file, GgufVerifyReport};
pub use crate::core::quantizer::{quantize_f16, quantize_f32, QuantizedTensor};
