//! grok-ozempic — SNN-logic quantization for Grok models
//!
//! Turns Grok's massive MoE into a sparse, membrane-driven, ternary spiking system.
//! "Ozempic for Grok" — we keep the intelligence, drop the fat.
//!
//! The batch pipeline writes a **GOZ1** packed checkpoint (see [`core::weight_pack`]);
//! weights are expected from JAX/NumPy (`.npy`) or safetensors — see the README.

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
pub use crate::core::ozempic::OzempicMoE;
pub use crate::core::stream::{append_grok1_arch_metadata, run_quantization, ShardStats};
pub use crate::core::weight_pack::{
    PackMetaValue, PackStreamWriter, PackTensorHeader, TENSOR_F16, TENSOR_TERNARY,
};
pub use crate::core::weight_pack_read::{verify_pack_file, PackVerifyReport};
pub use crate::core::quantizer::{quantize_f16, quantize_f32, QuantizedTensor};
pub use crate::core::manifest::{
    embedded_grok1_baseline, load_manifest, parse_manifest_bytes, DissectManifest, Fp16Entry,
    ManifestBlock, ManifestDefaults, ManifestModel, ManifestProducedBy, PreserveEntry,
    TernaryCandidate, GROK1_BASELINE_JSON, MANIFEST_NAME_CONVENTION_V1,
    MANIFEST_SCHEMA_VERSION,
};
pub use crate::core::precision::{decide as precision_decide, parse_precision_str};
pub use crate::core::selection::{
    classify as selection_classify, glob_match, TensorClass, LEGACY_DEFAULT_ROUTER_PATTERNS,
};
