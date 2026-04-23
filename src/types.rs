use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Quantization pipeline types
// ---------------------------------------------------------------------------

/// Controls which precision is applied to a given tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorPrecision {
    /// Two-bit ternary {-1, 0, +1} with saliency-gated GIF threshold.
    TernarySnN,
    /// Keep original FP16 — used for MoE routing gates.
    Fp16,
    /// Keep the tensor in its source dtype (routing-critical layers etc.).
    ///
    /// **Reserved in phase 1.** This variant is declared so the
    /// `xai-dissect` manifest contract can reference it, but it is **not
    /// yet consumed** by [`crate::core::stream::run_quantization`]. A
    /// follow-up PR will wire it through the selection / precision seams.
    ///
    /// **Temporary semantic shortcut:** during the transitional window
    /// the wiring code may alias `Preserve` to FP16 passthrough. That
    /// alias is explicitly temporary and must be removed once
    /// source-dtype passthrough lands. See the phase 2/3 tracking issue.
    //
    // TODO(phase-3): remove any Preserve→FP16 alias once source-dtype
    // passthrough is implemented in stream.rs + weight_pack.rs.
    Preserve,
}

/// Weight container layout for [`QuantizationConfig::input_dir`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationInputFormat {
    /// Hugging Face–style `*.safetensors` shards (memory-mapped).
    #[default]
    Safetensors,
    /// Directory of per-tensor `*.npy` files — **primary layout for JAX/Flax** (NumPy export).
    /// Use `__` in the filename stem in place of `.` in tensor names
    /// (e.g. `blk__0__weight.npy` → `blk.0.weight`).
    NpyDir,
}

/// Configuration for the out-of-core quantization pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantizationConfig {
    /// Directory that holds weight shards (see [`QuantizationInputFormat`]).
    pub input_dir: String,
    /// Path for the output **GOZ1** packed checkpoint (see `weight_pack`).
    pub output_path: String,
    /// GIF saliency threshold ratio: weights with |w| < threshold × rms(layer)
    /// are silenced to 0; the rest become ±1.
    pub gif_threshold: f32,
    /// Tensor name substrings that identify routing / gate tensors which should
    /// remain in FP16 instead of being ternary-quantized.
    ///
    /// **Legacy field.** When an `xai-dissect` manifest is supplied via
    /// [`QuantizationConfig::manifest_path`], the manifest wins and this
    /// list is ignored. A deprecation log line will be emitted by the
    /// selection seam introduced in phase 2.
    pub router_patterns: Vec<String>,
    /// Input layout: safetensors shards vs flat `.npy` tensors.
    pub input_format: QuantizationInputFormat,
    /// Optional path to an `xai-dissect` JSON manifest (schema v1).
    ///
    /// **Reserved in phase 1.** The field is exposed so callers can start
    /// plumbing manifests through configuration, but
    /// [`crate::core::stream::run_quantization`] does **not** consume it
    /// yet. Wiring lands in phase 2 via dedicated selection and precision
    /// modules.
    ///
    /// Precedence (phase 2+): this explicit path > `GROK_OZEMPIC_MANIFEST`
    /// env var > in-tree `dissect/grok-1/baseline.json` fallback (only
    /// when [`Self::use_embedded_baseline`] is `true`) > legacy
    /// `router_patterns` heuristic.
    #[serde(default)]
    pub manifest_path: Option<PathBuf>,
    /// Opt in to the compiled-in non-authoritative Grok-1 baseline
    /// manifest as a fallback when neither
    /// [`Self::manifest_path`] nor `GROK_OZEMPIC_MANIFEST` is set.
    ///
    /// Default is `false` so upgrading from phase 1 preserves existing
    /// legacy-heuristic behavior. Set to `true` for a Grok-1 export to
    /// pick up the reference manifest without pointing at a file.
    #[serde(default)]
    pub use_embedded_baseline: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            input_dir: String::new(),
            output_path: String::new(),
            gif_threshold: 0.05,
            router_patterns: Vec::new(),
            input_format: QuantizationInputFormat::Safetensors,
            manifest_path: None,
            use_embedded_baseline: false,
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    pub timestamp_ms: u64,
}

impl TelemetrySnapshot {
    pub fn thermal_stress(&self) -> f32 {
        ((self.gpu_temp_c - 60.0) / 30.0).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridConfig {
    #[serde(default)]
    pub model_path: String, // path to weights / GOZ1 pack as needed by your runner
    /// Hidden / embedding size (Grok-1 uses 6144).
    #[serde(default = "default_grok_embedding_dim")]
    pub embedding_dim: usize,
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_top_k_experts")]
    pub top_k_experts: usize,
    #[serde(default = "default_snn_steps")]
    pub snn_steps: usize,
    #[serde(default)]
    pub projection_mode: ProjectionMode,
    #[serde(default)]
    pub execution_mode: ExecutionMode,
}

fn default_grok_embedding_dim() -> usize {
    GROK1_HIDDEN_DIM
}
fn default_num_experts() -> usize {
    8
}
fn default_top_k_experts() -> usize {
    2
}
fn default_snn_steps() -> usize {
    4
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            embedding_dim: GROK1_HIDDEN_DIM,
            num_experts: 8,
            top_k_experts: 2,
            snn_steps: 4,
            projection_mode: ProjectionMode::SpikingTernary,
            execution_mode: ExecutionMode::SpikingSim,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    #[default]
    SpikingTernary, // your main mode
                    // add others later if needed
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    #[default]
    SpikingSim, // GIF + ternary
    DenseSim,   // for comparison
}

#[derive(Clone, Debug)]
pub struct HybridOutput {
    pub spike_train: Vec<Vec<usize>>,
    pub embedding: Vec<f32>,
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
}

/// Grok-1 hidden size (`hidden_size` / `embedding_length` in model cards).
pub const GROK1_HIDDEN_DIM: usize = 6144;
