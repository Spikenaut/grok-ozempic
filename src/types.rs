use serde::{Deserialize, Serialize};

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
    pub model_path: String,           // path to safetensors or GGUF
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub snn_steps: usize,
    pub projection_mode: ProjectionMode,
    pub execution_mode: ExecutionMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    SpikingTernary,   // your main mode
    // add others later if needed
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    SpikingSim,       // GIF + ternary
    DenseSim,         // for comparison
}

#[derive(Clone, Debug)]
pub struct HybridOutput {
    pub spike_train: Vec<Vec<usize>>,
    pub embedding: Vec<f32>,
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
}

pub const EMBEDDING_DIM: usize = 2048;  // adjust if Grok uses different dim
