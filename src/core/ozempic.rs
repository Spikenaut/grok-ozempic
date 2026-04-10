use half::f16;

use crate::{
    error::{GrokOzempicError, Result},
    types::HybridConfig,
};

/// Sparse Mixture-of-Experts router for Grok's MoE layers.
///
/// Implements top-k gating: for each input embedding only `top_k` experts
/// are activated, everything else is zeroed out.
pub struct OzempicMoE {
    pub num_experts: usize,
    pub top_k: usize,
    pub embedding_dim: usize,
    /// Expert gate weights: shape [num_experts × embedding_dim].
    gate_weights: Vec<Vec<f32>>,
}

impl OzempicMoE {
    /// Create a new `OzempicMoE` with zero-initialised gate weights (uniform routing
    /// until you call [`Self::load_gates_from_fp16_stacked_experts`] with real
    /// router / gate tensors).
    pub fn new(num_experts: usize, top_k: usize, embedding_dim: usize) -> Self {
        Self {
            num_experts,
            top_k,
            embedding_dim,
            gate_weights: vec![vec![0.0; embedding_dim]; num_experts],
        }
    }

    /// Build an `OzempicMoE` from a [`HybridConfig`].
    pub fn from_config(config: &HybridConfig) -> Self {
        Self::new(
            config.num_experts,
            config.top_k_experts,
            config.embedding_dim,
        )
    }

    /// Load all expert gate rows from a single FP16 blob: layout is
    /// `[num_experts × embedding_dim]` values in row-major order (each row is
    /// one expert's gate vector, little-endian IEEE half). Typical source: a
    /// quantized MoE router / gate tensor (e.g. from a GOZ1 pack or JAX `.npy`).
    pub fn load_gates_from_fp16_stacked_experts(&mut self, data: &[u8]) -> Result<()> {
        let need = self
            .num_experts
            .checked_mul(self.embedding_dim)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| GrokOzempicError::InvalidConfig(
                "gate buffer size overflow".into(),
            ))?;
        if data.len() != need {
            return Err(GrokOzempicError::DimensionMismatch {
                expected: need,
                got: data.len(),
            });
        }
        self.gate_weights.clear();
        self.gate_weights.reserve(self.num_experts);
        for e in 0..self.num_experts {
            let mut row = Vec::with_capacity(self.embedding_dim);
            for i in 0..self.embedding_dim {
                let o = (e * self.embedding_dim + i) * 2;
                let bits = u16::from_le_bytes([data[o], data[o + 1]]);
                row.push(f16::from_bits(bits).to_f32());
            }
            self.gate_weights.push(row);
        }
        Ok(())
    }

    /// Set the gate weight vector for a single expert.
    pub fn set_expert_weights(&mut self, expert_idx: usize, weights: Vec<f32>) -> Result<()> {
        if expert_idx >= self.num_experts {
            return Err(GrokOzempicError::ExpertOutOfRange {
                index: expert_idx,
                num_experts: self.num_experts,
            });
        }
        if weights.len() != self.embedding_dim {
            return Err(GrokOzempicError::DimensionMismatch {
                expected: self.embedding_dim,
                got: weights.len(),
            });
        }
        self.gate_weights[expert_idx] = weights;
        Ok(())
    }

    /// Route `embedding` through the MoE gate and return `(selected_experts, expert_weights)`.
    ///
    /// Uses a simple dot-product gate followed by top-k selection and softmax normalisation.
    pub fn route(&self, embedding: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        if embedding.len() != self.embedding_dim {
            return Err(GrokOzempicError::DimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }

        // Compute logits: dot product of embedding with each expert's gate vector.
        let logits: Vec<f32> = self
            .gate_weights
            .iter()
            .map(|w| w.iter().zip(embedding.iter()).map(|(a, b)| a * b).sum())
            .collect();

        // Top-k selection (indices sorted by descending logit).
        let mut indexed: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.top_k.min(self.num_experts);
        let selected: Vec<usize> = indexed[..top_k].iter().map(|(i, _)| *i).collect();
        let raw_scores: Vec<f32> = indexed[..top_k].iter().map(|(_, s)| *s).collect();

        // Softmax over the top-k scores.
        let max_score = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw_scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        Ok((selected, weights))
    }
}
