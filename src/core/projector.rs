use crate::{
    error::Result,
    types::{HybridConfig, ProjectionMode},
};

/// Projector maps input activations to a ternary spike representation.
///
/// In `SpikingTernary` mode each output value is one of {-1, 0, +1}, driven by
/// a leaky integrate-and-fire (GIF) membrane model.
pub struct Projector {
    pub input_dim: usize,
    pub output_dim: usize,
    pub mode: ProjectionMode,
    pub snn_steps: usize,
    /// Membrane potential for each output neuron (stateful across SNN steps).
    membrane: Vec<f32>,
}

impl Projector {
    /// Create a new `Projector` with the given dimensions and mode.
    pub fn new(input_dim: usize, output_dim: usize, mode: ProjectionMode, snn_steps: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            mode,
            snn_steps,
            membrane: vec![0.0; output_dim],
        }
    }

    /// Build a `Projector` from a [`HybridConfig`].
    pub fn from_config(config: &HybridConfig) -> Self {
        Self::new(
            config.embedding_dim,
            config.embedding_dim,
            config.projection_mode,
            config.snn_steps,
        )
    }

    /// Reset the membrane potentials to zero (call between independent inputs).
    pub fn reset(&mut self) {
        self.membrane.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Project `input` through `snn_steps` of GIF dynamics.
    ///
    /// Returns `(spike_train, final_embedding)` where `spike_train` contains
    /// the indices of neurons that fired in **any** step, and `final_embedding`
    /// is the per-neuron accumulated ternary value over all steps.
    pub fn project(&mut self, input: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        if input.len() != self.input_dim {
            return Err(crate::error::GrokOzempicError::DimensionMismatch {
                expected: self.input_dim,
                got: input.len(),
            });
        }

        let leak: f32 = 0.9;        // membrane leak factor per step
        let threshold: f32 = 1.0;   // firing threshold

        let mut fired_indices: std::collections::BTreeSet<usize> = Default::default();
        let mut embedding = vec![0.0f32; self.output_dim];

        for _ in 0..self.snn_steps {
            for (i, mem) in self.membrane.iter_mut().enumerate() {
                // Integrate: leak + input-driven current
                let current = if i < input.len() { input[i] } else { 0.0 };
                *mem = *mem * leak + current;

                // Fire?
                if *mem >= threshold {
                    fired_indices.insert(i);
                    embedding[i] += 1.0;
                    *mem -= threshold; // reset after spike (soft reset)
                } else if *mem <= -threshold {
                    fired_indices.insert(i);
                    embedding[i] -= 1.0;
                    *mem += threshold;
                }
            }
        }

        Ok((fired_indices.into_iter().collect(), embedding))
    }
}
