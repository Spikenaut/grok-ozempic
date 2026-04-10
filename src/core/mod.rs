pub mod weight_pack;
pub mod weight_pack_read;
pub mod npy;
pub mod ozempic;
pub mod projector;
pub mod quantizer;
pub mod stream;

use crate::{
    error::Result,
    types::{ExecutionMode, HybridConfig, HybridOutput},
};

use ozempic::OzempicMoE;
use projector::Projector;

/// The top-level hybrid model combining SNN projection with sparse MoE routing.
pub struct HybridModel {
    pub config: HybridConfig,
    projector: Projector,
    moe: OzempicMoE,
}

impl HybridModel {
    /// Construct a `HybridModel` from a [`HybridConfig`].
    pub fn from_config(config: HybridConfig) -> Self {
        let projector = Projector::from_config(&config);
        let moe = OzempicMoE::from_config(&config);
        Self { config, projector, moe }
    }

    /// Reset all stateful components (membrane potentials, etc.).
    pub fn reset(&mut self) {
        self.projector.reset();
    }

    /// Run a forward pass: project `input` then route through the MoE gate.
    ///
    /// Returns a [`HybridOutput`] containing the spike train, final embedding,
    /// selected expert indices, and their softmax weights.
    pub fn forward(&mut self, input: &[f32]) -> Result<HybridOutput> {
        let (spike_train_flat, embedding) = match self.config.execution_mode {
            ExecutionMode::SpikingSim => self.projector.project(input)?,
            ExecutionMode::DenseSim => {
                // Dense mode: bypass spiking, pass input directly as embedding.
                let embedding = input.to_vec();
                (vec![], embedding)
            }
        };

        let spike_train = vec![spike_train_flat];

        let (selected_experts, expert_weights) = self.moe.route(&embedding)?;

        Ok(HybridOutput {
            spike_train,
            embedding,
            expert_weights: Some(expert_weights),
            selected_experts: Some(selected_experts),
        })
    }
}
