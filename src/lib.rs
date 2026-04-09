//! grok-ozempic — SNN-logic quantization for Grok models
//!
//! Turns Grok's massive MoE into a sparse, membrane-driven, ternary spiking system.
//! "Ozempic for Grok" — we keep the intelligence, drop the fat.

pub mod types;
pub mod core;
pub mod error;

pub use types::{HybridConfig, HybridOutput, TelemetrySnapshot};
pub use core::HybridModel;

// Re-export main types for convenience
pub use crate::core::projector::Projector;
pub use crate::core::olmoe::OLMoE;
