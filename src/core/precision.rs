//! Precision policy — turn a [`crate::core::selection::TensorClass`] into a
//! concrete [`crate::types::TensorPrecision`] and an effective GIF
//! threshold.
//!
//! Threshold resolution order:
//! 1. Per-tensor `gif_threshold` carried on a `TernaryCandidate`.
//! 2. `manifest.defaults.gif_threshold` if set.
//! 3. `config.gif_threshold` (always present).
//!
//! Precision tier resolution:
//! - `TensorClass::Preserve`          → [`TensorPrecision::Preserve`]
//!   (FP16-at-rest in GOZ1 v1; semantically distinct from `Fp16`,
//!   identical on disk — see the `TensorPrecision::Preserve` docs).
//! - `TensorClass::Fp16`              → [`TensorPrecision::Fp16`]
//! - `TensorClass::TernaryCandidate`  → [`TensorPrecision::TernarySnN`]
//! - `TensorClass::Default`           → parsed from
//!   `manifest.defaults.precision` (if present), else
//!   [`TensorPrecision::TernarySnN`].
//!
//! Unknown `defaults.precision` strings are a **hard failure**
//! ([`GrokOzempicError::ManifestInvalidPrecision`]) to match the rest of
//! the manifest validation story.

use crate::{
    core::{manifest::DissectManifest, selection::TensorClass},
    error::{GrokOzempicError, Result},
    types::{QuantizationConfig, TensorPrecision},
};

/// Decide `(precision, effective_gif_threshold)` for a single tensor.
pub fn decide(
    class: &TensorClass,
    manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> Result<(TensorPrecision, f32)> {
    let precision = match class {
        TensorClass::Preserve { .. } => TensorPrecision::Preserve,
        TensorClass::Fp16 { .. } => TensorPrecision::Fp16,
        TensorClass::TernaryCandidate { .. } => TensorPrecision::TernarySnN,
        TensorClass::Default => resolve_default_precision(manifest)?,
    };
    let threshold = resolve_threshold(class, manifest, config);
    Ok((precision, threshold))
}

fn resolve_default_precision(manifest: Option<&DissectManifest>) -> Result<TensorPrecision> {
    let Some(m) = manifest else {
        return Ok(TensorPrecision::TernarySnN);
    };
    match m.defaults.precision.as_deref() {
        None => Ok(TensorPrecision::TernarySnN),
        Some(s) => parse_precision_str(s),
    }
}

fn resolve_threshold(
    class: &TensorClass,
    manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> f32 {
    if let TensorClass::TernaryCandidate {
        gif_threshold: Some(t),
        ..
    } = class
    {
        return *t;
    }
    if let Some(m) = manifest
        && let Some(t) = m.defaults.gif_threshold
    {
        return t;
    }
    config.gif_threshold
}

/// Parse a manifest precision string into a concrete
/// [`TensorPrecision`] tier.
///
/// Accepted values: `ternary_snn`, `fp16`, `preserve`.
pub fn parse_precision_str(s: &str) -> Result<TensorPrecision> {
    match s {
        "ternary_snn" => Ok(TensorPrecision::TernarySnN),
        "fp16" => Ok(TensorPrecision::Fp16),
        "preserve" => Ok(TensorPrecision::Preserve),
        other => Err(GrokOzempicError::ManifestInvalidPrecision {
            got: other.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::manifest::{
        DissectManifest, ManifestDefaults, ManifestModel, MANIFEST_NAME_CONVENTION_V1,
        MANIFEST_SCHEMA_VERSION,
    };

    fn manifest_with_defaults(precision: Option<&str>, gif: Option<f32>) -> DissectManifest {
        DissectManifest {
            schema: "xai-dissect.manifest".into(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            model: ManifestModel {
                family: "grok-1".into(),
                source: "xai-org/grok-1".into(),
                tensor_name_convention: MANIFEST_NAME_CONVENTION_V1.into(),
            },
            produced_by: None,
            defaults: ManifestDefaults {
                precision: precision.map(String::from),
                gif_threshold: gif,
            },
            preserve: vec![],
            fp16: vec![],
            ternary_candidates: vec![],
            blocks: vec![],
        }
    }

    fn config_with_threshold(t: f32) -> QuantizationConfig {
        QuantizationConfig {
            gif_threshold: t,
            ..Default::default()
        }
    }

    #[test]
    fn parse_precision_accepts_known_tiers() {
        assert_eq!(
            parse_precision_str("ternary_snn").unwrap(),
            TensorPrecision::TernarySnN
        );
        assert_eq!(
            parse_precision_str("fp16").unwrap(),
            TensorPrecision::Fp16
        );
        assert_eq!(
            parse_precision_str("preserve").unwrap(),
            TensorPrecision::Preserve
        );
    }

    #[test]
    fn parse_precision_hard_fails_on_unknown() {
        match parse_precision_str("mystery_tier") {
            Err(GrokOzempicError::ManifestInvalidPrecision { got }) => {
                assert_eq!(got, "mystery_tier");
            }
            other => panic!("expected ManifestInvalidPrecision, got {other:?}"),
        }
    }

    #[test]
    fn preserve_class_maps_to_preserve_precision() {
        let config = config_with_threshold(0.05);
        let cls = TensorClass::Preserve { reason: None };
        let (p, _) = decide(&cls, None, &config).unwrap();
        assert_eq!(p, TensorPrecision::Preserve);
    }

    #[test]
    fn fp16_class_maps_to_fp16_precision() {
        let config = config_with_threshold(0.05);
        let cls = TensorClass::Fp16 { reason: None };
        let (p, _) = decide(&cls, None, &config).unwrap();
        assert_eq!(p, TensorPrecision::Fp16);
    }

    #[test]
    fn ternary_candidate_maps_to_ternary_snn() {
        let config = config_with_threshold(0.05);
        let cls = TensorClass::TernaryCandidate {
            rank: Some(0.9),
            gif_threshold: None,
        };
        let (p, _) = decide(&cls, None, &config).unwrap();
        assert_eq!(p, TensorPrecision::TernarySnN);
    }

    #[test]
    fn candidate_threshold_overrides_manifest_default() {
        let config = config_with_threshold(0.05);
        let m = manifest_with_defaults(None, Some(0.08));
        let cls = TensorClass::TernaryCandidate {
            rank: None,
            gif_threshold: Some(0.03),
        };
        let (_, t) = decide(&cls, Some(&m), &config).unwrap();
        assert_eq!(t, 0.03);
    }

    #[test]
    fn manifest_default_threshold_overrides_config_default() {
        let config = config_with_threshold(0.05);
        let m = manifest_with_defaults(None, Some(0.08));
        let cls = TensorClass::Default;
        let (_, t) = decide(&cls, Some(&m), &config).unwrap();
        assert_eq!(t, 0.08);
    }

    #[test]
    fn config_default_wins_when_no_overrides() {
        let config = config_with_threshold(0.05);
        let cls = TensorClass::Default;
        let (_, t) = decide(&cls, None, &config).unwrap();
        assert_eq!(t, 0.05);
    }

    #[test]
    fn manifest_default_precision_applies_to_default_class() {
        let config = config_with_threshold(0.05);
        let m = manifest_with_defaults(Some("preserve"), None);
        let (p, _) = decide(&TensorClass::Default, Some(&m), &config).unwrap();
        assert_eq!(p, TensorPrecision::Preserve);
    }

    #[test]
    fn default_precision_falls_back_to_ternary_snn_without_manifest() {
        let config = config_with_threshold(0.05);
        let (p, _) = decide(&TensorClass::Default, None, &config).unwrap();
        assert_eq!(p, TensorPrecision::TernarySnN);
    }

    #[test]
    fn unknown_defaults_precision_bubbles_as_typed_error() {
        let config = config_with_threshold(0.05);
        let m = manifest_with_defaults(Some("bogus"), None);
        let err = decide(&TensorClass::Default, Some(&m), &config).unwrap_err();
        assert!(
            matches!(err, GrokOzempicError::ManifestInvalidPrecision { .. }),
            "got {err:?}"
        );
    }
}
