//! Tensor classification — the **single source of truth** for deciding
//! whether a tensor belongs to the `preserve`, `fp16`, `ternary_candidate`,
//! or default bucket.
//!
//! When an `xai-dissect` manifest is supplied, classification is driven by
//! the manifest using segment-anchored glob matching (see
//! `docs/dissect-manifest.md`). When no manifest is present, the module
//! falls back to the legacy substring-router heuristic that was previously
//! embedded in [`crate::core::stream`]; that heuristic lives here now and
//! nowhere else in the codebase.
//!
//! Resolution order inside a manifest:
//! `preserve` > `fp16` > `ternary_candidates` > `Default`.

use crate::core::manifest::{
    DissectManifest, Fp16Entry, PreserveEntry, TernaryCandidate,
};

/// Default legacy router-name substrings used when no manifest is present
/// and [`crate::types::QuantizationConfig::router_patterns`] is empty.
///
/// Kept here (not in [`crate::core::stream`]) so all classification logic
/// lives in one place.
pub const LEGACY_DEFAULT_ROUTER_PATTERNS: &[&str] = &[
    "router",
    "gate",
    "moe_gate",
    "expert_router",
    "routing",
];

/// Outcome of classification for a single tensor.
///
/// The classification is precision-agnostic: [`crate::core::precision::decide`]
/// turns it into a concrete [`crate::types::TensorPrecision`] plus effective
/// GIF threshold.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorClass {
    /// Tensor appeared in the manifest's `preserve` list — must keep its
    /// source precision (routing-critical layers, etc.).
    Preserve { reason: Option<String> },
    /// Tensor appeared in the manifest's `fp16` list **or** matched the
    /// legacy router heuristic when no manifest is present.
    Fp16 { reason: Option<String> },
    /// Tensor appeared in the manifest's `ternary_candidates` list with
    /// optional rank and per-tensor threshold override.
    TernaryCandidate {
        rank: Option<f32>,
        gif_threshold: Option<f32>,
    },
    /// No explicit manifest entry; falls through to manifest
    /// `defaults.precision` or the pipeline's global default.
    Default,
}

/// Classify `name` against the manifest (if any), otherwise fall back to
/// the legacy substring-router heuristic.
///
/// `legacy_patterns` is consulted only when `manifest.is_none()`; if the
/// slice is empty, [`LEGACY_DEFAULT_ROUTER_PATTERNS`] is used.
pub fn classify(
    name: &str,
    manifest: Option<&DissectManifest>,
    legacy_patterns: &[String],
) -> TensorClass {
    if let Some(m) = manifest {
        return classify_from_manifest(name, m);
    }
    classify_legacy(name, legacy_patterns)
}

fn classify_from_manifest(name: &str, manifest: &DissectManifest) -> TensorClass {
    // Precedence: preserve > fp16 > ternary_candidates > default.
    if let Some(entry) = find_match(name, &manifest.preserve, |e: &PreserveEntry| &e.name) {
        return TensorClass::Preserve {
            reason: entry.reason.clone(),
        };
    }
    if let Some(entry) = find_match(name, &manifest.fp16, |e: &Fp16Entry| &e.name) {
        return TensorClass::Fp16 {
            reason: entry.reason.clone(),
        };
    }
    if let Some(entry) = find_match(
        name,
        &manifest.ternary_candidates,
        |e: &TernaryCandidate| &e.name,
    ) {
        return TensorClass::TernaryCandidate {
            rank: entry.rank,
            gif_threshold: entry.gif_threshold,
        };
    }
    TensorClass::Default
}

fn classify_legacy(name: &str, legacy_patterns: &[String]) -> TensorClass {
    let hit = if legacy_patterns.is_empty() {
        LEGACY_DEFAULT_ROUTER_PATTERNS
            .iter()
            .any(|p| name.contains(p))
    } else {
        legacy_patterns.iter().any(|p| name.contains(p.as_str()))
    };
    if hit {
        TensorClass::Fp16 { reason: None }
    } else {
        TensorClass::Default
    }
}

fn find_match<'a, T, F>(name: &str, entries: &'a [T], get_pattern: F) -> Option<&'a T>
where
    F: Fn(&T) -> &String,
{
    entries.iter().find(|e| glob_match(get_pattern(e), name))
}

/// Segment-anchored glob matcher.
///
/// `*` matches **exactly one** dotted segment (the text between two `.`
/// characters, or the text before the first `.` / after the last `.`).
/// Pattern and name must have the same number of segments; each segment
/// must equal `*` or match the literal.
///
/// Examples:
/// - `glob_match("blk.*.attn_router.weight", "blk.0.attn_router.weight")` → `true`
/// - `glob_match("blk.*.attn_router.weight", "blk.0.sub.attn_router.weight")` → `false`
/// - `glob_match("gate", "blk.0.ffn_gate.weight")` → `false`
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let p: Vec<&str> = pattern.split('.').collect();
    let n: Vec<&str> = name.split('.').collect();
    if p.len() != n.len() {
        return false;
    }
    p.iter().zip(n.iter()).all(|(a, b)| *a == "*" || a == b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::manifest::{
        DissectManifest, Fp16Entry, ManifestDefaults, ManifestModel, PreserveEntry,
        TernaryCandidate, MANIFEST_NAME_CONVENTION_V1, MANIFEST_SCHEMA_VERSION,
    };

    fn empty_manifest() -> DissectManifest {
        DissectManifest {
            schema: "xai-dissect.manifest".into(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            model: ManifestModel {
                family: "grok-1".into(),
                source: "xai-org/grok-1".into(),
                tensor_name_convention: MANIFEST_NAME_CONVENTION_V1.into(),
            },
            produced_by: None,
            defaults: ManifestDefaults::default(),
            preserve: vec![],
            fp16: vec![],
            ternary_candidates: vec![],
            blocks: vec![],
        }
    }

    // -------- glob_match --------

    #[test]
    fn glob_exact_match() {
        assert!(glob_match("blk.0.attn_router.weight", "blk.0.attn_router.weight"));
    }

    #[test]
    fn glob_single_segment_wildcard() {
        assert!(glob_match(
            "blk.*.attn_router.weight",
            "blk.0.attn_router.weight"
        ));
        assert!(glob_match(
            "blk.*.attn_router.weight",
            "blk.42.attn_router.weight"
        ));
    }

    #[test]
    fn glob_wildcard_does_not_cross_segments() {
        assert!(!glob_match(
            "blk.*.attn_router.weight",
            "blk.0.sub.attn_router.weight"
        ));
    }

    #[test]
    fn glob_rejects_substring_like_gate_false_positive() {
        // The legacy 'gate' substring heuristic would match this; the
        // glob matcher must not.
        assert!(!glob_match("gate", "blk.0.ffn_gate.weight"));
    }

    #[test]
    fn glob_rejects_different_segment_counts() {
        assert!(!glob_match("a.b.c", "a.b"));
        assert!(!glob_match("a.b", "a.b.c"));
    }

    // -------- classify (manifest path) --------

    #[test]
    fn manifest_precedence_preserve_wins_over_fp16() {
        let mut m = empty_manifest();
        // Both lists contain a matching pattern; preserve must win.
        m.preserve.push(PreserveEntry {
            name: "blk.*.attn_router.weight".into(),
            reason: Some("routing-critical".into()),
        });
        m.fp16.push(Fp16Entry {
            name: "blk.*.attn_router.weight".into(),
            reason: Some("legacy".into()),
        });
        let cls = classify("blk.0.attn_router.weight", Some(&m), &[]);
        assert!(matches!(cls, TensorClass::Preserve { .. }), "got {cls:?}");
    }

    #[test]
    fn manifest_precedence_fp16_wins_over_ternary() {
        let mut m = empty_manifest();
        m.fp16.push(Fp16Entry {
            name: "blk.0.w.weight".into(),
            reason: None,
        });
        m.ternary_candidates.push(TernaryCandidate {
            name: "blk.0.w.weight".into(),
            rank: Some(0.9),
            gif_threshold: None,
        });
        let cls = classify("blk.0.w.weight", Some(&m), &[]);
        assert!(matches!(cls, TensorClass::Fp16 { .. }), "got {cls:?}");
    }

    #[test]
    fn manifest_ternary_candidate_fields_propagate() {
        let mut m = empty_manifest();
        m.ternary_candidates.push(TernaryCandidate {
            name: "blk.0.ffn_up.weight".into(),
            rank: Some(0.98),
            gif_threshold: Some(0.04),
        });
        let cls = classify("blk.0.ffn_up.weight", Some(&m), &[]);
        match cls {
            TensorClass::TernaryCandidate { rank, gif_threshold } => {
                assert_eq!(rank, Some(0.98));
                assert_eq!(gif_threshold, Some(0.04));
            }
            other => panic!("expected TernaryCandidate, got {other:?}"),
        }
    }

    #[test]
    fn manifest_unmatched_name_is_default() {
        let m = empty_manifest();
        let cls = classify("blk.0.ffn_up.weight", Some(&m), &[]);
        assert!(matches!(cls, TensorClass::Default), "got {cls:?}");
    }

    // -------- classify (legacy path) --------

    #[test]
    fn legacy_default_substrings_match_router_names() {
        assert!(matches!(
            classify("blk.0.moe_gate.weight", None, &[]),
            TensorClass::Fp16 { .. }
        ));
        assert!(matches!(
            classify("model.layers.4.expert_router.weight", None, &[]),
            TensorClass::Fp16 { .. }
        ));
        assert!(matches!(
            classify("transformer.h.1.routing.weight", None, &[]),
            TensorClass::Fp16 { .. }
        ));
    }

    #[test]
    fn legacy_default_leaves_ffn_weights_as_default() {
        // Note: with the legacy heuristic, 'ffn_gate' false-matches
        // 'gate'. That's a known weakness — see the manifest path for
        // the fix. This test documents current legacy behavior.
        assert!(matches!(
            classify("blk.0.ffn_down.weight", None, &[]),
            TensorClass::Default
        ));
        assert!(matches!(
            classify("blk.0.ffn_up.weight", None, &[]),
            TensorClass::Default
        ));
    }

    #[test]
    fn legacy_custom_patterns_override_defaults() {
        let patterns = vec!["special_router".to_string()];
        assert!(matches!(
            classify("blk.0.special_router.weight", None, &patterns),
            TensorClass::Fp16 { .. }
        ));
        // Default 'gate' is not consulted when a custom list is given.
        assert!(matches!(
            classify("blk.0.gate.weight", None, &patterns),
            TensorClass::Default
        ));
    }
}
