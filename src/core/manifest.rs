//! xai-dissect manifest ingestion (schema v1).
//!
//! `grok-ozempic` consumes JSON manifests produced by the upstream
//! [`xai-dissect`](https://github.com/rmems/xai-dissect) repository to drive
//! tensor selection and per-tensor precision policy. The manifest is the
//! single contract between the two repositories; `grok-ozempic` does **not**
//! depend on `xai-dissect` as a runtime crate.
//!
//! See `docs/dissect-manifest.md` for the full schema description.
//!
//! This module handles **ingestion only**: JSON parsing, schema
//! validation, and caching of the embedded Grok-1 fallback. The
//! pipeline-side wiring (tensor classification and precision policy)
//! lives in [`crate::core::selection`] and [`crate::core::precision`].

use std::{fs, path::Path, sync::OnceLock};

use serde::{Deserialize, Serialize};

use crate::error::{GrokOzempicError, Result};

/// The in-tree, non-authoritative reference manifest for Grok-1 embedded
/// into the binary at build time.
///
/// Used as the last-resort fallback by the manifest precedence chain in
/// [`crate::core::stream`] when neither
/// [`crate::types::QuantizationConfig::manifest_path`] nor the
/// `GROK_OZEMPIC_MANIFEST` environment variable is set. Parsed exactly
/// once per process via [`embedded_grok1_baseline`].
///
/// `xai-dissect` remains the authoritative source of truth; this constant
/// is a bootstrapping convenience.
pub const GROK1_BASELINE_JSON: &str =
    include_str!("../../dissect/grok-1/baseline.json");

/// Schema version understood by this loader.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Tensor name convention accepted in v1 (canonical Grok-1 naming used by
/// [`crate::core::npy::npy_stem_to_tensor_name`]).
pub const MANIFEST_NAME_CONVENTION_V1: &str = "blk.{L}.{role}.weight";

/// Top-level manifest document.
///
/// Unknown top-level fields are tolerated for forward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissectManifest {
    /// Schema family tag, e.g. `"xai-dissect.manifest"`. Informational.
    #[serde(default)]
    pub schema: String,

    /// Integer schema version. Must equal [`MANIFEST_SCHEMA_VERSION`].
    pub schema_version: u32,

    /// Model-identity block. `tensor_name_convention` is validated
    /// strictly by the loader.
    pub model: ManifestModel,

    /// Optional provenance block.
    #[serde(default)]
    pub produced_by: Option<ManifestProducedBy>,

    /// Pipeline-wide defaults applied when a tensor matches none of the
    /// typed sections.
    #[serde(default)]
    pub defaults: ManifestDefaults,

    /// Routing-critical / no-touch tensors. Consumed by the
    /// selection seam (`crate::core::selection::classify`) and mapped
    /// to [`crate::types::TensorPrecision::Preserve`], which in GOZ1
    /// v1 is FP16-at-rest (see that variant's docs).
    #[serde(default)]
    pub preserve: Vec<PreserveEntry>,

    /// Tensors that must be emitted as FP16 passthrough.
    #[serde(default)]
    pub fp16: Vec<Fp16Entry>,

    /// Tensors ranked as candidates for ternary / SAAQ-oriented
    /// compression.
    #[serde(default)]
    pub ternary_candidates: Vec<TernaryCandidate>,

    /// Optional structural metadata (block/expert counts). Advisory in v1.
    #[serde(default)]
    pub blocks: Vec<ManifestBlock>,
}

/// Model identity carried in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestModel {
    pub family: String,
    #[serde(default)]
    pub source: String,
    pub tensor_name_convention: String,
}

/// Provenance block (informational).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestProducedBy {
    #[serde(default)]
    pub tool: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub commit: Option<String>,
}

/// Pipeline-wide defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManifestDefaults {
    /// Default precision tier. Free-form string in the manifest; mapping
    /// into [`crate::types::TensorPrecision`] happens in the selection
    /// seam (phase 2).
    #[serde(default)]
    pub precision: Option<String>,

    /// Default GIF threshold multiplier. Optional.
    #[serde(default)]
    pub gif_threshold: Option<f32>,
}

/// An entry in the `preserve` list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreserveEntry {
    /// Exact tensor name or simple glob where `*` matches **exactly one**
    /// dotted segment. See `docs/dissect-manifest.md` for the full
    /// matching rules.
    pub name: String,
    #[serde(default)]
    pub reason: Option<String>,
}

/// An entry in the `fp16` list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp16Entry {
    pub name: String,
    #[serde(default)]
    pub reason: Option<String>,
}

/// An entry in `ternary_candidates`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryCandidate {
    pub name: String,
    /// Optional rank hint in `[0, 1]`. Higher = stronger candidate for
    /// aggressive ternary compression.
    #[serde(default)]
    pub rank: Option<f32>,
    /// Optional per-tensor override of the global GIF threshold.
    #[serde(default)]
    pub gif_threshold: Option<f32>,
}

/// Advisory block metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestBlock {
    pub index: u32,
    #[serde(default)]
    pub experts: Option<u32>,
    #[serde(default)]
    pub role: Option<String>,
}

/// Load and validate a manifest from disk.
///
/// # Errors
///
/// Returns a typed [`GrokOzempicError`] variant rather than a best-effort
/// parse if:
///
/// - the file cannot be read ([`GrokOzempicError::ManifestIo`]),
/// - the JSON is malformed ([`GrokOzempicError::ManifestParse`]),
/// - `schema_version` is not [`MANIFEST_SCHEMA_VERSION`]
///   ([`GrokOzempicError::ManifestSchemaVersion`]),
/// - `model.tensor_name_convention` is not
///   [`MANIFEST_NAME_CONVENTION_V1`]
///   ([`GrokOzempicError::ManifestNameConventionMismatch`]).
///
/// Unknown top-level fields are tolerated.
pub fn load_manifest(path: &Path) -> Result<DissectManifest> {
    let bytes = fs::read(path).map_err(|e| GrokOzempicError::ManifestIo {
        path: path.display().to_string(),
        source: e,
    })?;
    parse_manifest_bytes(&bytes, &path.display().to_string())
}

/// Parse and validate a manifest from already-loaded bytes.
///
/// Shared validation path used by [`load_manifest`] (filesystem) and
/// [`embedded_grok1_baseline`] (`include_str!`). The `label` is used in
/// error messages to identify the origin of the bytes.
pub fn parse_manifest_bytes(bytes: &[u8], label: &str) -> Result<DissectManifest> {
    let manifest: DissectManifest =
        serde_json::from_slice(bytes).map_err(|e| GrokOzempicError::ManifestParse {
            path: label.to_string(),
            source: e,
        })?;

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(GrokOzempicError::ManifestSchemaVersion {
            got: manifest.schema_version,
            expected: MANIFEST_SCHEMA_VERSION,
        });
    }

    if manifest.model.tensor_name_convention != MANIFEST_NAME_CONVENTION_V1 {
        return Err(GrokOzempicError::ManifestNameConventionMismatch {
            got: manifest.model.tensor_name_convention.clone(),
            expected: MANIFEST_NAME_CONVENTION_V1.to_string(),
        });
    }

    // Validate defaults.precision eagerly so a malformed manifest always
    // hard-fails regardless of which tensors are actually processed. Without
    // this check an invalid string would only surface when a tensor falls
    // through to the Default class — a coverage-dependent, non-deterministic
    // failure mode.
    if let Some(ref p) = manifest.defaults.precision {
        match p.as_str() {
            "ternary_snn" | "fp16" | "preserve" => {}
            other => {
                return Err(GrokOzempicError::ManifestInvalidPrecision {
                    got: other.to_string(),
                });
            }
        }
    }

    Ok(manifest)
}

/// Return the embedded Grok-1 reference manifest, parsed lazily on first
/// call and cached for the remainder of the process.
///
/// The underlying JSON is compiled into the binary from
/// `dissect/grok-1/baseline.json` via [`GROK1_BASELINE_JSON`]. If parsing
/// ever fails (which would require the CI guard to regress), the error is
/// returned on every call; parsing is not retried.
pub fn embedded_grok1_baseline() -> Result<&'static DissectManifest> {
    static CACHE: OnceLock<std::result::Result<DissectManifest, GrokOzempicError>> =
        OnceLock::new();
    match CACHE.get_or_init(|| {
        parse_manifest_bytes(
            GROK1_BASELINE_JSON.as_bytes(),
            "<embedded grok-1 baseline>",
        )
    }) {
        Ok(m) => Ok(m),
        Err(e) => Err(clone_typed_error(e)),
    }
}

/// Clone a [`GrokOzempicError`] for cases where we hand shared refs out
/// of a [`OnceLock`] but still want owned typed errors on the caller side.
///
/// Only the subset of variants that [`parse_manifest_bytes`] can produce
/// needs real logic; everything else falls through to a best-effort
/// stringified [`GrokOzempicError::InvalidConfig`].
fn clone_typed_error(e: &GrokOzempicError) -> GrokOzempicError {
    match e {
        GrokOzempicError::ManifestParse { path, source: _ } => {
            GrokOzempicError::InvalidConfig(format!(
                "embedded grok-1 baseline parse error at {path}"
            ))
        }
        GrokOzempicError::ManifestSchemaVersion { got, expected } => {
            GrokOzempicError::ManifestSchemaVersion {
                got: *got,
                expected: *expected,
            }
        }
        GrokOzempicError::ManifestNameConventionMismatch { got, expected } => {
            GrokOzempicError::ManifestNameConventionMismatch {
                got: got.clone(),
                expected: expected.clone(),
            }
        }
        other => GrokOzempicError::InvalidConfig(format!(
            "embedded grok-1 baseline failed to load: {other}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, contents: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("grok-ozempic-manifest-test-{name}.json"));
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path
    }

    fn valid_manifest_json() -> &'static str {
        r#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 1,
            "model": {
                "family": "grok-1",
                "source": "xai-org/grok-1",
                "tensor_name_convention": "blk.{L}.{role}.weight"
            },
            "produced_by": {
                "tool": "xai-dissect",
                "version": "0.1.0"
            },
            "defaults": {
                "precision": "ternary_snn",
                "gif_threshold": 0.05
            },
            "preserve": [
                { "name": "blk.*.attn_router.weight", "reason": "routing-critical" }
            ],
            "fp16": [
                { "name": "token_embd.weight" }
            ],
            "ternary_candidates": [
                { "name": "blk.0.ffn_up.weight", "rank": 0.98, "gif_threshold": 0.04 }
            ],
            "blocks": [
                { "index": 0, "experts": 8, "role": "moe" }
            ]
        }"#
    }

    #[test]
    fn loads_valid_v1_manifest() {
        let path = write_tmp("valid", valid_manifest_json());
        let manifest = load_manifest(&path).expect("valid manifest should load");
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.model.family, "grok-1");
        assert_eq!(manifest.preserve.len(), 1);
        assert_eq!(manifest.fp16.len(), 1);
        assert_eq!(manifest.ternary_candidates.len(), 1);
        assert_eq!(manifest.blocks.len(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_manifest_io_error() {
        let mut path = std::env::temp_dir();
        path.push("grok-ozempic-manifest-test-does-not-exist.json");
        let _ = fs::remove_file(&path);
        let err = load_manifest(&path).unwrap_err();
        assert!(
            matches!(err, GrokOzempicError::ManifestIo { .. }),
            "expected ManifestIo, got {err:?}"
        );
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let json = r#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 2,
            "model": {
                "family": "grok-1",
                "tensor_name_convention": "blk.{L}.{role}.weight"
            }
        }"#;
        let path = write_tmp("version", json);
        let err = load_manifest(&path).unwrap_err();
        assert!(
            matches!(
                err,
                GrokOzempicError::ManifestSchemaVersion { got: 2, expected: 1 }
            ),
            "expected ManifestSchemaVersion, got {err:?}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn mismatched_name_convention_hard_fails() {
        let json = r#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 1,
            "model": {
                "family": "grok-1",
                "tensor_name_convention": "model.layers.{L}.weight"
            }
        }"#;
        let path = write_tmp("convention", json);
        let err = load_manifest(&path).unwrap_err();
        assert!(
            matches!(
                err,
                GrokOzempicError::ManifestNameConventionMismatch { .. }
            ),
            "expected ManifestNameConventionMismatch, got {err:?}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn unknown_top_level_field_is_tolerated() {
        let json = r#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 1,
            "future_field": { "hello": "world" },
            "model": {
                "family": "grok-1",
                "tensor_name_convention": "blk.{L}.{role}.weight"
            }
        }"#;
        let path = write_tmp("unknown", json);
        let manifest = load_manifest(&path).expect("unknown fields should be tolerated");
        assert_eq!(manifest.schema_version, 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn in_tree_grok1_baseline_loads() {
        // Guard: the committed reference manifest must always parse against
        // the current loader. It is non-authoritative (xai-dissect is the
        // source of truth) but it is the crate's own bootstrapping example.
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("dissect/grok-1/baseline.json");
        let manifest = load_manifest(&path).expect("baseline must load");
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.model.family, "grok-1");
        assert_eq!(
            manifest.model.tensor_name_convention,
            MANIFEST_NAME_CONVENTION_V1
        );
    }

    #[test]
    fn embedded_grok1_baseline_loads_and_caches() {
        // Ensures include_str!-embedded baseline parses via the shared
        // parse_manifest_bytes path and that OnceLock caches the result.
        let a = embedded_grok1_baseline().expect("embedded baseline must load");
        let b = embedded_grok1_baseline().expect("embedded baseline must load (2nd call)");
        assert!(std::ptr::eq(a, b), "OnceLock must return the same ref");
        assert_eq!(a.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(a.model.family, "grok-1");
    }

    #[test]
    fn parse_manifest_bytes_rejects_bad_version() {
        let json = br#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 99,
            "model": {
                "family": "grok-1",
                "tensor_name_convention": "blk.{L}.{role}.weight"
            }
        }"#;
        let err = parse_manifest_bytes(json, "<test>").unwrap_err();
        assert!(
            matches!(err, GrokOzempicError::ManifestSchemaVersion { got: 99, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn malformed_json_is_parse_error() {
        let path = write_tmp("malformed", "{ not valid json");
        let err = load_manifest(&path).unwrap_err();
        assert!(
            matches!(err, GrokOzempicError::ManifestParse { .. }),
            "expected ManifestParse, got {err:?}"
        );
        let _ = fs::remove_file(&path);
    }

    /// Regression guard for the eager defaults.precision validation:
    /// a manifest with an invalid precision string must always hard-fail
    /// at parse time regardless of which tensors are present.
    #[test]
    fn invalid_defaults_precision_is_rejected_at_parse_time() {
        let json = br#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 1,
            "model": {
                "family": "grok-1",
                "tensor_name_convention": "blk.{L}.{role}.weight"
            },
            "defaults": { "precision": "bogus_tier" },
            "preserve": [ { "name": "blk.*.*.weight" } ]
        }"#;
        // All tensors would be covered by the preserve glob, but the
        // invalid precision string must still fail at parse time.
        let err = parse_manifest_bytes(json, "<test>").unwrap_err();
        assert!(
            matches!(
                err,
                GrokOzempicError::ManifestInvalidPrecision { ref got } if got == "bogus_tier"
            ),
            "expected ManifestInvalidPrecision(bogus_tier), got {err:?}"
        );
    }
}
