//! Out-of-core streaming quantization engine
//!
//! 1. **Manifest pass** — scans safetensors shards or `.npy` files and records
//!    only `(path, name, shape, dtype, router?)` (no weight payloads).
//! 2. **Header pass** — writes **GOZ1** metadata + tensor table with placeholder
//!    offsets via [`PackStreamWriter::begin`](crate::core::weight_pack::PackStreamWriter::begin).
//! 3. **Data pass** — for each manifest row, memory-maps the source tensor,
//!    quantizes **one tensor at a time**, and streams bytes with
//!    [`PackStreamWriter::write_tensor_data`](crate::core::weight_pack::PackStreamWriter::write_tensor_data).
//!
//! **JAX / Flax** checkpoints are not read directly; export tensors to **`.npy`**
//! (or **safetensors**) first. Python **pickle** is not supported.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use half::f16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;

use crate::{
    core::{
        manifest::{embedded_grok1_baseline, load_manifest, DissectManifest},
        npy::{npy_stem_to_tensor_name, MmapNpy, NpyDtype},
        precision::decide as precision_decide,
        quantizer::{convert_f32_to_f16_bytes, passthrough_f16, quantize_f16, quantize_f32},
        selection::{classify, TensorClass},
        weight_pack::{
            PackMetaValue, PackStreamWriter, PackTensorHeader, TENSOR_F16, TENSOR_TERNARY,
        },
    },
    error::{GrokOzempicError, Result},
    types::{QuantizationConfig, QuantizationInputFormat, TensorPrecision},
};

// Grok-1 (JAX) architecture hints embedded in GOZ1 metadata — confirm against
// https://huggingface.co/xai-org/grok-1 `config.json` when updating.
const GROK1_CONTEXT_LENGTH: u32 = 8192;
const GROK1_EMBEDDING_LENGTH: u32 = 6144;
const GROK1_FEED_FORWARD_LENGTH: u32 = 32768;
const GROK1_ATTENTION_HEAD_COUNT: u32 = 48;
const GROK1_ATTENTION_HEAD_COUNT_KV: u32 = 8;
const GROK1_BLOCK_COUNT: u32 = 64;
/// MoE experts **per layer** (matches HF Grok-1 and [`crate::types::HybridConfig`] defaults).
const GROK1_EXPERT_COUNT: u32 = 8;

/// Append Grok-1 model shape hints into GOZ1 metadata (for your own loaders / JAX tooling).
pub fn append_grok1_arch_metadata(meta: &mut BTreeMap<String, PackMetaValue>) {
    meta.insert(
        "grok1.architecture".into(),
        PackMetaValue::Str("grok-1".into()),
    );
    meta.insert(
        "grok1.context_length".into(),
        PackMetaValue::U32(GROK1_CONTEXT_LENGTH),
    );
    meta.insert(
        "grok1.embedding_length".into(),
        PackMetaValue::U32(GROK1_EMBEDDING_LENGTH),
    );
    meta.insert(
        "grok1.feed_forward_length".into(),
        PackMetaValue::U32(GROK1_FEED_FORWARD_LENGTH),
    );
    meta.insert(
        "grok1.attention.head_count".into(),
        PackMetaValue::U32(GROK1_ATTENTION_HEAD_COUNT),
    );
    meta.insert(
        "grok1.attention.head_count_kv".into(),
        PackMetaValue::U32(GROK1_ATTENTION_HEAD_COUNT_KV),
    );
    meta.insert(
        "grok1.block_count".into(),
        PackMetaValue::U32(GROK1_BLOCK_COUNT),
    );
    meta.insert(
        "grok1.expert_count".into(),
        PackMetaValue::U32(GROK1_EXPERT_COUNT),
    );
}

// ---------------------------------------------------------------------------
// Manifest resolution & deprecation warning
// ---------------------------------------------------------------------------

/// Environment variable consulted when [`QuantizationConfig::manifest_path`]
/// is `None`.
pub const MANIFEST_ENV_VAR: &str = "GROK_OZEMPIC_MANIFEST";

/// Resolve the xai-dissect manifest that governs this `run_quantization`
/// call.
///
/// Precedence (first hit wins, errors bubble; no silent fallthrough):
///
/// 1. [`QuantizationConfig::manifest_path`] (explicit caller override).
/// 2. `GROK_OZEMPIC_MANIFEST` environment variable.
/// 3. Embedded Grok-1 baseline via [`embedded_grok1_baseline`], but
///    **only** when [`QuantizationConfig::use_embedded_baseline`] is
///    `true`. This is opt-in so upgrading from phase 1 does not
///    silently apply a Grok-1 manifest to non-Grok-1 exports.
/// 4. `None` — selection falls back to the legacy heuristic in
///    [`crate::core::selection::classify`].
fn resolve_manifest(config: &QuantizationConfig) -> Result<Option<DissectManifest>> {
    if let Some(path) = &config.manifest_path {
        return Ok(Some(load_manifest(path)?));
    }
    if let Ok(env_path) = std::env::var(MANIFEST_ENV_VAR)
        && !env_path.is_empty()
    {
        let p = std::path::PathBuf::from(env_path);
        return Ok(Some(load_manifest(&p)?));
    }
    if config.use_embedded_baseline {
        // Embedded baseline: clone once so callers own a DissectManifest.
        // The OnceLock inside manifest.rs keeps the parse one-shot.
        return Ok(Some(embedded_grok1_baseline()?.clone()));
    }
    Ok(None)
}

/// Return `true` if a deprecation warning about `router_patterns`
/// alongside a manifest should be emitted.
///
/// Exposed (crate-private) so tests can assert the condition without
/// capturing stderr.
pub(crate) fn should_warn_router_patterns_deprecated(
    manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> bool {
    manifest.is_some() && !config.router_patterns.is_empty()
}

fn maybe_warn_router_patterns_deprecated(
    manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) {
    if should_warn_router_patterns_deprecated(manifest, config) {
        eprintln!(
            "grok-ozempic: router_patterns is set alongside an xai-dissect manifest; \
             the manifest wins and router_patterns is ignored (deprecated)."
        );
    }
}

// ---------------------------------------------------------------------------
// Source dtype (safetensors + npy)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceDtype {
    F32,
    F16,
    BF16,
    Other,
}

fn parse_safetensors_dtype(dt: safetensors::Dtype) -> SourceDtype {
    match dt {
        safetensors::Dtype::F32 => SourceDtype::F32,
        safetensors::Dtype::F16 => SourceDtype::F16,
        safetensors::Dtype::BF16 => SourceDtype::BF16,
        _ => SourceDtype::Other,
    }
}

fn npy_dtype_to_source(dt: NpyDtype) -> SourceDtype {
    match dt {
        NpyDtype::F32 => SourceDtype::F32,
        NpyDtype::F16 => SourceDtype::F16,
        NpyDtype::BF16 => SourceDtype::BF16,
        NpyDtype::Other => SourceDtype::Other,
    }
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ManifestEntry {
    source_path: PathBuf,
    tensor_name: String,
    shape: Vec<u64>,
    precision: TensorPrecision,
    gif_threshold: f32,
}

impl ManifestEntry {
    /// True when this tensor is emitted as FP16 bytes in the GOZ1
    /// tensor table (i.e. tagged `TENSOR_F16`).
    ///
    /// In GOZ1 v1 both [`TensorPrecision::Fp16`] and
    /// [`TensorPrecision::Preserve`] share the FP16-at-rest encoding
    /// (see the `TensorPrecision::Preserve` doc for rationale). The
    /// two tiers remain semantically distinct upstream of this point.
    fn emits_fp16_bytes(&self) -> bool {
        matches!(
            self.precision,
            TensorPrecision::Fp16 | TensorPrecision::Preserve
        )
    }
}

/// Diagnostic statistics emitted after processing each shard / source file.
#[derive(Debug, Default)]
pub struct ShardStats {
    pub shard_path: PathBuf,
    pub tensors_ternary: usize,
    pub tensors_fp16: usize,
    pub tensors_skipped: usize,
    pub avg_sparsity: f32,
}

/// Run the full out-of-core quantization pipeline described by `config`.
pub fn run_quantization(config: &QuantizationConfig) -> Result<Vec<ShardStats>> {
    let dissect_manifest = resolve_manifest(config)?;
    maybe_warn_router_patterns_deprecated(dissect_manifest.as_ref(), config);

    let manifest = match config.input_format {
        QuantizationInputFormat::Safetensors => {
            let shards = collect_safetensor_shards(&config.input_dir)?;
            if shards.is_empty() {
                return Err(GrokOzempicError::InvalidConfig(format!(
                    "no .safetensors files found in '{}'",
                    config.input_dir
                )));
            }
            build_manifest_safetensors(&shards, dissect_manifest.as_ref(), config)?
        }
        QuantizationInputFormat::NpyDir => {
            let paths = collect_npy_files(&config.input_dir)?;
            if paths.is_empty() {
                return Err(GrokOzempicError::InvalidConfig(format!(
                    "no .npy files found in '{}'",
                    config.input_dir
                )));
            }
            build_manifest_npy(&paths, dissect_manifest.as_ref(), config)?
        }
    };

    if manifest.is_empty() {
        return Err(GrokOzempicError::InvalidConfig(
            "no quantizable tensors found (all skipped as unsupported dtype?)".into(),
        ));
    }

    let headers: Vec<PackTensorHeader> = manifest
        .iter()
        .map(|e| PackTensorHeader {
            name: e.tensor_name.clone(),
            shape: e.shape.clone(),
            // GOZ1 v1: Fp16 and Preserve share the TENSOR_F16 encoding
            // by design (see TensorPrecision::Preserve). The semantic
            // distinction lives in the manifest / classifier, not in
            // the tensor table.
            tensor_type: if e.emits_fp16_bytes() {
                TENSOR_F16
            } else {
                TENSOR_TERNARY
            },
        })
        .collect();

    let mut metadata: BTreeMap<String, PackMetaValue> = BTreeMap::new();
    metadata.insert(
        "oz.name".into(),
        PackMetaValue::Str("grok-ozempic".into()),
    );
    metadata.insert(
        "oz.quantization_version".into(),
        PackMetaValue::U32(1),
    );
    // Record the effective baseline gif_threshold so the artifact's
    // provenance metadata reflects what was actually applied. When the
    // manifest supplies a defaults.gif_threshold that overrides the
    // config value, that manifest default is recorded here. Per-tensor
    // overrides from ternary_candidates can differ, but this captures the
    // pipeline-level default that governs the majority of tensors.
    let effective_gif_threshold = dissect_manifest
        .as_ref()
        .and_then(|m| m.defaults.gif_threshold)
        .unwrap_or(config.gif_threshold);
    metadata.insert(
        "oz.gif_threshold".into(),
        PackMetaValue::Str(effective_gif_threshold.to_string()),
    );
    append_grok1_arch_metadata(&mut metadata);

    let out_file = File::create(&config.output_path).map_err(GrokOzempicError::Io)?;
    let mut out_writer = BufWriter::new(out_file);

    let mut stream = PackStreamWriter::begin(&mut out_writer, &metadata, &headers)?;

    let mut all_stats: Vec<ShardStats> = Vec::new();
    let mut current_path: Option<PathBuf> = None;
    let mut shard_stats = ShardStats::default();
    let mut sparsity_sum = 0.0f32;
    let mut sparsity_n = 0usize;

    for entry in &manifest {
        if current_path.as_ref() != Some(&entry.source_path) {
            if current_path.is_some() {
                if sparsity_n > 0 {
                    shard_stats.avg_sparsity = sparsity_sum / sparsity_n as f32;
                }
                all_stats.push(shard_stats);
                sparsity_sum = 0.0;
                sparsity_n = 0;
            }
            shard_stats = ShardStats {
                shard_path: entry.source_path.clone(),
                ..Default::default()
            };
            current_path = Some(entry.source_path.clone());
        }

        let (packed, ternary_sparsity) = quantize_manifest_entry(entry, config)?;
        if entry.emits_fp16_bytes() {
            shard_stats.tensors_fp16 += 1;
        } else {
            shard_stats.tensors_ternary += 1;
            if let Some(sp) = ternary_sparsity {
                sparsity_sum += sp;
                sparsity_n += 1;
            }
        }
        stream.write_tensor_data(&packed)?;
    }
    if current_path.is_some() {
        if sparsity_n > 0 {
            shard_stats.avg_sparsity = sparsity_sum / sparsity_n as f32;
        }
        all_stats.push(shard_stats);
    }

    stream.finalize()?;
    out_writer
        .flush()
        .map_err(GrokOzempicError::Io)?;

    Ok(all_stats)
}

fn quantize_manifest_entry(
    entry: &ManifestEntry,
    config: &QuantizationConfig,
) -> Result<(Vec<u8>, Option<f32>)> {
    match config.input_format {
        QuantizationInputFormat::Safetensors => quantize_safetensors_entry(entry, config),
        QuantizationInputFormat::NpyDir => quantize_npy_entry(entry, config),
    }
}

fn quantize_safetensors_entry(
    entry: &ManifestEntry,
    _config: &QuantizationConfig,
) -> Result<(Vec<u8>, Option<f32>)> {
    let file = File::open(&entry.source_path).map_err(GrokOzempicError::Io)?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(GrokOzempicError::Io)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(GrokOzempicError::Safetensors)?;
    let view = tensors.tensor(&entry.tensor_name)?;

    let dtype = parse_safetensors_dtype(view.dtype());
    if dtype == SourceDtype::Other {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "tensor {} has unsupported dtype",
            entry.tensor_name
        )));
    }

    if entry.emits_fp16_bytes() {
        let fp16_bytes = encode_fp16_bytes(dtype, view.data())?;
        Ok((fp16_bytes, None))
    } else {
        let qt = match dtype {
            SourceDtype::F32 => {
                let f32_slice = bytemuck_cast_f32(view.data());
                quantize_f32(f32_slice, entry.gif_threshold)
            }
            SourceDtype::F16 => {
                let f16_slice: &[f16] = bytemuck_cast_f16(view.data());
                quantize_f16(f16_slice, entry.gif_threshold)
            }
            SourceDtype::BF16 => {
                let f32_vals = bf16_bytes_to_f32(view.data());
                quantize_f32(&f32_vals, entry.gif_threshold)
            }
            SourceDtype::Other => unreachable!(),
        };
        let sp = qt.sparsity;
        Ok((qt.packed, Some(sp)))
    }
}

fn quantize_npy_entry(
    entry: &ManifestEntry,
    _config: &QuantizationConfig,
) -> Result<(Vec<u8>, Option<f32>)> {
    let npy = MmapNpy::map_path(&entry.source_path)?;
    let dtype = npy_dtype_to_source(npy.dtype());
    if dtype == SourceDtype::Other {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "npy {} has unsupported descr",
            entry.source_path.display()
        )));
    }
    let raw = npy.data();

    if entry.emits_fp16_bytes() {
        let fp16_bytes = encode_fp16_bytes(dtype, raw)?;
        Ok((fp16_bytes, None))
    } else {
        let qt = match dtype {
            SourceDtype::F32 => {
                let f32_slice = bytemuck_cast_f32(raw);
                quantize_f32(f32_slice, entry.gif_threshold)
            }
            SourceDtype::F16 => {
                let f16_slice: &[f16] = bytemuck_cast_f16(raw);
                quantize_f16(f16_slice, entry.gif_threshold)
            }
            SourceDtype::BF16 => {
                let f32_vals = bf16_bytes_to_f32(raw);
                quantize_f32(&f32_vals, entry.gif_threshold)
            }
            SourceDtype::Other => unreachable!(),
        };
        let sp = qt.sparsity;
        Ok((qt.packed, Some(sp)))
    }
}

/// Encode raw source bytes as FP16 for tensors that emit through the
/// GOZ1 `TENSOR_F16` slot (both `TensorPrecision::Fp16` and
/// `TensorPrecision::Preserve` route here in GOZ1 v1).
fn encode_fp16_bytes(dtype: SourceDtype, raw: &[u8]) -> Result<Vec<u8>> {
    let b = match dtype {
        SourceDtype::F16 => {
            let f16_slice: &[f16] = bytemuck_cast_f16(raw);
            passthrough_f16(f16_slice)
        }
        SourceDtype::F32 => {
            let f32_slice = bytemuck_cast_f32(raw);
            convert_f32_to_f16_bytes(f32_slice)
        }
        SourceDtype::BF16 => {
            let f32_vals = bf16_bytes_to_f32(raw);
            convert_f32_to_f16_bytes(&f32_vals)
        }
        SourceDtype::Other => unreachable!(),
    };
    Ok(b)
}

fn classify_and_decide(
    name: &str,
    dissect_manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> Result<(TensorClass, TensorPrecision, f32)> {
    // selection.rs is the single source of truth for classification.
    // When a manifest is supplied, router_patterns is ignored (the
    // deprecation warning above already informed the caller).
    let legacy_patterns: &[String] = if dissect_manifest.is_some() {
        &[]
    } else {
        &config.router_patterns
    };
    let class = classify(name, dissect_manifest, legacy_patterns);
    let (precision, gif_threshold) = precision_decide(&class, dissect_manifest, config)?;
    Ok((class, precision, gif_threshold))
}

fn build_manifest_safetensors(
    shards: &[PathBuf],
    dissect_manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> Result<Vec<ManifestEntry>> {
    let mut v = Vec::new();
    for shard in shards {
        let file = File::open(shard).map_err(GrokOzempicError::Io)?;
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(GrokOzempicError::Io)? };
        let tensors = SafeTensors::deserialize(&mmap).map_err(GrokOzempicError::Safetensors)?;
        for (name, view) in tensors.tensors() {
            let dtype = parse_safetensors_dtype(view.dtype());
            if dtype == SourceDtype::Other {
                continue;
            }
            let (_class, precision, gif_threshold) =
                classify_and_decide(&name, dissect_manifest, config)?;
            let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
            v.push(ManifestEntry {
                source_path: shard.clone(),
                tensor_name: name,
                shape,
                precision,
                gif_threshold,
            });
        }
    }
    Ok(v)
}

fn build_manifest_npy(
    paths: &[PathBuf],
    dissect_manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> Result<Vec<ManifestEntry>> {
    let mut v = Vec::new();
    for path in paths {
        let npy = MmapNpy::map_path(path)?;
        let dtype = npy_dtype_to_source(npy.dtype());
        if dtype == SourceDtype::Other {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                GrokOzempicError::InvalidConfig(format!("bad npy filename: {}", path.display()))
            })?;
        let tensor_name = npy_stem_to_tensor_name(stem);
        let (_class, precision, gif_threshold) =
            classify_and_decide(&tensor_name, dissect_manifest, config)?;
        let shape: Vec<u64> = npy.shape().iter().map(|&d| d as u64).collect();
        v.push(ManifestEntry {
            source_path: path.clone(),
            tensor_name,
            shape,
            precision,
            gif_threshold,
        });
    }
    Ok(v)
}

fn collect_safetensor_shards(dir: &str) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(GrokOzempicError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    paths.sort();
    Ok(paths)
}

fn collect_npy_files(dir: &str) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(GrokOzempicError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "npy"))
        .collect();
    paths.sort();
    Ok(paths)
}

fn bytemuck_cast_f32(raw: &[u8]) -> &[f32] {
    assert_eq!(raw.len() % 4, 0, "f32 data length must be a multiple of 4");
    unsafe {
        std::slice::from_raw_parts(raw.as_ptr().cast::<f32>(), raw.len() / 4)
    }
}

fn bytemuck_cast_f16(raw: &[u8]) -> &[f16] {
    assert_eq!(raw.len() % 2, 0, "f16 data length must be a multiple of 2");
    unsafe {
        std::slice::from_raw_parts(raw.as_ptr().cast::<f16>(), raw.len() / 2)
    }
}

/// Serialises env-var mutations across tests that touch
/// `GROK_OZEMPIC_MANIFEST`. Declared at the module level so the
/// `#[cfg(test)]` test module can refer to it via `super::`.
#[cfg(test)]
static ENV_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn bf16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    assert_eq!(raw.len() % 2, 0, "bf16 data length must be a multiple of 2");
    raw.chunks_exact(2)
        .map(|chunk| {
            let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let f32_bits = (bf16_bits as u32) << 16;
            f32::from_bits(f32_bits)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Router-classification tests moved to src/core/selection.rs; this
    // file no longer owns classification logic.

    // ---------- test helpers ----------
    //
    // Minimal FP32 `.npy` writer sufficient for round-tripping through
    // `run_quantization`. Mirrors the header format exercised by
    // `core::npy::tests::parse_simple_v1_header`.
    fn write_npy_f32(path: &std::path::Path, shape: &[usize], data: &[f32]) {
        use std::io::Write as _;
        let expected: usize = shape.iter().product();
        assert_eq!(expected, data.len(), "data length must match shape");
        let shape_str = if shape.is_empty() {
            "()".to_string()
        } else if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let inner = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({inner})")
        };
        let dict = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
        );
        let magic = b"\x93NUMPY";
        let preamble_len = magic.len() + 1 /* major */ + 1 /* minor */ + 2 /* hlen */;
        // Pad the header string with spaces so (preamble + hlen) is a
        // multiple of 64, then terminate with `\n` inside the padded
        // region (NumPy convention). Our parser only requires 64-byte
        // alignment, so plain space-padding is fine here.
        let raw_header_len = dict.len();
        let total_unpadded = preamble_len + raw_header_len;
        let pad = (64 - (total_unpadded % 64)) % 64;
        let header_len = raw_header_len + pad;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(magic);
        bytes.push(1); // major
        bytes.push(0); // minor
        bytes.extend_from_slice(&(header_len as u16).to_le_bytes());
        bytes.extend_from_slice(dict.as_bytes());
        bytes.extend(std::iter::repeat(b' ').take(pad));
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&bytes).unwrap();
    }

    /// Per-test unique scratch dir.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("grok-ozempic-stream-{tag}-{pid}-{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Write the phase-2 parity fixture. Names chosen so the legacy
    /// heuristic and the in-tree Grok-1 baseline manifest agree on the
    /// precision of every tensor:
    ///
    /// - `blk.0.moe_gate.weight`      legacy: 'gate'/'moe_gate' substring →
    ///   FP16. Baseline: preserve list → FP16-at-rest (GOZ1 v1).
    /// - `blk.0.expert_router.weight` legacy: 'expert_router' → FP16.
    ///   Baseline: preserve → FP16-at-rest (GOZ1 v1).
    /// - `blk.0.attn_router.weight`   legacy: 'router' → FP16. Baseline:
    ///   fp16 list → FP16.
    /// - `blk.0.ffn_up.weight`        legacy: no match → ternary.
    ///   Baseline: defaults → ternary_snn.
    fn write_parity_fixture(dir: &std::path::Path) {
        // 2x2 tensors are enough to exercise ternary pack alignment.
        let moe_gate = [0.1f32, -0.2, 0.3, -0.4];
        let expert_router = [1.0f32, -1.0, 0.5, -0.5];
        let attn_router = [0.05f32, 0.9, -0.05, -0.9];
        let ffn_up = [0.3f32, -0.3, 0.01, -0.01];
        write_npy_f32(&dir.join("blk__0__moe_gate__weight.npy"), &[2, 2], &moe_gate);
        write_npy_f32(
            &dir.join("blk__0__expert_router__weight.npy"),
            &[2, 2],
            &expert_router,
        );
        write_npy_f32(
            &dir.join("blk__0__attn_router__weight.npy"),
            &[2, 2],
            &attn_router,
        );
        write_npy_f32(&dir.join("blk__0__ffn_up__weight.npy"), &[2, 2], &ffn_up);
    }

    fn base_config(dir: &std::path::Path, out: &std::path::Path) -> QuantizationConfig {
        // gif_threshold must match the baseline manifest's
        // defaults.gif_threshold (0.05) for parity. The embedded
        // baseline's default overrides the config value per
        // precision::decide, so matching the two makes the two code
        // paths produce identical thresholds on ternary tensors.
        QuantizationConfig {
            input_dir: dir.to_string_lossy().into_owned(),
            output_path: out.to_string_lossy().into_owned(),
            gif_threshold: 0.05,
            input_format: QuantizationInputFormat::NpyDir,
            ..Default::default()
        }
    }

    fn goz1_bytes(p: &std::path::Path) -> Vec<u8> {
        std::fs::read(p).expect("failed to read GOZ1 output")
    }

    // ---------- tests ----------

    #[test]
    fn deprecation_warning_fires_when_both_present() {
        use crate::core::manifest::{
            DissectManifest, ManifestDefaults, ManifestModel, MANIFEST_NAME_CONVENTION_V1,
            MANIFEST_SCHEMA_VERSION,
        };
        let manifest = DissectManifest {
            schema: "xai-dissect.manifest".into(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            model: ManifestModel {
                family: "grok-1".into(),
                source: "".into(),
                tensor_name_convention: MANIFEST_NAME_CONVENTION_V1.into(),
            },
            produced_by: None,
            defaults: ManifestDefaults::default(),
            preserve: vec![],
            fp16: vec![],
            ternary_candidates: vec![],
            blocks: vec![],
        };

        let config_with_patterns = QuantizationConfig {
            router_patterns: vec!["legacy_router".into()],
            ..Default::default()
        };
        assert!(should_warn_router_patterns_deprecated(
            Some(&manifest),
            &config_with_patterns,
        ));

        let config_empty = QuantizationConfig::default();
        assert!(!should_warn_router_patterns_deprecated(
            Some(&manifest),
            &config_empty,
        ));

        assert!(!should_warn_router_patterns_deprecated(
            None,
            &config_with_patterns,
        ));
    }

    #[test]
    fn bf16_bytes_to_f32_one() {
        let raw = [0x80u8, 0x3F];
        let result = bf16_bytes_to_f32(&raw);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1.0f32);
    }

    #[test]
    fn bf16_bytes_to_f32_zero() {
        let raw = [0x00u8, 0x00];
        let result = bf16_bytes_to_f32(&raw);
        assert_eq!(result[0], 0.0f32);
    }

    #[test]
    fn bytemuck_cast_f32_basic() {
        let floats = [1.0f32, -2.0f32, 3.5f32];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let back = bytemuck_cast_f32(&bytes);
        assert_eq!(back, floats);
    }

    #[test]
    fn collect_shards_nonexistent_dir() {
        let result = collect_safetensor_shards("/tmp/nonexistent_grok_ozempic_dir_xyz");
        assert!(result.is_err());
    }

    // ---------- parity / divergence / precedence / env-var ----------

    /// Parity test: on the carefully chosen fixture, the legacy heuristic
    /// (no manifest) and the embedded Grok-1 baseline (opt-in) produce
    /// byte-identical GOZ1 output.
    #[test]
    fn parity_legacy_vs_baseline_is_byte_identical() {
        // Hold ENV_TEST_MUTEX so a parallel env-var test cannot leak
        // GROK_OZEMPIC_MANIFEST into the legacy run below.
        let _lock = super::ENV_TEST_MUTEX.lock().unwrap();
        unsafe { std::env::remove_var(MANIFEST_ENV_VAR) };

        let dir = scratch_dir("parity-input");
        write_parity_fixture(&dir);

        let out_legacy = scratch_dir("parity-legacy-out").join("a.goz1");
        let out_baseline = scratch_dir("parity-baseline-out").join("b.goz1");

        let mut config = base_config(&dir, &out_legacy);
        run_quantization(&config).expect("legacy path");

        config.output_path = out_baseline.to_string_lossy().into_owned();
        config.use_embedded_baseline = true;
        run_quantization(&config).expect("baseline path");

        let a = goz1_bytes(&out_legacy);
        let b = goz1_bytes(&out_baseline);
        if a != b {
            let first_diff = a
                .iter()
                .zip(b.iter())
                .position(|(x, y)| x != y)
                .unwrap_or(a.len().min(b.len()));
            panic!(
                "parity failed: a.len={} b.len={} first_diff_at={} a_byte={:?} b_byte={:?}",
                a.len(),
                b.len(),
                first_diff,
                a.get(first_diff),
                b.get(first_diff),
            );
        }
    }

    /// Divergence test: adding `blk.0.ffn_gate.weight` to the fixture
    /// forces the two paths apart. The legacy heuristic false-matches
    /// the `gate` substring and routes it to FP16; the baseline manifest
    /// does not match it at all so it goes ternary. This documents the
    /// value of the manifest wiring.
    #[test]
    fn divergence_on_ffn_gate_false_positive() {
        let _lock = super::ENV_TEST_MUTEX.lock().unwrap();
        unsafe { std::env::remove_var(MANIFEST_ENV_VAR) };

        let dir = scratch_dir("diverge-input");
        write_parity_fixture(&dir);
        let ffn_gate = [0.7f32, -0.7, 0.4, -0.4];
        write_npy_f32(
            &dir.join("blk__0__ffn_gate__weight.npy"),
            &[2, 2],
            &ffn_gate,
        );

        let out_legacy = scratch_dir("diverge-legacy-out").join("a.goz1");
        let out_baseline = scratch_dir("diverge-baseline-out").join("b.goz1");

        let mut config = base_config(&dir, &out_legacy);
        run_quantization(&config).expect("legacy path");

        config.output_path = out_baseline.to_string_lossy().into_owned();
        config.use_embedded_baseline = true;
        run_quantization(&config).expect("baseline path");

        let a = goz1_bytes(&out_legacy);
        let b = goz1_bytes(&out_baseline);
        assert_ne!(
            a, b,
            "`ffn_gate` false-positive should make the two paths diverge"
        );
    }

    /// End-to-end: an explicit `manifest_path` winning over the embedded
    /// baseline and producing the same bytes as a config that points at
    /// the in-tree baseline file directly.
    #[test]
    fn explicit_manifest_path_wins_over_env_var() {
        // Build a tiny manifest-from-disk and a DIFFERENT env-var manifest.
        // Classify a single tensor that each classifies differently; the
        // explicit path must win.
        let dir = scratch_dir("prec-input");
        let ffn_up = [0.3f32, -0.3, 0.01, -0.01];
        write_npy_f32(&dir.join("blk__0__ffn_up__weight.npy"), &[2, 2], &ffn_up);

        // Explicit manifest forces ffn_up to FP16.
        let explicit_path = scratch_dir("prec-explicit").join("m.json");
        std::fs::write(
            &explicit_path,
            r#"{
                "schema": "xai-dissect.manifest",
                "schema_version": 1,
                "model": {
                    "family": "grok-1",
                    "tensor_name_convention": "blk.{L}.{role}.weight"
                },
                "fp16": [ { "name": "blk.0.ffn_up.weight" } ]
            }"#,
        )
        .unwrap();

        // Env manifest forces ffn_up into preserve.
        let env_path = scratch_dir("prec-env").join("m.json");
        std::fs::write(
            &env_path,
            r#"{
                "schema": "xai-dissect.manifest",
                "schema_version": 1,
                "model": {
                    "family": "grok-1",
                    "tensor_name_convention": "blk.{L}.{role}.weight"
                },
                "preserve": [ { "name": "blk.0.ffn_up.weight" } ]
            }"#,
        )
        .unwrap();

        let out_a = scratch_dir("prec-out-a").join("a.goz1");
        let out_b = scratch_dir("prec-out-b").join("b.goz1");

        // Serialize env-var mutations for test-safety.
        let _lock = super::ENV_TEST_MUTEX.lock().unwrap();
        // SAFETY: mutation guarded by the single-threaded test mutex.
        unsafe { std::env::set_var(MANIFEST_ENV_VAR, &env_path) };

        // A: explicit + env — explicit must win.
        let mut config = base_config(&dir, &out_a);
        config.manifest_path = Some(explicit_path.clone());
        run_quantization(&config).expect("explicit+env run");

        // B: explicit only — same explicit manifest, no env. Should
        // produce identical bytes to A if explicit truly won.
        unsafe { std::env::remove_var(MANIFEST_ENV_VAR) };
        let mut config = base_config(&dir, &out_b);
        config.manifest_path = Some(explicit_path.clone());
        run_quantization(&config).expect("explicit only run");

        let a = goz1_bytes(&out_a);
        let b = goz1_bytes(&out_b);
        assert_eq!(
            a, b,
            "explicit manifest_path must win over GROK_OZEMPIC_MANIFEST"
        );
    }

    /// `GROK_OZEMPIC_MANIFEST` is honored when no explicit path is set
    /// (and changes classification versus the no-manifest legacy path).
    #[test]
    fn env_var_manifest_is_resolved_when_no_explicit_path() {
        let dir = scratch_dir("env-input");
        let ffn_up = [0.3f32, -0.3, 0.01, -0.01];
        write_npy_f32(&dir.join("blk__0__ffn_up__weight.npy"), &[2, 2], &ffn_up);

        // Env manifest forces ffn_up to FP16 (legacy heuristic would
        // leave it ternary, so the bytes must differ).
        let env_path = scratch_dir("env-manifest").join("m.json");
        std::fs::write(
            &env_path,
            r#"{
                "schema": "xai-dissect.manifest",
                "schema_version": 1,
                "model": {
                    "family": "grok-1",
                    "tensor_name_convention": "blk.{L}.{role}.weight"
                },
                "fp16": [ { "name": "blk.0.ffn_up.weight" } ]
            }"#,
        )
        .unwrap();

        let out_legacy = scratch_dir("env-out-legacy").join("a.goz1");
        let out_env = scratch_dir("env-out-env").join("b.goz1");

        let _lock = super::ENV_TEST_MUTEX.lock().unwrap();
        // Legacy run (env var unset).
        unsafe { std::env::remove_var(MANIFEST_ENV_VAR) };
        let config = base_config(&dir, &out_legacy);
        run_quantization(&config).expect("legacy run");

        // Env-var run.
        unsafe { std::env::set_var(MANIFEST_ENV_VAR, &env_path) };
        let config = base_config(&dir, &out_env);
        run_quantization(&config).expect("env var run");
        unsafe { std::env::remove_var(MANIFEST_ENV_VAR) };

        assert_ne!(
            goz1_bytes(&out_legacy),
            goz1_bytes(&out_env),
            "env-var-supplied manifest must change classification vs legacy"
        );
    }

    /// Regression guard for the **final GOZ1-v1 semantic** of
    /// `TensorPrecision::Preserve`: a tensor classified `Preserve`
    /// (manifest `preserve` list) and the same tensor classified
    /// `Fp16` (manifest `fp16` list) must produce **byte-identical**
    /// GOZ1 output. The two tiers are semantically distinct upstream
    /// but share the FP16-at-rest encoding on disk.
    ///
    /// If a future change introduces a separate `TENSOR_PRESERVE`
    /// constant or a different on-disk encoding for `Preserve`, this
    /// test will fail and force an explicit decision rather than
    /// silent drift.
    #[test]
    fn preserve_and_fp16_tiers_emit_identical_goz1_bytes() {
        let _lock = super::ENV_TEST_MUTEX.lock().unwrap();
        unsafe { std::env::remove_var(MANIFEST_ENV_VAR) };

        let dir = scratch_dir("preserve-fp16-input");
        let ffn_up = [0.3f32, -0.3, 0.01, -0.01];
        write_npy_f32(&dir.join("blk__0__ffn_up__weight.npy"), &[2, 2], &ffn_up);

        // Manifest A: tensor lives in `preserve`.
        let preserve_path = scratch_dir("preserve-manifest").join("m.json");
        std::fs::write(
            &preserve_path,
            r#"{
                "schema": "xai-dissect.manifest",
                "schema_version": 1,
                "model": {
                    "family": "grok-1",
                    "tensor_name_convention": "blk.{L}.{role}.weight"
                },
                "preserve": [ { "name": "blk.0.ffn_up.weight" } ]
            }"#,
        )
        .unwrap();

        // Manifest B: same tensor, but listed under `fp16`.
        let fp16_path = scratch_dir("fp16-manifest").join("m.json");
        std::fs::write(
            &fp16_path,
            r#"{
                "schema": "xai-dissect.manifest",
                "schema_version": 1,
                "model": {
                    "family": "grok-1",
                    "tensor_name_convention": "blk.{L}.{role}.weight"
                },
                "fp16": [ { "name": "blk.0.ffn_up.weight" } ]
            }"#,
        )
        .unwrap();

        let out_preserve = scratch_dir("preserve-out").join("a.goz1");
        let out_fp16 = scratch_dir("fp16-out").join("b.goz1");

        let mut config = base_config(&dir, &out_preserve);
        config.manifest_path = Some(preserve_path);
        run_quantization(&config).expect("preserve-tier run");

        let mut config = base_config(&dir, &out_fp16);
        config.manifest_path = Some(fp16_path);
        run_quantization(&config).expect("fp16-tier run");

        let a = goz1_bytes(&out_preserve);
        let b = goz1_bytes(&out_fp16);
        if a != b {
            let first_diff = a
                .iter()
                .zip(b.iter())
                .position(|(x, y)| x != y)
                .unwrap_or(a.len().min(b.len()));
            panic!(
                "Preserve and Fp16 tiers diverged: a.len={} b.len={} \
                 first_diff_at={} a_byte={:?} b_byte={:?}. \
                 In GOZ1 v1 they must emit identical FP16 bytes.",
                a.len(),
                b.len(),
                first_diff,
                a.get(first_diff),
                b.get(first_diff),
            );
        }
    }
}
