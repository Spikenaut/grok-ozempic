//! Out-of-core streaming quantization engine
//!
//! 1. **Manifest pass** — scans safetensors shards or `.npy` files and records
//!    only `(path, name, shape, dtype, router?)` (no weight payloads).
//! 2. **Header pass** — writes GGUF metadata + tensor info with placeholder
//!    offsets via [`GgufStreamWriter::begin`].
//! 3. **Data pass** — for each manifest row, memory-maps the source tensor,
//!    quantizes **one tensor at a time**, and streams bytes with
//!    [`GgufStreamWriter::write_tensor_data`] (no accumulation of all tensors).
//!
//! Pickle / pure JAX checkpoints are not read here; convert to safetensors or
//! `.npy` first.

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
        gguf::{
            GgufMetaValue, GgufStreamWriter, TensorHeader, GGUF_TENSOR_TYPE_F16,
            GGUF_TENSOR_TYPE_TERNARY,
        },
        npy::{npy_stem_to_tensor_name, MmapNpy, NpyDtype},
        quantizer::{convert_f32_to_f16_bytes, passthrough_f16, quantize_f16, quantize_f32},
    },
    error::{GrokOzempicError, Result},
    types::{QuantizationConfig, QuantizationInputFormat},
};

// Grok-1 architecture hints for downstream loaders (llama.cpp-style keys).
// Confirm against https://huggingface.co/xai-org/grok-1 `config.json` when updating.
const GROK1_CONTEXT_LENGTH: u32 = 8192;
const GROK1_EMBEDDING_LENGTH: u32 = 6144;
const GROK1_FEED_FORWARD_LENGTH: u32 = 32768;
const GROK1_ATTENTION_HEAD_COUNT: u32 = 48;
const GROK1_ATTENTION_HEAD_COUNT_KV: u32 = 8;
const GROK1_BLOCK_COUNT: u32 = 64;
const GROK1_EXPERT_COUNT: u32 = 256;

/// Append Grok-1-ish architecture metadata so GGUF consumers can size the graph.
pub fn append_grok1_arch_metadata(meta: &mut BTreeMap<String, GgufMetaValue>) {
    meta.insert(
        "general.architecture".into(),
        GgufMetaValue::Str("grok".into()),
    );
    meta.insert(
        "grok.context_length".into(),
        GgufMetaValue::U32(GROK1_CONTEXT_LENGTH),
    );
    meta.insert(
        "grok.embedding_length".into(),
        GgufMetaValue::U32(GROK1_EMBEDDING_LENGTH),
    );
    meta.insert(
        "grok.feed_forward_length".into(),
        GgufMetaValue::U32(GROK1_FEED_FORWARD_LENGTH),
    );
    meta.insert(
        "grok.attention.head_count".into(),
        GgufMetaValue::U32(GROK1_ATTENTION_HEAD_COUNT),
    );
    meta.insert(
        "grok.attention.head_count_kv".into(),
        GgufMetaValue::U32(GROK1_ATTENTION_HEAD_COUNT_KV),
    );
    meta.insert(
        "grok.block_count".into(),
        GgufMetaValue::U32(GROK1_BLOCK_COUNT),
    );
    meta.insert(
        "grok.expert_count".into(),
        GgufMetaValue::U32(GROK1_EXPERT_COUNT),
    );
}

// ---------------------------------------------------------------------------
// Tensor classification
// ---------------------------------------------------------------------------

fn is_router_tensor(name: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        let defaults = [
            "router", "gate", "moe_gate", "expert_router", "routing",
        ];
        return defaults.iter().any(|p| name.contains(p));
    }
    patterns.iter().any(|p| name.contains(p.as_str()))
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
    is_router: bool,
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
    let manifest = match config.input_format {
        QuantizationInputFormat::Safetensors => {
            let shards = collect_safetensor_shards(&config.input_dir)?;
            if shards.is_empty() {
                return Err(GrokOzempicError::InvalidConfig(format!(
                    "no .safetensors files found in '{}'",
                    config.input_dir
                )));
            }
            build_manifest_safetensors(&shards, &config.router_patterns)?
        }
        QuantizationInputFormat::NpyDir => {
            let paths = collect_npy_files(&config.input_dir)?;
            if paths.is_empty() {
                return Err(GrokOzempicError::InvalidConfig(format!(
                    "no .npy files found in '{}'",
                    config.input_dir
                )));
            }
            build_manifest_npy(&paths, &config.router_patterns)?
        }
    };

    if manifest.is_empty() {
        return Err(GrokOzempicError::InvalidConfig(
            "no quantizable tensors found (all skipped as unsupported dtype?)".into(),
        ));
    }

    let headers: Vec<TensorHeader> = manifest
        .iter()
        .map(|e| TensorHeader {
            name: e.tensor_name.clone(),
            shape: e.shape.clone(),
            tensor_type: if e.is_router {
                GGUF_TENSOR_TYPE_F16
            } else {
                GGUF_TENSOR_TYPE_TERNARY
            },
        })
        .collect();

    let mut metadata: BTreeMap<String, GgufMetaValue> = BTreeMap::new();
    metadata.insert(
        "general.name".into(),
        GgufMetaValue::Str("grok-ozempic".into()),
    );
    metadata.insert(
        "general.quantization_version".into(),
        GgufMetaValue::U32(1),
    );
    metadata.insert(
        "grok_ozempic.gif_threshold".into(),
        GgufMetaValue::Str(config.gif_threshold.to_string()),
    );
    append_grok1_arch_metadata(&mut metadata);

    let out_file = File::create(&config.output_path).map_err(GrokOzempicError::Io)?;
    let mut out_writer = BufWriter::new(out_file);

    let mut stream = GgufStreamWriter::begin(&mut out_writer, &metadata, &headers)?;

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
        if entry.is_router {
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
    config: &QuantizationConfig,
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

    if entry.is_router {
        let fp16_bytes = router_fp16_bytes(dtype, view.data())?;
        Ok((fp16_bytes, None))
    } else {
        let qt = match dtype {
            SourceDtype::F32 => {
                let f32_slice = bytemuck_cast_f32(view.data());
                quantize_f32(f32_slice, config.gif_threshold)
            }
            SourceDtype::F16 => {
                let f16_slice: &[f16] = bytemuck_cast_f16(view.data());
                quantize_f16(f16_slice, config.gif_threshold)
            }
            SourceDtype::BF16 => {
                let f32_vals = bf16_bytes_to_f32(view.data());
                quantize_f32(&f32_vals, config.gif_threshold)
            }
            SourceDtype::Other => unreachable!(),
        };
        let sp = qt.sparsity;
        Ok((qt.packed, Some(sp)))
    }
}

fn quantize_npy_entry(
    entry: &ManifestEntry,
    config: &QuantizationConfig,
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

    if entry.is_router {
        let fp16_bytes = router_fp16_bytes(dtype, raw)?;
        Ok((fp16_bytes, None))
    } else {
        let qt = match dtype {
            SourceDtype::F32 => {
                let f32_slice = bytemuck_cast_f32(raw);
                quantize_f32(f32_slice, config.gif_threshold)
            }
            SourceDtype::F16 => {
                let f16_slice: &[f16] = bytemuck_cast_f16(raw);
                quantize_f16(f16_slice, config.gif_threshold)
            }
            SourceDtype::BF16 => {
                let f32_vals = bf16_bytes_to_f32(raw);
                quantize_f32(&f32_vals, config.gif_threshold)
            }
            SourceDtype::Other => unreachable!(),
        };
        let sp = qt.sparsity;
        Ok((qt.packed, Some(sp)))
    }
}

fn router_fp16_bytes(dtype: SourceDtype, raw: &[u8]) -> Result<Vec<u8>> {
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

fn build_manifest_safetensors(
    shards: &[PathBuf],
    patterns: &[String],
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
            let is_router = is_router_tensor(&name, patterns);
            let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
            v.push(ManifestEntry {
                source_path: shard.clone(),
                tensor_name: name,
                shape,
                is_router,
            });
        }
    }
    Ok(v)
}

fn build_manifest_npy(paths: &[PathBuf], patterns: &[String]) -> Result<Vec<ManifestEntry>> {
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
        let is_router = is_router_tensor(&tensor_name, patterns);
        let shape: Vec<u64> = npy.shape().iter().map(|&d| d as u64).collect();
        v.push(ManifestEntry {
            source_path: path.clone(),
            tensor_name,
            shape,
            is_router,
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

    #[test]
    fn router_classification_defaults() {
        assert!(is_router_tensor("blk.0.moe_gate.weight", &[]));
        assert!(is_router_tensor("model.layers.4.expert_router.weight", &[]));
        assert!(is_router_tensor("transformer.h.1.routing.weight", &[]));
        assert!(!is_router_tensor("blk.0.ffn_up.weight", &[]));
        assert!(!is_router_tensor("blk.0.ffn_down.weight", &[]));
    }

    #[test]
    fn router_classification_custom_patterns() {
        let patterns = vec!["special_router".to_string()];
        assert!(is_router_tensor("blk.0.special_router.weight", &patterns));
        assert!(!is_router_tensor("blk.0.gate.weight", &patterns));
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
}
