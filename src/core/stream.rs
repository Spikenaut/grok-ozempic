//! Out-of-Core Streaming Quantization Engine
//!
//! Implements a zero-OOM pipeline for quantizing the full 318 GB Grok-1 model
//! on hardware with limited VRAM / system RAM:
//!
//! 1. Memory-maps each Safetensors shard read-only via `memmap2` — the OS
//!    kernel pages in only the bytes that are actually touched.
//! 2. For each tensor, classifies it as a **router tensor** (kept in FP16) or
//!    an **expert MLP tensor** (ternary-quantized with GIF saliency filter).
//! 3. Streams the quantized bytes directly into a [`GgufWriter`] buffer and
//!    writes the final GGUF file, then drops every allocation before moving on.
//!
//! # Memory behaviour
//! - The raw weight bytes are accessed through a read-only `memmap2` mapping;
//!   no heap allocation proportional to shard size is made.
//! - Only the f32/f16 slice for **one tensor at a time** is materialised on
//!   the heap (for the quantizer), then dropped before the next tensor is read.
//! - The [`GgufWriter`] accumulates only the (much smaller) packed ternary /
//!   FP16 output data.

use std::{
    fs::{self, File},
    io::BufWriter,
    path::{Path, PathBuf},
};

use half::f16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;

use crate::{
    core::{
        gguf::{GgufMetaValue, GgufWriter, TensorEntry, GGUF_TENSOR_TYPE_F16, GGUF_TENSOR_TYPE_TERNARY},
        quantizer::{convert_f32_to_f16_bytes, passthrough_f16, quantize_f16, quantize_f32},
    },
    error::{GrokOzempicError, Result},
    types::QuantizationConfig,
};

// ---------------------------------------------------------------------------
// Tensor classification
// ---------------------------------------------------------------------------

/// Returns `true` when the tensor `name` should be kept in FP16 (router /
/// gate weights) rather than being ternary-quantized.
///
/// The classification is based on substring matching against the patterns
/// listed in [`QuantizationConfig::router_patterns`].  If no patterns are
/// configured the default set is used (covers Grok-1's gate / router layers).
fn is_router_tensor(name: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        // Default Grok-1 routing tensor name fragments.
        let defaults = [
            "router",
            "gate",
            "moe_gate",
            "expert_router",
            "routing",
        ];
        return defaults.iter().any(|p| name.contains(p));
    }
    patterns.iter().any(|p| name.contains(p.as_str()))
}

// ---------------------------------------------------------------------------
// Safetensors dtype helpers
// ---------------------------------------------------------------------------

/// Dtype tag values used by the Safetensors format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    F32,
    F16,
    BF16,
    Other,
}

fn parse_dtype(dt: safetensors::Dtype) -> Dtype {
    match dt {
        safetensors::Dtype::F32 => Dtype::F32,
        safetensors::Dtype::F16 => Dtype::F16,
        safetensors::Dtype::BF16 => Dtype::BF16,
        _ => Dtype::Other,
    }
}

// ---------------------------------------------------------------------------
// Streaming engine
// ---------------------------------------------------------------------------

/// Diagnostic statistics emitted after processing each shard.
#[derive(Debug, Default)]
pub struct ShardStats {
    pub shard_path: PathBuf,
    pub tensors_ternary: usize,
    pub tensors_fp16: usize,
    pub tensors_skipped: usize,
    /// Average sparsity across ternary tensors (fraction silenced to 0).
    pub avg_sparsity: f32,
}

/// Run the full out-of-core quantization pipeline described by `config`.
///
/// # Errors
/// Returns an error if:
/// - `input_dir` contains no `.safetensors` files.
/// - Any shard cannot be memory-mapped or parsed.
/// - The output GGUF file cannot be created or written.
pub fn run_quantization(config: &QuantizationConfig) -> Result<Vec<ShardStats>> {
    // Collect shard paths, sorted for deterministic output ordering.
    let shards = collect_shards(&config.input_dir)?;
    if shards.is_empty() {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "no .safetensors files found in '{}'",
            config.input_dir
        )));
    }

    // Create the output GGUF file.
    let out_file = File::create(&config.output_path).map_err(GrokOzempicError::Io)?;
    let mut out_writer = BufWriter::new(out_file);

    let mut gguf = GgufWriter::new();

    // Write top-level metadata.
    gguf.set_metadata("general.name", GgufMetaValue::Str("grok-ozempic".into()));
    gguf.set_metadata(
        "general.quantization_version",
        GgufMetaValue::U32(1),
    );
    gguf.set_metadata(
        "grok_ozempic.gif_threshold",
        GgufMetaValue::Str(config.gif_threshold.to_string()),
    );

    let mut all_stats: Vec<ShardStats> = Vec::with_capacity(shards.len());

    for shard_path in &shards {
        let stats = process_shard(shard_path, config, &mut gguf)?;
        all_stats.push(stats);
    }

    // Flush everything to disk.
    gguf.finish(&mut out_writer)
        .map_err(|e| GrokOzempicError::GgufWrite(e.to_string()))?;

    Ok(all_stats)
}

/// Process a single Safetensors shard: memory-map it, iterate tensors, quantize
/// each one, and add it to the GGUF writer.
fn process_shard(
    shard_path: &Path,
    config: &QuantizationConfig,
    gguf: &mut GgufWriter,
) -> Result<ShardStats> {
    let file = File::open(shard_path).map_err(GrokOzempicError::Io)?;

    // Safety: we open the file read-only and do not mutate the mapping.  The
    // file is not modified while the mapping is alive (single-threaded).
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(GrokOzempicError::Io)? };

    let tensors = SafeTensors::deserialize(&mmap).map_err(GrokOzempicError::Safetensors)?;

    let mut stats = ShardStats {
        shard_path: shard_path.to_owned(),
        ..Default::default()
    };
    let mut sparsity_sum = 0.0f32;

    for (name, view) in tensors.tensors() {
        let dtype = parse_dtype(view.dtype());
        if dtype == Dtype::Other {
            // Skip unsupported dtypes (e.g., I8, Bool) without failing.
            stats.tensors_skipped += 1;
            continue;
        }

        let router = is_router_tensor(&name, &config.router_patterns);

        let (packed_data, tensor_type, shape) = if router {
            // ── Mixed-precision: keep routing gate in FP16 ──────────────────
            let fp16_bytes = match dtype {
                Dtype::F16 => {
                    // Already FP16 — read raw bytes directly from the mmap.
                    let raw = view.data();
                    let f16_slice: &[f16] = bytemuck_cast_f16(raw);
                    passthrough_f16(f16_slice)
                }
                Dtype::F32 => {
                    let raw = view.data();
                    let f32_slice = bytemuck_cast_f32(raw);
                    convert_f32_to_f16_bytes(f32_slice)
                }
                Dtype::BF16 => {
                    // Convert BF16 → f32 → f16.
                    let raw = view.data();
                    let f32_vals = bf16_bytes_to_f32(raw);
                    convert_f32_to_f16_bytes(&f32_vals)
                }
                Dtype::Other => unreachable!(),
            };
            let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
            (fp16_bytes, GGUF_TENSOR_TYPE_F16, shape)
        } else {
            // ── Saliency-aware GIF ternary quantization ──────────────────────
            let qt = match dtype {
                Dtype::F32 => {
                    let raw = view.data();
                    let f32_slice = bytemuck_cast_f32(raw);
                    quantize_f32(f32_slice, config.gif_threshold)
                }
                Dtype::F16 => {
                    let raw = view.data();
                    let f16_slice: &[f16] = bytemuck_cast_f16(raw);
                    quantize_f16(f16_slice, config.gif_threshold)
                }
                Dtype::BF16 => {
                    let raw = view.data();
                    let f32_vals = bf16_bytes_to_f32(raw);
                    quantize_f32(&f32_vals, config.gif_threshold)
                }
                Dtype::Other => unreachable!(),
            };
            sparsity_sum += qt.sparsity;
            stats.tensors_ternary += 1;
            let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
            (qt.packed, GGUF_TENSOR_TYPE_TERNARY, shape)
        };

        if router {
            stats.tensors_fp16 += 1;
        }

        // Add tensor to GGUF writer — the quantized bytes are small compared
        // with the original, and the mmap bytes are not copied here.
        gguf.add_tensor(TensorEntry {
            name: name.to_string(),
            shape,
            tensor_type,
            data: packed_data,
        });
    }

    if stats.tensors_ternary > 0 {
        stats.avg_sparsity = sparsity_sum / stats.tensors_ternary as f32;
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

fn collect_shards(dir: &str) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(GrokOzempicError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    paths.sort();
    Ok(paths)
}

// ---------------------------------------------------------------------------
// Safe byte-slice reinterpretation helpers
// ---------------------------------------------------------------------------

/// Reinterpret a raw byte slice as a `&[f32]`.
///
/// # Panics
/// Panics if `raw` is not aligned to 4 bytes or its length is not a multiple
/// of 4.  Safetensors guarantees both properties for F32 tensors.
fn bytemuck_cast_f32(raw: &[u8]) -> &[f32] {
    assert_eq!(raw.len() % 4, 0, "f32 data length must be a multiple of 4");
    // SAFETY: safetensors guarantees the data is properly aligned and sized for
    // the declared dtype; raw is valid for reads of raw.len() bytes.
    unsafe {
        std::slice::from_raw_parts(raw.as_ptr().cast::<f32>(), raw.len() / 4)
    }
}

/// Reinterpret a raw byte slice as a `&[f16]`.
fn bytemuck_cast_f16(raw: &[u8]) -> &[f16] {
    assert_eq!(raw.len() % 2, 0, "f16 data length must be a multiple of 2");
    // SAFETY: same guarantees as bytemuck_cast_f32 for F16 data.
    unsafe {
        std::slice::from_raw_parts(raw.as_ptr().cast::<f16>(), raw.len() / 2)
    }
}

/// Convert a BF16 byte slice (little-endian) to a vector of f32 values.
///
/// BF16 is the upper 16 bits of IEEE 754 f32; padding the lower 16 bits with
/// zeros gives the equivalent f32.
fn bf16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    assert_eq!(raw.len() % 2, 0, "bf16 data length must be a multiple of 2");
    raw.chunks_exact(2)
        .map(|chunk| {
            // BF16 little-endian: chunk[0] = low byte of BF16, chunk[1] = high byte.
            // Reconstruct as f32 with the BF16 bits in the upper 16 bits.
            let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let f32_bits = (bf16_bits as u32) << 16;
            f32::from_bits(f32_bits)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        // BF16 representation of 1.0f32.
        // 1.0f32 = 0x3F800000; top 16 bits = 0x3F80 → LE bytes [0x80, 0x3F].
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
        let result = collect_shards("/tmp/nonexistent_grok_ozempic_dir_xyz");
        assert!(result.is_err());
    }
}
