# grok-ozempic

**SNN-inspired ternary quantization for Grok-scale MoE weights.** Keep routing sharp in FP16, pack the rest into 2-bit `{-1, 0, +1}` using a saliency (GIF-style) threshold. Think *Ozempic for Grok* — less bulk, same routing story.

- **Authoritative classification** comes from an `xai-dissect` schema-v1 manifest (segment-anchored globs, three precision tiers, per-tensor GIF-threshold overrides). The legacy substring heuristic is preserved as a fallback when no manifest is supplied. See [`docs/dissect-manifest.md`](docs/dissect-manifest.md).
- **Output format is GOZ1** — our own packed checkpoint (magic `GOZ1`, see [`src/core/weight_pack.rs`](src/core/weight_pack.rs)). **Not GGUF.**
- **Batch pass is out-of-core** — one tensor in RAM at a time, streamed via memory-mapped I/O.
- **[Grok-1](https://huggingface.co/xai-org/grok-1)** ships as JAX / Flax weights; this crate does **not** load checkpoints directly. Export tensors to `.npy` (recommended for JAX) or `safetensors` first. **Pickle is not supported.**

---

## How it works

For each tensor, the pipeline decides between three tiers:

| Tier          | Meaning                                                                                                                                                                                                                   | GOZ1 v1 on-disk              |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `preserve`    | Routing-critical / no-touch (MoE routers, expert gates, attention readouts). Kept distinct from `fp16` for manifest-intent traceability and forward compatibility with a future format version that may do source-dtype passthrough. | FP16-at-rest (`TENSOR_F16`)  |
| `fp16`        | Explicit FP16 passthrough tier.                                                                                                                                                                                           | FP16 (`TENSOR_F16`)          |
| `ternary_snn` | Saliency-gated 2-bit ternary. For each weight: `τ = gif_threshold × rms(layer)`; `|w| < τ → 0`, else `±1`. Four trits pack into one byte.                                                                                 | 2-bit ternary (`TENSOR_TERNARY`) |

The three-pass streaming pipeline lives in [`src/core/stream.rs`](src/core/stream.rs):

1. **Manifest pass** — scan shards / `.npy` files, record `(path, name, shape, dtype, precision)` only. No weight payloads loaded.
2. **Header pass** — write the GOZ1 metadata + tensor table with placeholder offsets ([`PackStreamWriter::begin`](src/core/weight_pack.rs)).
3. **Data pass** — for each manifest row, memory-map the source tensor, quantize or pass through, stream bytes with [`PackStreamWriter::write_tensor_data`](src/core/weight_pack.rs). Offsets are patched in `finalize`.

## xai-dissect manifest integration

Upstream `xai-dissect` is the authoritative source of truth for tensor classification. `grok-ozempic` consumes the JSON contract but **never depends on `xai-dissect` as a Cargo crate** and never writes manifests.

Resolution order inside `run_quantization` (first hit wins; errors bubble, no silent fallthrough):

1. `QuantizationConfig.manifest_path` — explicit caller override.
2. `GROK_OZEMPIC_MANIFEST` environment variable.
3. Embedded Grok-1 baseline compiled from [`dissect/grok-1/baseline.json`](dissect/grok-1/baseline.json), **only when** `QuantizationConfig.use_embedded_baseline = true` (opt-in; default is `false` so upgrading preserves legacy behavior).
4. Legacy `router_patterns` substring heuristic in [`src/core/selection.rs`](src/core/selection.rs).

When both a manifest and `router_patterns` are provided, the manifest wins and a one-line deprecation warning is logged.

Manifest resolution inside a single file: `preserve` > `fp16` > `ternary_candidates` > `defaults`. Name matching uses segment-anchored globs where `*` matches **exactly one dotted segment** — for example, `blk.*.attn_router.weight` matches `blk.0.attn_router.weight` but not `blk.0.sub.attn_router.weight`. Unknown `defaults.precision` strings hard-fail at parse time.

## Inputs

- **Safetensors** — memory-mapped `*.safetensors` shards (`QuantizationInputFormat::Safetensors`).
- **Flat `.npy`** — one tensor per file (`QuantizationInputFormat::NpyDir`). Filename mapping: `blk__0__weight.npy` → `blk.0.weight`. **Fortran-order `.npy` is rejected.**

## GOZ1 format

Little-endian container defined in [`src/core/weight_pack.rs`](src/core/weight_pack.rs):

```
GOZ1 magic (u32) | version (u32) | tensor_count (u64) | meta_count (u64)
  metadata KV pairs (U32 or Str)
  tensor table: name, shape, tensor_type, offset
  32-byte alignment
  tensor payloads, each 32-byte-aligned
```

Verify layout and offsets with [`verify_pack_file`](src/core/weight_pack_read.rs).

## Hybrid runtime

Optional inference stack in [`src/core/mod.rs`](src/core/mod.rs):

- `HybridModel` — combines an SNN `Projector` with a sparse MoE `OzempicMoE` gate.
- Default hidden size **6144** (`GROK1_HIDDEN_DIM`).
- Load router gates from a stacked-expert FP16 tensor via `OzempicMoE::load_gates_from_fp16_stacked_experts`.

## Build and test

```bash
cargo build --release
cargo test
```

Rust **2024 edition**. A trailing `[workspace]` in `Cargo.toml` keeps this crate standalone when nested under a parent workspace.

## Usage sketch

```rust
use grok_ozempic::{
    core::stream::run_quantization,
    core::weight_pack_read::verify_pack_file,
    types::{QuantizationConfig, QuantizationInputFormat},
};

let config = QuantizationConfig {
    input_dir: "./grok1-npy".into(),
    output_path: "./grok1.goz1".into(),
    input_format: QuantizationInputFormat::NpyDir,
    gif_threshold: 0.05,
    // Option 1: explicit manifest path.
    manifest_path: Some("./grok1-dissect.json".into()),
    // Option 2: opt in to the in-tree baseline when no manifest is supplied.
    use_embedded_baseline: false,
    ..Default::default()
};

run_quantization(&config)?;
verify_pack_file(std::path::Path::new(&config.output_path))?;
```

Alternatively, export `GROK_OZEMPIC_MANIFEST=/path/to/manifest.json` to have every `run_quantization` call pick up a manifest without code changes.

## Project layout

| Module                                             | Responsibility                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| [`src/core/stream.rs`](src/core/stream.rs)         | Three-pass quantization pipeline, manifest resolution, deprecation warning. |
| [`src/core/selection.rs`](src/core/selection.rs)   | `TensorClass` + `classify()`; segment-anchored glob matching; single source of truth for classification. |
| [`src/core/precision.rs`](src/core/precision.rs)   | `decide()` — maps `TensorClass` + manifest + config to `(TensorPrecision, gif_threshold)`. |
| [`src/core/manifest.rs`](src/core/manifest.rs)     | xai-dissect manifest loader, schema-v1 validation, embedded Grok-1 baseline (`OnceLock`-cached). |
| [`src/core/quantizer.rs`](src/core/quantizer.rs)   | GIF-threshold ternary quantizer, FP16 passthrough, trit packing.         |
| [`src/core/weight_pack.rs`](src/core/weight_pack.rs) | GOZ1 streaming writer.                                                   |
| [`src/core/weight_pack_read.rs`](src/core/weight_pack_read.rs) | GOZ1 layout verifier.                                                    |
| [`src/core/npy.rs`](src/core/npy.rs)               | Minimal `.npy` header parser with mmap access.                           |
| [`src/core/mod.rs`](src/core/mod.rs) + `ozempic.rs` + `projector.rs` | Hybrid SNN runtime (`HybridModel`, `OzempicMoE`, `Projector`).           |

## Repository

**https://github.com/Spikenaut/grok-ozempic**

## License

GPL-3.0-or-later (see `Cargo.toml`).
