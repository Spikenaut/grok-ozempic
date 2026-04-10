# grok-ozempic

SNN-inspired **ternary quantization** for Grok-scale MoE weights: keep routing sharp in **FP16**, pack the rest into **2-bit {-1, 0, +1}** using a saliency (GIF-style) threshold. Think “Ozempic for Grok” — less bulk, same routing story.

**[Grok-1](https://huggingface.co/xai-org/grok-1)** ships as **JAX / Flax** weights; this crate does **not** load checkpoints directly. Export tensors to **`.npy`** (recommended for JAX) or **safetensors**, then run the batch pipeline. **Pickle** is not supported.

The batch pass is **out-of-core** (one tensor at a time in RAM). Output is our own **GOZ1** packed file (magic `GOZ1`, see [`src/core/weight_pack.rs`](src/core/weight_pack.rs)) — **not GGUF**.

## Features

- **GOZ1 writer** — streaming packed checkpoint: metadata + tensor table + payloads (`PackStreamWriter` in `weight_pack`).
- **Verifier** — `verify_pack_file` (`weight_pack_read`) checks layout and offsets.
- **Inputs:** mmap **`*.safetensors`** or flat **`.npy`** (`QuantizationInputFormat::NpyDir`). Filename mapping: `blk__0__weight.npy` → `blk.0.weight`. **Fortran-order `.npy` is rejected.**
- **Mixed precision:** gate/router tensors stay **FP16**; others **ternary**.
- **Hybrid runtime:** `HybridModel`, `Projector`, `OzempicMoE`; default hidden **6144** (`GROK1_HIDDEN_DIM`). Load gates via `OzempicMoE::load_gates_from_fp16_stacked_experts`.

## Build

```bash
cargo build --release
cargo test
```

Rust **2024** edition. Trailing **`[workspace]`** in `Cargo.toml` helps when this crate lives under another workspace.

## Usage sketch

- **Batch:** [`QuantizationConfig`](src/types.rs) + [`run_quantization`](src/core/stream.rs) → GOZ1 file at `output_path`.
- **Check:** [`verify_pack_file`](src/core/weight_pack_read.rs).
- **Runtime:** [`HybridConfig`](src/types.rs) + [`HybridModel`](src/core/mod.rs).

## Repository

**https://github.com/Spikenaut/grok-ozempic**

## License

GPL-3.0-or-later (see `Cargo.toml`).
