# xai-dissect Manifest — schema v1

Machine-readable JSON contract that lets `grok-ozempic` consume structural
analysis produced by the upstream [`xai-dissect`](https://github.com/rmems/xai-dissect)
repository, without depending on it as a runtime crate.

This document freezes **schema v1**. Phase 1 implements the loader only;
the batch pipeline in `src/core/stream.rs` is not rewired yet.

## Authority and source of truth

- **`xai-dissect` is authoritative.** Manifests are produced there.
- The `dissect/grok-1/baseline.json` file committed in this repo is a
  **non-authoritative fallback / reference copy** regenerated from
  `xai-dissect` releases. Do not treat it as ground truth.
- `grok-ozempic` **never writes manifests** and never depends on
  `xai-dissect` as a Cargo crate.

## Delivery

The manifest reaches `grok-ozempic` through one of the following paths,
resolved in order (first hit wins):

1. Explicit `QuantizationConfig.manifest_path` (caller-provided).
2. Environment variable `GROK_OZEMPIC_MANIFEST`.
3. In-tree baseline at `dissect/grok-1/baseline.json` (reference fallback).
4. Legacy heuristic in `stream.rs` (`router_patterns` substring match) —
   preserved only until phase 2 wiring lands.

Phase 1 exposes the config field and loader; enforcement of the resolution
order lives in the selection/precision seams introduced in phase 2.

## Manifest precedence over legacy `router_patterns`

When a manifest is provided, it **wins** over the legacy
`QuantizationConfig.router_patterns` substring list. In phase 2 the
wiring code will log a deprecation warning if both are present. The
legacy field remains supported only for the manifest-less fallback path.

## Schema v1

```json
{
  "schema": "xai-dissect.manifest",
  "schema_version": 1,
  "model": {
    "family": "grok-1",
    "source": "xai-org/grok-1",
    "tensor_name_convention": "blk.{L}.{role}.weight"
  },
  "produced_by": {
    "tool": "xai-dissect",
    "version": "0.x.y",
    "commit": "optional-sha"
  },
  "defaults": {
    "precision": "ternary_snn",
    "gif_threshold": 0.05
  },
  "preserve": [
    { "name": "blk.*.attn_router.weight", "reason": "routing-critical" }
  ],
  "fp16": [
    { "name": "token_embd.weight", "reason": "embedding table" }
  ],
  "ternary_candidates": [
    { "name": "blk.0.ffn_up.weight",   "rank": 0.98, "gif_threshold": 0.04 },
    { "name": "blk.0.ffn_down.weight", "rank": 0.95 }
  ],
  "blocks": [
    { "index": 0, "experts": 8, "role": "moe" }
  ]
}
```

### Resolution order inside a manifest

`preserve` > `fp16` > `ternary_candidates` > `defaults`.

### Name matching

- Exact tensor names or simple globs where `*` matches **exactly one**
  dotted segment (e.g. `blk.*.attn_router.weight` matches
  `blk.0.attn_router.weight` but **not**
  `blk.0.sub.attn_router.weight`).
- Matching is anchored at dotted segments. The pattern and the tensor
  name must have the same segment count; each segment must equal `*` or
  match the literal.
- **No regular expressions in v1.**
- `gate` substring matches are intentionally not performed; this is why
  globs must be segment-anchored. Historical false positives like
  `ffn_gate` being swept up by a `gate` substring are impossible in v1.

### Precision tiers (v1)

| Tier           | Meaning                                                 |
| -------------- | ------------------------------------------------------- |
| `preserve`     | Keep source dtype. **Reserved in phase 1** — consumed by the pipeline in a later phase. During the transitional window this may be aliased to FP16 passthrough; the alias is temporary and tracked via a follow-up issue. |
| `fp16`         | Force FP16 passthrough (current router behavior).       |
| `ternary_snn` | Ternary {-1, 0, +1} with GIF saliency threshold.         |

### Per-tensor fields

- `rank` — optional hint in `[0, 1]`. Higher = stronger ternary candidate.
- `gif_threshold` — optional per-tensor override of the global threshold.
- `reason` — free-form human string, ignored by the loader.

## Hard-fail validation

The loader **must** reject the following with typed errors rather than
best-effort parse:

- `schema_version` other than `1`.
- `model.tensor_name_convention` other than `"blk.{L}.{role}.weight"`
  (v1 only supports the canonical Grok-1 naming used by
  `src/core/npy.rs::npy_stem_to_tensor_name`).
- Non-existent / unreadable manifest file.
- Malformed JSON.

Unknown top-level fields are **tolerated** for forward compatibility.

## Versioning

Future schema versions bump `schema_version`. Loaders must refuse any
version they do not explicitly understand. This prevents silent drift
between `xai-dissect` and `grok-ozempic`.
