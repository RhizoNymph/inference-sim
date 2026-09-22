# AISimulate calibration-profile converter

`tools/aisimulate_calibration/convert.py` turns NVIDIA AISimulate's measured
per-operation GPU performance tables into an inference-sim calibration profile.
It composes a dense-decoder forward pass op by op from measured kernel
latencies, fits the phase-level linear basis this repo consumes, and emits a
profile TOML with holdout-validated statistics, honest feature ranges, derived
efficiency scalars and benchmark residual entries.

Upstream: <https://github.com/ai-dynamo/AISimulate> (Apache-2.0).  The tool only
ever **reads** a clone; it vendors no data beyond the small self-test fixture.

## Scope

- Dense Llama-style decoders (`LlamaForCausalLM`, and the Qwen2/Qwen3 shapes
  AISimulate routes to the same `LLAMA` model impl).
- vLLM-style tensor parallelism: fused QKV, fused gate+up, row-parallel
  o-proj/down-proj, vocab-parallel lm_head, two custom allreduces per layer.
- Tensor ranks that the measured allreduce table covers (`2`, `4`, `8` on
  `h100_sxm`).
- Two emitted fits, one per phase, because the loader takes the **first** fit in
  TOML order that matches a `(phase, target)` pair.

## Non-scope

- MoE, MLA, Mamba/GDN/KDA linear attention, sparse attention, DSA, multimodal
  encoders.  AISimulate models these with separate tables and op classes.
- Pipeline parallelism, expert parallelism, context parallelism.  `pp` is pinned
  to `1`, so AISimulate's `P2P` op contributes zero; CP is an SGLang-only feature
  upstream.
- Speculative decoding (`nextn` / MTP), beam search, chunked prefill, prefix
  caching.  All are modelled upstream as multipliers or `prefix > 0`; here
  `prefix = 0`, `beam_width = 1`, `mtp_scale_factor = 1`.
- Quantisation other than the model dtype's own GEMM table.  `fp8_static` needs
  the `compute_scale` / `scale_matrix` correction tables, which vLLM 0.24.0 does
  not ship.
- Backend-level aggregate corrections.  AISimulate's vLLM backend adds a
  `layers * 0.8 ms` prefill dispatch overhead and a
  `min(1 + log2(b)/8, 2.0)` TTFT queuing factor on top of the op walk; this tool
  reproduces the **op walk only**, because inference-sim applies its own
  scheduler overhead and queueing model.

## Op walk

Mirrors `python/aisimulate/src/aisimulate_core/sdk/models/llama.py`
(`LLAMAModel.__init__`), with the runtime token counts from
`crates/core/src/perfmodel/session.rs` (`query_context_op`,
`run_generation_ops_step_beamed`).

Let `h` = hidden size, `H` = attention heads, `KV` = kv heads, `d` = head dim,
`I` = intermediate size, `V` = vocab, `tp` = tensor ranks, `L` = layers.

Per-GPU shard widths (AISimulate floor-divides `H`, `I`, `V` and **ceil**-divides
`KV`):

```
H_g   = H // tp          KV_g = ceil(KV / tp)
qkv_n = H*d // tp + d*KV_g*2          (fused Q,K,V - one GEMM)
proj  = (n = h,            k = H*d // tp)
gate  = (n = 2*I // tp,    k = h)     (fused gate+up - one GEMM)
ffn2  = (n = h,            k = I // tp)
logit = (n = V // tp,      k = h)     (always bf16 upstream)
```

### Prefill, `x = b * s` (logits GEMM at `x = b`)

```
prefill_ms(b, s, tp) =
      mem_op(x * h * dtype_bytes)                       # embedding, once
    + allreduce(x * h)                                  # context_embedding_ar
    + L * [ mem_op(x * 8h)                              # add_norm_1
          + gemm(x, qkv_n, h)
          + context_attention(b, s)                     # see below
          + gemm(x, h, H*d//tp)
          + mem_op(x * 8h)                              # add_norm_2
          + gemm(x, 2*I//tp, h)
          + mem_op(x * 2*(2*I//tp + I//tp))             # act_gate
          + gemm(x, h, I//tp)
          + 2 * allreduce(x * h) ]                      # ar_1 + ar_2
    + gemm(b, V//tp, h)                                 # logits, once, m = b
```

`context_attention(b, s)` is the measured FMHA row plus AISimulate's fused
extras, all through the same analytic memory-op formula:

```
q_num = H_g * d ;  kv_num = KV_g * d
extra = 2 * mem_op(2*q_num + 2*kv_num)                  # apply_rope (default on)
      + 2 * mem_op(kv_num * dtype_bytes)                # kv_write (K and V)
context_attention = table(b, s) + 1.1 * extra
```

`use_qk_norm` is false for Llama, so the QK-RMSNorm term is omitted.

### Decode, `x = b`, one step per generated token

```
step_ms(b, c, tp) =
      mem_op(b * h * dtype_bytes)
    + allreduce(b * h)
    + L * [ same layer body as above, with generation_attention(b, c) ]
    + gemm(b, V//tp, h)

decode_ms(b, s, d, tp) = sum over j = 1..d of step_ms(b, s + j, tp)
```

The sum is **exact** - every step is looked up at its own context length
`c = s + j` - not `d` times the mean.  Only the generation-attention term depends
on `c`, so the implementation evaluates the attention curve once over
`c in [s+1, s+d]`, cumulative-sums it, and reuses the context-independent
remainder.  Generation attention carries no fused extras upstream (only the
`use_qk_norm` term, which Llama disables).

### Memory ops

AISimulate has no measured table for norms, activations, embeddings or the
attention fused extras.  It uses
`mem_op_latency_ms` (`crates/core/src/perfmodel/operators/attention.rs`):

```
mem_op(bytes) = (bytes / (mem_bw * mem_bw_empirical_scaling_factor)
                 + mem_empirical_constant_latency) * 1000     # ms
```

On `h100_sxm` that is `3.35 TB/s`, `0.8` and `3 us`.  The 3 us constant is a
per-kernel floor, so it dominates small-batch decode: three elementwise ops per
layer cost at least `9 us` per layer regardless of shape.  Elementwise byte
counts are `2 * (dim_in + dim_out)` per token
(`crates/core/src/perfmodel/py_ops.rs`, `PyElementWise::new`).

### Allreduce

`message_size` is an **element** count, `num_tokens * hidden_size`, not bytes
(`crates/core/src/perfmodel/operators/communication.rs`,
`CustomAllReduceOp::query`).  AISimulate's loader drops every row whose `backend`
or `kernel_source` ends in `_eager` outside the `b60` system, so the
CUDA-graph curve (`vllm_graph`) is the one that backs vLLM predictions; the
converter applies the same filter.  `tp = 1` short-circuits to zero.

## Interpolation rules

| table | axes | rule |
| --- | --- | --- |
| `gemm_perf` | `m` on a collected `(n, k)` site | linear in raw `m` |
| `gemm_perf` | off-site `(n, k)` | inverse-square-distance util transfer in log2 space |
| `context_attention_perf` | `batch`, `isl` | exact `batch`, linear in `sqrt(isl)` |
| `generation_attention_perf` | `batch`, `isl + step` | exact `batch`, linear in the sequence axis |
| `custom_allreduce_perf` | `message_size` | linear in raw element count |

The GEMM off-site path mirrors AISimulate's `Resolver::ScatteredSites`
(`crates/core/src/perfmodel/perf_database/gemm.rs`,
`perf_interp.rs::resolve_pair`): rank collected `(n, k)` sites by Euclidean
distance in `log2` space, gate at `2.0` octaves, take the 4 nearest, evaluate each
site's `m`-curve at the query `m`, and blend achieved **utilisation**
`SOL_i / latency_i` with weights `1 / d^2`.  The answer is
`SOL(query) / blended_util`, with

```
SOL = max( 2*m*n*k / tc_flops * 1000 ,
           dtype_bytes * (m*n + m*k + n*k) / mem_bw * 1000 )    # ms
```

Measured rows are raised to `SOL` at load time, exactly as
`clamp_gemm_grids_to_sol` does upstream.

The context-attention `sqrt` blend is AISimulate's `grid_sqrt_axis` transform on
the sequence axis only (the `s^2` curvature); every other axis blends raw.
Generation attention additionally reproduces the upstream `+/-10%` five-sample
smoothing: for a context `s`, five samples spread over
`[int(0.9 s), int(1.1 s)]` are resolved independently and averaged.

Every lookup outside the measured envelope is a **structured error**, not an
extrapolation.  `CoverageError` carries the table name and the offending query
and exits non-zero.  Concretely:

- `(b, s)` prefill samples with `b * s` above the measured GEMM `m` sweep
  (`32768` on `h100_sxm / vllm / 0.24.0`) are dropped from the grid with a debug
  log rather than extrapolated; if that leaves no samples the run fails.
- A head geometry `(H_g, KV_g, head_dim)` with no measured attention slice is a
  hard failure - it is never borrowed from a neighbouring geometry.
- A `(n, k)` GEMM shape with no collected site within 2 octaves is a hard
  failure.

## Fit basis and why

```
prefill_ms ~ intercept
           + c0 * batch_prompt_tokens_per_tensor_rank          (b*s / tp)
           + c1 * batch_prompt_tokens_squared_per_tensor_rank  (b*s^2 / tp)

decode_ms  ~ intercept
           + c0 * decode_tokens_per_tensor_rank                (d / tp)
           + c1 * decode_batch_tokens_per_tensor_rank          (b*d / tp)
           + c2 * decode_batch_context_tokens_per_tensor_rank  (b*d*ctx / tp)
             with ctx = s + (d + 1) / 2
```

These names are a frozen contract with the sibling Rust branch that adds the
basis features to `src/solver/calibration_fits.rs`.

`1/tp` is folded **into the features** rather than carried as a separate
`tensor_ranks` feature.  The loader picks the first fit matching `(phase,
target)`, so a profile can only carry one prefill fit and one decode fit; a
single fit therefore has to span every tensor rank, and a linear model cannot
express `coef * tokens / tp` unless the division is already inside the basis.
The same logic drives the `b*s^2` prefill term (attention is quadratic in
sequence length) and the `b*d*ctx` decode term (per-step KV reads grow with the
context).

Coefficients and the intercept are fitted with **non-negative** least squares
(Lawson-Hanson active set).  Latency cannot decrease when a basis quantity grows,
and an unconstrained fit happily produces negative coefficients that extrapolate
to negative latency outside the sample grid.

`feature_ranges` are the actual min/max of each composite feature **over the
training split**, so the loader's interpolated/extrapolated classification is
honest about what was fitted, not about what was measured.

Statistics are split honestly: `r_squared`, `adjusted_r_squared`, `rmse`,
`rmse_pct`, `mean_abs_pct_error` and `max_abs_pct_error` come from the training
split; the `validation_*` fields and `validation_sample_count` come from a
disjoint holdout split (`--holdout-fraction`, default `0.2`, seeded).

`[[benchmarks]]` entries are drawn only from the holdout split, so
`measured_ms` (the composed value) versus `predicted_ms` (the fit) is a genuine
out-of-sample residual.

### Derived scalars

- `compute_efficiency` - the 90th percentile of measured MFU
  `2mnk / latency / tc_flops[dtype]` across the whole GEMM table for the model
  dtype, clamped to `(0, 1]`.
- `decode_memory_bandwidth_scale` - the system YAML's
  `gpu.mem_bw_empirical_scaling_factor` verbatim.
- `nccl_version` - the system YAML's `misc.nccl_version`.
- `environment_hash` - `sha256` over the four input table files.
- `source` - the AISimulate git commit, the `system/backend/version` triple, and
  the phrase `composed from measured op tables`.

## Limits

- **Composition is not an end-to-end measurement.**  Summing measured kernel
  latencies assumes serial execution with no overlap, no launch-gap variance, no
  scheduler work and no CPU-side dispatch.  AISimulate itself adds backend-level
  corrections on top (see Non-scope); those are deliberately excluded here.
- **Eager vs graph allreduce.**  The `vllm_eager` rows are filtered out to match
  upstream, so the profile describes a CUDA-graph-captured deployment.  An eager
  deployment will be slower than this profile predicts.
- **Coverage bounds.**  On `h100_sxm / vllm / 0.24.0` the GEMM `m` sweep tops out
  at `32768`, so `b * s` above that is not represented; the attention staircase
  additionally thins out at large `batch x isl`.  The emitted `feature_ranges`
  are the honest witness of what survived.
- **Fit expressiveness.**  The frozen three-term decode basis has no
  tp-independent `d` term, while the measured per-step cost has a real
  tp-independent component (the fixed `3 us` memory-op floors and the allreduce
  latency floor).  The fit absorbs that into the intercept, which inflates the
  relative error at small `d` - visible as a large `mean_abs_pct_error` next to a
  high `r_squared`.  Both numbers are emitted; neither is massaged.
- **Shared-layer inheritance is not reproduced.**  AISimulate's
  `gemm/vllm/0.24.0/reuse.yaml` lets that table borrow missing keys from
  `0.25.0`.  The converter reads the `0.24.0` file only.  For the `bfloat16`
  slice this is a no-op - it is a complete `22 x 22` `(n, k)` grid x `74` `m`
  values - but an `fp8` run would see holes AISimulate does not.
- **Kernel-lane selection is simplified.**  AISimulate resolves an ordered walk
  of `kernel_source` lanes and takes the first whose whole discrete slice exists.
  The converter takes the densest lane for the requested slice, which selects
  `vllm_flash_attn_fa3` on `h100_sxm / vllm / 0.24.0` - the same lane the upstream
  density tier picks - and never mixes rows across lanes.
- **Generation-attention rows are not SOL-clamped** the way upstream clamps them
  at load.  Because every query stays inside the measured envelope, the clamp is
  inert for in-range lookups.

## Files

| path | role |
| --- | --- |
| `tools/aisimulate_calibration/convert.py` | uv script entry point: PEP 723 metadata, argparse CLI, self-test driver |
| `tools/aisimulate_calibration/converter/errors.py` | structured exception hierarchy with per-class exit codes |
| `tools/aisimulate_calibration/converter/logging_setup.py` | structured key-value log formatter |
| `tools/aisimulate_calibration/converter/spec.py` | `ModelSpec`, `ShardedModel`, `GpuSpec`, `GridSpec`, `Provenance`, feature-name constants |
| `tools/aisimulate_calibration/converter/tables.py` | measured-table loaders, interpolation, derived `compute_efficiency` |
| `tools/aisimulate_calibration/converter/opwalk.py` | `RankTables` op walk and phase-sample composition |
| `tools/aisimulate_calibration/converter/fitting.py` | NNLS, train/holdout split, fit statistics |
| `tools/aisimulate_calibration/converter/emit.py` | byte-stable profile TOML rendering |
| `tools/aisimulate_calibration/converter/sources.py` | input-path resolution, system-spec reading, environment hash, git commit |
| `tools/aisimulate_calibration/converter/pipeline.py` | `RunPlan` and the load/compose/fit/render sequence |
| `tools/aisimulate_calibration/fixtures/` | tiny CSV slice + `golden_profile.toml` for `--self-test` |
| `tools/aisimulate_calibration/fixtures/README.md` | exact `duckdb` commands that regenerate the fixture |
| `examples/calibration_h100_vllm_llama31_70b.toml` | generated profile for Llama-3.1-70B on `h100_sxm / vllm / 0.24.0`, `tp` 2/4/8 |
| `src/config/calibration_config.rs` | the parser this profile must satisfy |
| `src/config/sections.rs` | the serde sections defining every emitted field name |

`convert.py` puts its own directory on `sys.path` before importing `converter.*`,
so `uv run tools/aisimulate_calibration/convert.py` works from any cwd without an
install step.

### Key types

- `ModelSpec` / `ShardedModel` - dense decoder geometry and its per-GPU shard.
- `GpuSpec` - the system-YAML scalars (`mem_bw`, scaling factor, constant
  latency, per-dtype `tc_flops`, `nccl_version`, GPUs per node).
- `GemmTable`, `AttentionTable`, `AllReduceCurve` - measured tables plus their
  interpolation rules.
- `RankTables` - one tensor rank's tables and the op walk
  (`prefill_ms`, `decode_series_ms`).
- `Sample`, `PhaseFit`, `FitStats` - composed samples, fitted models, statistics.
- `RunPlan` - everything one invocation needs.
- `ConverterError` subclasses - `ConfigurationError` (2), `TableLoadError` (3),
  `CoverageError` (4), `FitError` (5), `SelfTestError` (6); each exits with its
  own code.

## Invariants

- Exactly one `[[fits]]` block per phase, prefill first, because the first
  matching `(phase, target)` wins in the loader.
- Feature name lists are emitted verbatim from `PREFILL_FEATURES` /
  `DECODE_FEATURES` and must stay byte-identical to the Rust basis.
- `len(features) == len(coefficients)`, and every `feature_ranges[].feature`
  names a feature in the same fit - both are validated by
  `parse_calibration_fit`.
- Every emitted float carries a `.` or an exponent so the Rust `toml` loader
  parses it as `f64` rather than an integer.
- The rendered profile is byte-stable: fixed key order, `%.6g` float formatting,
  seeded splits, and a date injected by the fixture in self-test mode.
- The holdout split is disjoint from the training split, and benchmarks are drawn
  only from the holdout.
- `pipeline_ranks = 1` on every emitted benchmark; the walk models no `P2P`.

## Usage

```sh
# generate a profile from an AISimulate clone
uv run tools/aisimulate_calibration/convert.py \
  --aisimulate-dir /path/to/AISimulate \
  --system h100_sxm --backend vllm --version 0.24.0 \
  --model-preset llama-3.1-70b --tp 2,4,8 \
  --output examples/calibration_h100_vllm_llama31_70b.toml

# verify the composition + fit + emit pipeline against the checked-in fixture
uv run tools/aisimulate_calibration/convert.py --self-test
```

Model geometry comes from `--model-preset` (`llama-3.1-8b`, `llama-3.1-70b`,
`qwen3-32b`) and can be overridden field by field with `--layers`,
`--hidden-size`, `--attention-heads`, `--kv-heads`, `--head-dim`,
`--intermediate-size`, `--vocab-size`, `--parameters-gb`, `--dtype`,
`--kv-dtype`.  The sample grid is `--batch-sizes`, `--prompt-tokens`,
`--decode-tokens`; the split is `--holdout-fraction` and `--seed`.

### Regenerating the example profile

The generated profile records the AISimulate commit it was composed from, so
regenerate it whenever the upstream tables move:

```sh
git clone --depth 1 https://github.com/ai-dynamo/AISimulate.git /tmp/aisimulate
uv run tools/aisimulate_calibration/convert.py \
  --aisimulate-dir /tmp/aisimulate \
  --model-preset llama-3.1-70b --tp 2,4,8 \
  --output examples/calibration_h100_vllm_llama31_70b.toml
```

Regenerating the fixture is documented in
`tools/aisimulate_calibration/fixtures/README.md`.
