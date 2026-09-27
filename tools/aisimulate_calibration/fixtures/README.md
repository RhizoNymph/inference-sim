# Self-test fixture

These CSVs are a minimal slice of the real AISimulate `h100_sxm / vllm / 0.24.0`
measured tables — exactly the rows a tiny two-layer dense decoder needs at
`tp = 2`.  `convert.py --self-test` composes them, fits both phases, emits a
profile, and diffs it against `golden_profile.toml`.

| file | rows | contents |
| --- | --- | --- |
| `gemm_perf.csv` | 32 | 4 collected `(n, k)` sites x 8 `m` values |
| `context_attention_perf.csv` | 9 | `(4 heads, 1 kv head, head_dim 128)` x batch x isl |
| `generation_attention_perf.csv` | 15 | same geometry x batch x `isl + step` |
| `custom_allreduce_perf.csv` | 16 | `num_gpus = 2`, both `vllm_graph` and `vllm_eager` |
| `system.json` | - | the scalars `convert.py` reads out of `h100_sxm.yaml` |
| `fixture.json` | - | model spec, sample grid, pinned date and commit |

`custom_allreduce_perf.csv` deliberately keeps the `vllm_eager` rows so the
self-test exercises the eager-row filter that AISimulate applies at load time.

## Fixture model

`fixture.json` describes a 2-layer decoder with `hidden_size = 1024`,
`attention_heads = 8`, `kv_heads = 2`, `head_dim = 128`,
`intermediate_size = 2048`, `vocab_size = 2048`.  At `tp = 2` every sharded GEMM
lands exactly on a collected `(n, k)` site — `(768, 1024)` qkv, `(1024, 512)`
o-proj, `(2048, 1024)` fused gate+up, `(1024, 1024)` down-proj and lm_head — and
every `m` (`b * s` for prefill, `b` for decode and the logits GEMM) is an exact
grid point.  The golden profile is therefore a pure function of the checked-in
rows, with no interpolation drift.

## Regenerating

Set `AIS` to an AISimulate clone and `FIX` to this directory, then run the four
`duckdb` commands below from a shell.  `duckdb` v1.5 or newer is required.

```sh
AIS=/path/to/AISimulate
DATA="$AIS/python/aisimulate/src/aisimulate_core/systems/data/h100_sxm"
FIX="$(git rev-parse --show-toplevel)/tools/aisimulate_calibration/fixtures"

duckdb -c "COPY (
  SELECT gemm_dtype, m, n, k, latency
  FROM '$DATA/gemm/vllm/0.24.0/gemm_perf.parquet'
  WHERE gemm_dtype = 'bfloat16'
    AND (n, k) IN ((768, 1024), (1024, 512), (1024, 1024), (2048, 1024))
    AND m IN (1, 2, 4, 128, 256, 512, 1024, 2048)
  ORDER BY n, k, m
) TO '$FIX/gemm_perf.csv' (HEADER, DELIMITER ',');"

duckdb -c "COPY (
  SELECT kernel_source, batch_size, isl, num_heads, num_key_value_heads,
         head_dim, beam_width, attn_dtype, kv_cache_dtype, window_size, latency
  FROM '$DATA/attention/vllm/0.24.0/context_attention_perf.parquet'
  WHERE num_heads = 4 AND num_key_value_heads = 1 AND head_dim = 128
    AND window_size = 0 AND beam_width = 1
    AND attn_dtype = 'bfloat16' AND kv_cache_dtype = 'bfloat16'
    AND batch_size IN (1, 2, 4) AND isl IN (128, 256, 512)
  ORDER BY batch_size, isl
) TO '$FIX/context_attention_perf.csv' (HEADER, DELIMITER ',');"

duckdb -c "COPY (
  SELECT kernel_source, batch_size, isl, step, num_heads, num_key_value_heads,
         head_dim, beam_width, kv_cache_dtype, window_size, latency
  FROM '$DATA/attention/vllm/0.24.0/generation_attention_perf.parquet'
  WHERE num_heads = 4 AND num_key_value_heads = 1 AND head_dim = 128
    AND window_size = 0 AND beam_width = 1 AND kv_cache_dtype = 'bfloat16'
    AND batch_size IN (1, 2, 4) AND step IN (63, 127, 255, 511, 1023)
  ORDER BY batch_size, step
) TO '$FIX/generation_attention_perf.csv' (HEADER, DELIMITER ',');"

duckdb -c "COPY (
  SELECT kernel_source, allreduce_dtype, num_gpus, message_size, latency, backend
  FROM '$DATA/comm/vllm/0.24.0/custom_allreduce_perf.parquet'
  WHERE num_gpus = 2
    AND message_size IN (1024, 2048, 4096, 131072, 262144, 524288, 1048576, 2097152)
  ORDER BY backend, message_size
) TO '$FIX/custom_allreduce_perf.csv' (HEADER, DELIMITER ',');"
```

Why those coordinates:

- `m` covers every prefill `b * s` in the fixture grid (`128 .. 2048`) plus the
  decode / logits `m = b` values (`1, 2, 4`).
- `isl` covers the fixture prompt lengths.
- `step` values are `seq - 1` for `seq in {64, 128, 256, 512, 1024}`, because the
  generation table is keyed by `isl + step` and the +/-10% five-sample smoothing
  reaches down to `116` and up to `572` for this grid.
- `message_size` is an **element** count (`num_tokens * hidden_size`), so it
  covers `{1, 2, 4} * 1024` for decode and `{128 .. 2048} * 1024` for prefill.

Then refresh the golden profile and confirm the diff is clean:

```sh
uv run tools/aisimulate_calibration/convert.py --self-test --update-golden
uv run tools/aisimulate_calibration/convert.py --self-test
```
