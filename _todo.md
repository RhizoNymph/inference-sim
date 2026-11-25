# Accuracy roadmap: 3090 lab now, H100s later

Goal: remove every error that does not depend on the GPU while the 3090 lab is
cheap, and make measurement-to-profile turn-key, so H100 time goes only to what
needs H100s (NVLink, InfiniBand, large/MoE models, high-throughput regimes).

"Ready for H100s" means: after calibration on the 3090s, every testable regime
is under ~10% error, and residual errors show no trend with batch size,
context length, or node count. Trending errors are model bugs; flat errors are
constants an H100 run can fit.

## Current state (2026-09-29)

- Single node, static batches, Qwen2.5-7B bf16, vLLM 0.29: 1.3% end-to-end
  error on held-out shapes after fitting `compute_efficiency = 0.85` and
  `decode_memory_bandwidth_scale = 0.84` (uncalibrated: 37%).
- Two-node pipeline parallel over 10GbE: 1.1% end-to-end after the pipeline
  fix (was 50%). Prefill still 14% off; network cost barely mattered in this
  test.
- Untested: serving under load, tensor parallelism, disaggregation, MoE, long
  contexts, other GPUs, heterogeneous/degraded topology, the AISimulate-derived
  H100 profile against real H100s.

## Before H100s (3090 lab)

- [ ] 1. [measured 2026-09-30; simulator comparison pending the harness]
      Serving loop: run `vllm bench serve` at several request rates;
      compare TTFT, TPOT, ITL and throughput with the simulator's serving mode.
      Expected gap: the simulator schedules all prefill before all decode,
      while vLLM interleaves chunked prefill with decode steps. Likely fix is
      the authoritative event loop from goals.md.
- [x] 2. [done 2026-09-30, uncommitted in staging; small-message regime
      addressed by measured collective curves, 4.0% e2e, validation ledger
      entry 7] Tensor parallelism across nodes over Ethernet
      (TP=2, node0+node1):
      the real test of the collective and topology model, since TP crosses
      the network every layer. Expected fixes: one-way-slower links
      (node0->node1 is 3.6 Gb/s vs 9.4 Gb/s back) and a small-message regime
      in the collective cost model.
- [ ] 3. Disaggregated prefill/decode across nodes (1 prefill + 1-2 decode)
      with a vLLM KV connector, to validate the KV-transfer model for the
      first time. Needs ~30 GB freed on node2 for the 3-node variant.
- [ ] 4. Turn-key measurement pipeline: unattended benchmark harness, a fitter
      that emits a complete calibration TOML with holdout stats and feature
      ranges, and a committed validation ledger recording measured accuracy
      per regime.
- [ ] 5. Calibration transfer: keep the 3090 constants and test a different
      model (Qwen2.5-14B across 2 nodes), long contexts (8k-16k) and larger
      batches. Tells us which constants are per-GPU vs per-model, which sizes
      the H100 plan.
- [ ] 6. Small structural gaps already visible: per-iteration overhead floor
      (short prefill 16% under), pipeline microbatching inside one
      multi-request batch (0.85x measured vs 1.03x simulated), decode batch
      non-linearity around CUDA-graph capture sizes.
- [ ] 7. Degraded states: throttled NIC, lowered GPU clocks, node removed
      mid-run. Blocked: `tc` and `nvidia-smi -lgc` need root on the nodes and
      there is no passwordless sudo.
- [ ] Write the H100 plan before getting hardware: fixed matrix of models x
      parallelism x shapes x rates, unattended scripts, time budget.

## Progress log

- 2026-09-30 serving baseline (node0, Qwen2.5-7B, 512 in / 128 out, rates
  1-10 req/s + burst): saturates at ~6 req/s (~800 output tok/s). TPOT rises
  21.8 -> 76 ms while median ITL only 19.4 -> 23.4 ms: decode steps that carry
  a prefill chunk are slow and inflate TPOT. This is the interleaving the
  simulator's prefill-first schedule cannot express. Data:
  lab-harness/lab-runs/2026-09-30-serving-baseline/.
- 2026-09-30 TP=2 over 10GbE: real prefill is ~5x slower than one GPU;
  simulator predicted 0.5x (75% e2e error). Fixed: two all-reduces per layer,
  full activation per rank, collective latency paid per token crossing, TP
  collectives block the next layer by default
  (`allow_compute_comm_overlap = false`). With node0's NIC set to the measured
  3.61 Gb/s and `collective_latency_scale = 1.824` fitted on one shape:
  held-out e2e error 9.6% (prefill 7.5%, decode 14.5%). PP=1 1.4% and PP=2
  1.6% e2e with the same network settings (no regression; PP is an
  independent check of the network fit). Data:
  lab-harness/lab-runs/2026-09-30-tp2/.
- Open from TP: batch-8 decode 18-25% under (57 KB all-reduces), likely an
  NCCL protocol regime. Next step: run nccl-tests all_reduce_perf across
  node0/node1 over a message-size sweep and fit a size-dependent collective
  curve instead of inferring alpha/beta from end-to-end runs.
- Open from TP: the simulator cannot represent a link slower in one
  direction; the lab config approximates it by setting node0's NIC to the
  slow direction.

## After H100s (ordered by information per H100-hour)

- [ ] 1. Validate the AISimulate-derived H100 profile
      (`examples/calibration_h100_vllm_llama31_70b.toml`) against real
      silicon. Decides whether we need to measure most other GPUs ourselves.
- [ ] 2. Intra-node tensor parallelism over NVLink/NVSwitch (TP 2/4/8), and
      FP8.
- [ ] 3. Multi-node over InfiniBand with GPUDirect: rails, NIC affinity,
      NUMA. Fit allreduce curves per fabric with `nccl-tests`.
- [ ] 4. Large and MoE models (70B, Mixtral/DeepSeek-class): expert
      parallelism and MoE all-to-all have no test today.
- [ ] 5. High-throughput regimes: batch 128-256, 32k+ contexts, real request
      traces.
- [ ] 6. Heterogeneous pools if the 3090s can join (e.g. H100 prefill with
      3090 decode).

## Lab notes

- Nodes: node0/node1/node2, 1x RTX 3090 each, 2x10GbE LACP bond (`bond0`).
- node0: snap Docker, use `--runtime=nvidia`; shares its GPU with the
  `tei-nomic` embedding container. node1/node2: no NVIDIA container runtime,
  run natively from `~/inference-sim-lab/.venv` (vllm 0.29.0, torch 2.13.0,
  ray 2.58.0).
- Override `HF_HOME=$HOME/.cache/huggingface` per process (node0's default
  points at an unmounted NAS path; node1's at `~/Models`).
- node2 has ~9 GB free disk.
