# H100 validation plan

What to run on H100s, in what order, and what "done" means for each step.
Everything here is ordered by information per H100-hour: the expensive
hardware should only answer questions the 3090 lab cannot. The 3090 lab
validated the simulator's structure (roofline shape, pipeline and tensor
parallel traffic, the serving iteration engine, measured collective curves);
the H100 runs calibrate constants and test the regimes a 3090 cannot reach.

Every run goes through `tools/lab` (see `docs/features/lab_harness.md`), so a
session is a list of spec files, not hand-typed commands. Accuracy results go
into `docs/validation_ledger.md` with their raw data under `lab-runs/`.

## Lessons from the 3090 lab that shape this plan

1. **Give every model its real MLP width.** Specs without `ffn_hidden_size`
   derive FLOPs from a 4x-hidden default, and a fitted efficiency constant
   silently absorbs the error until a second model exposes it (ledger
   entry 11). Every H100 model spec must carry `ffn_hidden_size` (and expert
   counts for MoE) taken from `config.json`.
2. **Test transfer, not just fit.** A constant fitted on one model and shape
   range can look like 1% error and still be wrong. Every fit gets validated on
   shapes, models and parallelism layouts it was not fitted on.
3. **Measure collectives directly.** Fitting alpha-beta network scalars from
   end-to-end runs missed NCCL's protocol steps; measured curves did not
   (TP=2: 9.6% -> 4.0%, ledger entries 7 and 12). Fit a curve per fabric scope
   before any multi-GPU end-to-end run.
4. **Time point-to-point transfers on the receiver.** Sender-side timing of
   NCCL sends returns before delivery for small messages.
5. **Peak FLOPs come from the clock you actually run at.** The 3090's
   datasheet peak assumes a lower boost clock than the card sustains; read
   `nvidia-smi --query-gpu=clocks.max.sm` (and the locked clock, if the
   cluster pins clocks) and set `peak_f16_tflops` accordingly.
6. **Compute efficiency depends on tokens per step.** It rose from ~0.66 at
   256 tokens to ~0.89 at 1k+ on the 3090; fit the efficiency curve from a
   prefill token sweep rather than one scalar.
7. **Watch KV capacity.** vLLM's KV cache shrinks as `max_num_batched_tokens`
   grows (24,896 tokens at 32768 vs 82,864 at 2048 on the 3090 with
   Qwen2.5-7B). Size static-batch shapes from the capacity vLLM reports, or
   the run measures preemption instead of the regime you meant to test.
8. **Orchestrate from inside the cluster.** Run the harness on a head node
   with LAN access to every node; a lossy link to the controller broke 3090
   runs mid-experiment.

## Before the session (no H100 time spent)

- [ ] Lab config `tools/lab/labs/<cluster>.toml`: nodes, ssh hosts (LAN
      addresses from the head node), launch method (container image pinned to
      a vLLM release at least a week old, or a venv), `HF_HOME`, NCCL/GLOO
      interfaces, `CUDA_HOME`, Ray port.
- [ ] Simulator cluster TOML for the same hardware: GPU profile (H100 SXM
      80 GB, HBM 3.35 TB/s, clock-adjusted peak), NVLink/NVSwitch intra-node,
      InfiniBand NICs and rails, GPU-to-NIC affinity, NUMA maps.
- [ ] Model specs with `ffn_hidden_size`, `kv_heads`, `parameters_gb` from the
      actual safetensors, and expert counts: a dense 8B, a dense 70B, and one
      MoE (Mixtral-8x7B or a DeepSeek-class model) at minimum.
- [ ] Weights pre-staged on every node (shared filesystem or rsync from the
      head node; mind the huggingface_hub shared-blob layout when copying).
- [ ] Every spec dry-runs cleanly (`lab.py run <spec> --dry-run`) and the
      harness tests pass.
- [ ] Record the AISimulate-derived prediction for every phase-1 shape
      before measuring (`examples/calibration_h100_vllm_llama31_70b.toml`), so
      phase 1 is a blind test.

## Phases (ordered by information per H100-hour)

Budgets assume one 8xH100 node for phases 1-3 and two nodes for phase 4.

### Phase 1: blind test of the AISimulate-derived profile (~2 h)

Static-batch runs of Llama-3.1-70B, TP 2/4/8, the same shape grid as the
generated profile (batch 1-64, prompts 128-16384, decode 32-1024).

Done when: predicted vs measured is recorded for every shape. If end-to-end
error is under ~10%, AISimulate's measured tables can stand in for GPUs we do
not own; if not, the per-op residuals show which tables or compositions
diverge, and phases 2-3 become the calibration source.

### Phase 2: intra-node collective curves and fabric (~1 h)

`collective-bench` spec (or `nccl-tests` with `-G` for CUDA graphs) on one
node: all_reduce, all_gather, reduce_scatter, all_to_all at 2/4/8 ranks;
send_recv in both directions, receiver-timed; 1 KiB to 8 GiB. Convert with
`lab.py curves` into `[[collective_curves]]` for the cluster TOML.

Done when: curves cover every rank count and message size the later phases
use (TP decode sizes through 70B prefill sizes) without extrapolation.

### Phase 3: single-node calibration and transfer (~3 h)

1. Prefill token sweep (batch 1, prompts 16-16384) and decode batch sweep
   (batch 1-256, inside the reported KV capacity) for the dense 8B at TP=1:
   fit the token-dependent efficiency curve and the decode bandwidth scale.
2. Hold those constants fixed and validate: dense 8B at TP 2/4/8, dense 70B
   at TP 4/8, FP8 variants, long context (32k+).
3. Serving sweeps (`vllm bench serve`, Poisson rates to past saturation) for
   the 8B at TP=1 and the 70B at TP=8 against the iteration engine.

Done when: static-batch end-to-end error is under ~10% on held-out models and
layouts with no trend in batch, context or TP degree, and serving TPOT/ITL
p50 are within ~15% below saturation.

### Phase 4: multi-node over InfiniBand (~3 h)

1. Inter-node collective curves (same sweep as phase 2, across nodes and
   rails, GPUDirect on and off if configurable).
2. 70B at TP=8 x PP=2 and TP=16 across nodes; disaggregated prefill/decode
   with NIXL over InfiniBand (one prefill node, one decode node), serving
   sweeps through the same proxy pattern as the 3090 lab.

Done when: multi-node static error is under ~10%, and disaggregated TTFT/TPOT
are within ~20% with the KV-transfer time attributed correctly.

### Phase 5: MoE and high throughput (~2 h)

MoE model at EP 2/4/8 (static and serving), batch 128-256, 32k+ contexts,
and a replayed production-like trace if one is available.

Done when: MoE all-to-all and expert compute are within ~15%, or the
residuals point to the specific missing term.

### Phase 6 (optional): heterogeneous pools

If the 3090 lab can join the H100 network: H100 prefill with 3090 decode,
exercising heterogeneous routing and KV transfer across GPU generations.

## Total and contingency

About 11 H100-hours for phases 1-5, plus roughly 30% contingency for reruns.
If time is cut, keep phases 1-3: they decide whether measured AISimulate data
is trustworthy and calibrate the constants every later prediction depends on.
