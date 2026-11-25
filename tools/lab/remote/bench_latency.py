"""Static-batch latency benchmark matching inference-sim's base-solver request shapes.

Runs on a lab node inside the vLLM environment (the vllm/vllm-openai container
or a native venv); it is copied there by `tools/lab/lab.py run` and imports
only what vLLM itself ships (numpy + vllm). Loads the model once, then for each
(batch, prompt) shape times `LLM.generate()` at output lengths

* 1               -> prefill_ms (prefill plus one sampled token),
* 1 + decode      -> full; decode_ms = median(full) - median(prefill),
* decode          -> end_to_end_ms (what the simulator's end_to_end phase models),

with prefix caching disabled by default and fresh random prompts every
iteration, reporting medians over `--iters` after `--warmup` rounds.

Output protocol (consumed by labharness.results): every line this script
writes to `--output` (and echoes to stdout) is one JSON object with a
`record` key:

* `lab.static_batch.v1` - one per shape, the measurement;
* `lab.done.v1`         - written last, after every shape succeeded.

vLLM writes INFO logs to stdout too, so consumers must count only lines that
parse as JSON objects carrying one of these record tags - never raw lines.

This is a generalisation of lab-runs/2026-09-28-static-batch/bench_latency.py;
with the defaults below it performs the same measurement.
"""

import argparse
import contextlib
import json
import statistics
import sys
import time

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

RESULT_RECORD = "lab.static_batch.v1"
DONE_RECORD = "lab.done.v1"


def timed_generate(llm, rng, batch, prompt, max_tokens, vocab):
    prompts = [
        TokensPrompt(prompt_token_ids=rng.integers(1000, vocab - 1000, size=prompt).tolist())
        for _ in range(batch)
    ]
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0)
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    produced = {len(output.outputs[0].token_ids) for output in outputs}
    if produced != {max_tokens}:
        raise RuntimeError(f"expected {max_tokens} tokens per request, got {produced}")
    return elapsed_ms


def parse_shapes(text):
    shapes = []
    for shape in text.split(","):
        batch, prompt = (int(value) for value in shape.lower().split("x"))
        shapes.append((batch, prompt))
    return shapes


def emit(sink, record):
    line = json.dumps(record)
    print(line, flush=True)
    if sink is not None:
        sink.write(line + "\n")
        sink.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--shapes", default="1x512,1x2048,8x512,8x2048,32x512")
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--backend", default=None, help="distributed executor backend, e.g. ray")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="append JSON result lines here")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        distributed_executor_backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    vocab = llm.get_tokenizer().vocab_size
    rng = np.random.default_rng(args.seed)

    with contextlib.ExitStack() as stack:
        sink = stack.enter_context(open(args.output, "a", encoding="utf-8")) if args.output else None
        shapes = parse_shapes(args.shapes)
        for batch, prompt in shapes:
            for _ in range(args.warmup):
                timed_generate(llm, rng, batch, prompt, 1, vocab)
                timed_generate(llm, rng, batch, prompt, 1 + args.decode, vocab)
            prefill = [timed_generate(llm, rng, batch, prompt, 1, vocab) for _ in range(args.iters)]
            full = [
                timed_generate(llm, rng, batch, prompt, 1 + args.decode, vocab) for _ in range(args.iters)
            ]
            e2e = [timed_generate(llm, rng, batch, prompt, args.decode, vocab) for _ in range(args.iters)]
            prefill_ms = statistics.median(prefill)
            decode_ms = statistics.median(full) - prefill_ms
            emit(
                sink,
                {
                    "record": RESULT_RECORD,
                    "batch": batch,
                    "prompt": prompt,
                    "decode": args.decode,
                    "tp": args.tp,
                    "pp": args.pp,
                    "prefill_ms": round(prefill_ms, 3),
                    "decode_ms": round(decode_ms, 3),
                    "decode_ms_per_step": round(decode_ms / args.decode, 4),
                    "end_to_end_ms": round(statistics.median(e2e), 3),
                    "prefill_spread_ms": round(max(prefill) - min(prefill), 3),
                    "e2e_spread_ms": round(max(e2e) - min(e2e), 3),
                    "iters": args.iters,
                },
            )
        emit(sink, {"record": DONE_RECORD, "shapes": len(shapes)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
