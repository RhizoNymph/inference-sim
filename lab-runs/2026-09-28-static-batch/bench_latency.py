"""Static-batch latency benchmark matching inference-sim's base-solver request shapes.

Runs inside the vllm/vllm-openai container. Loads the model once, then for each
(batch, prompt) shape times generate() at output lengths 1 (prefill) and
1 + decode (prefill + decode steps), with prefix caching disabled and fresh
random prompts every iteration. Emits one JSON line per shape.
"""

import argparse
import json
import statistics
import sys
import time

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--shapes", default="1x512,1x2048,8x512,8x2048,32x512")
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        distributed_executor_backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        max_num_batched_tokens=16384,
        max_num_seqs=64,
        enable_prefix_caching=False,
        seed=0,
    )
    vocab = llm.get_tokenizer().vocab_size
    rng = np.random.default_rng(0)

    for shape in args.shapes.split(","):
        batch, prompt = (int(value) for value in shape.split("x"))
        for _ in range(args.warmup):
            timed_generate(llm, rng, batch, prompt, 1, vocab)
            timed_generate(llm, rng, batch, prompt, 1 + args.decode, vocab)
        prefill = [timed_generate(llm, rng, batch, prompt, 1, vocab) for _ in range(args.iters)]
        full = [timed_generate(llm, rng, batch, prompt, 1 + args.decode, vocab) for _ in range(args.iters)]
        e2e = [timed_generate(llm, rng, batch, prompt, args.decode, vocab) for _ in range(args.iters)]
        prefill_ms = statistics.median(prefill)
        decode_ms = statistics.median(full) - prefill_ms
        print(json.dumps({
            "batch": batch, "prompt": prompt, "decode": args.decode,
            "tp": args.tp, "pp": args.pp,
            "prefill_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "decode_ms_per_step": round(decode_ms / args.decode, 4),
            "end_to_end_ms": round(statistics.median(e2e), 3),
            "prefill_spread_ms": round(max(prefill) - min(prefill), 3),
            "e2e_spread_ms": round(max(e2e) - min(e2e), 3),
        }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
