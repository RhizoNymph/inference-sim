#!/usr/bin/env bash
# Frontend latency probe: same server as the serving baseline; low-rate
# requests that generate exactly one token, at 512- and 16-token prompts.
set -euo pipefail
LAB=$HOME/inference-sim-lab
OUT=$LAB/frontend/results
mkdir -p "$OUT"
export HF_HOME=$HOME/.cache/huggingface
MODEL=Qwen/Qwen2.5-7B-Instruct
docker rm -f isim-serve >/dev/null 2>&1 || true
docker run -d --name isim-serve --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -e VLLM_CACHE_ROOT=/tmp/vllm-cache \
  -v "$HF_HOME:/hf" -p 127.0.0.1:8000:8000 \
  vllm/vllm-openai:v0.29.0 \
  --model "$MODEL" --dtype bfloat16 --max-model-len 4096 --max-num-batched-tokens 2048 \
  --max-num-seqs 64 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --seed 0 > /dev/null
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  [ -z "$(docker ps -q --filter name=isim-serve)" ] && { echo "SERVER_DIED"; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "SERVER_NOT_READY"; exit 1; }
echo "SERVER_READY"
for IN in 512 16; do
  for OUTLEN in 1 2; do
    "$LAB/.venv/bin/vllm" bench serve --backend vllm --base-url http://127.0.0.1:8000 --model "$MODEL" \
      --dataset-name random --random-input-len $IN --random-output-len $OUTLEN --random-range-ratio 0 \
      --ignore-eos --num-prompts 40 --request-rate 0.5 --seed 0 --disable-tqdm \
      --percentile-metrics ttft,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir "$OUT" --result-filename "in${IN}_out${OUTLEN}.json" > "$OUT/bench_in${IN}_out${OUTLEN}.log" 2>&1 \
      && echo "DONE in$IN out$OUTLEN" || echo "FAILED in$IN out$OUTLEN"
  done
done
docker rm -f isim-serve >/dev/null 2>&1 || true
echo "ALL_DONE"
