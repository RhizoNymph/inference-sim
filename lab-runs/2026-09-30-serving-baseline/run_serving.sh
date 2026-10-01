#!/usr/bin/env bash
# Serving baseline: vLLM 0.29.0 server on node0 (container, run as the user), bench client from the node0 venv.
set -euo pipefail
LAB=$HOME/inference-sim-lab
OUT=$LAB/serving/results
mkdir -p "$OUT"
export HF_HOME=$HOME/.cache/huggingface
MODEL=Qwen/Qwen2.5-7B-Instruct

docker rm -f isim-serve >/dev/null 2>&1 || true
docker run -d --name isim-serve --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -e VLLM_CACHE_ROOT=/tmp/vllm-cache \
  -v "$HF_HOME:/hf" -p 127.0.0.1:8000:8000 \
  vllm/vllm-openai:v0.29.0 \
  --model "$MODEL" --dtype bfloat16 --max-model-len 4096 --max-num-batched-tokens 2048 \
  --max-num-seqs 64 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --seed 0 \
  > "$OUT/container_id.txt"

for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  if [ -z "$(docker ps -q --filter name=isim-serve)" ]; then echo "SERVER_DIED"; docker logs isim-serve > "$OUT/server.log" 2>&1 || true; exit 1; fi
  sleep 5
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "SERVER_NOT_READY"; exit 1; }
echo "SERVER_READY $(date -Is)"

for RATE in 1 2 4 6 8 10 inf; do
  "$LAB/.venv/bin/vllm" bench serve --backend vllm --base-url http://127.0.0.1:8000 --model "$MODEL" \
    --dataset-name random --random-input-len 512 --random-output-len 128 --random-range-ratio 0 \
    --ignore-eos --num-prompts 200 --request-rate "$RATE" --seed 0 --disable-tqdm \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "rate_${RATE}.json" > "$OUT/bench_rate_${RATE}.log" 2>&1 \
    && echo "RATE_DONE $RATE" || { echo "RATE_FAILED $RATE"; }
done
docker logs isim-serve > "$OUT/server.log" 2>&1 || true
docker rm -f isim-serve >/dev/null 2>&1 || true
echo "ALL_DONE $(date -Is)"
