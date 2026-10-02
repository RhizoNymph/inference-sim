#!/usr/bin/env bash
# Disaggregated prefill/decode serving benchmark on the RTX 3090 lab.
#   prefill: vLLM on node0 (port 8100), decode: vLLM on node1 (port 8200),
#   KV over NIXL (UCX TCP on bond0, decode pulls from prefill: node0 -> node1),
#   proxy + bench client on node0. Same traffic as the colocated baseline.
# Run from the workstation:  bash run_disagg.sh <local-results-dir> [rates...]
set -euo pipefail

OUT_LOCAL=${1:?usage: run_disagg.sh <local-results-dir> [rates...]}
shift || true
RATES=${*:-"1 2 4 6 8 inf"}
MODEL=Qwen/Qwen2.5-7B-Instruct
LAB='$HOME/inference-sim-lab'
REMOTE_RUN='$HOME/inference-sim-lab/runs/disagg'
P_HOST=10.1.1.69
D_HOST=10.1.1.68
# ssh targets: aliases from the workstation, LAN IPs when run on a lab node.
P_SSH=${P_SSH:-node0}
D_SSH=${D_SSH:-node1}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT_LOCAL"

ENGINE_ARGS="--dtype bfloat16 --max-model-len 4096 --max-num-batched-tokens 2048 --max-num-seqs 64 --gpu-memory-utilization 0.85 --no-enable-prefix-caching --seed 0"
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
COMMON_ENV="PATH=$LAB/.venv/bin:\$PATH HF_HOME=\$HOME/.cache/huggingface CUDA_HOME=/usr/local/cuda UCX_TLS=tcp,cuda_copy,self,sm UCX_NET_DEVICES=bond0 PYTHONUNBUFFERED=1"

teardown() {
  echo "teardown"
  ssh -o BatchMode=yes $P_SSH "pkill -f '[d]isagg_proxy.py' || true; pkill -f '[v]llm serve' || true; pkill -f '[V]LLM::' || true" || true
  ssh -o BatchMode=yes $D_SSH "pkill -f '[v]llm serve' || true; pkill -f '[V]LLM::' || true" || true
}
trap teardown EXIT

for node in "$P_SSH" "$D_SSH"; do
  ssh -o BatchMode=yes "$node" "mkdir -p $REMOTE_RUN; pkill -f '[v]llm serve' || true; pkill -f '[V]LLM::' || true"
done
scp -q "$HERE/disagg_proxy.py" "$P_SSH":inference-sim-lab/runs/disagg/

ssh -o BatchMode=yes $D_SSH "cd $REMOTE_RUN || exit 1; nohup env $COMMON_ENV VLLM_NIXL_SIDE_CHANNEL_HOST=$D_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  vllm serve $MODEL --host $D_HOST --port 8200 $ENGINE_ARGS --kv-transfer-config '$KV_CONFIG' > decode_server.log 2>&1 < /dev/null &"
ssh -o BatchMode=yes $P_SSH "cd $REMOTE_RUN || exit 1; nohup env $COMMON_ENV VLLM_NIXL_SIDE_CHANNEL_HOST=$P_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  vllm serve $MODEL --host $P_HOST --port 8100 $ENGINE_ARGS --kv-transfer-config '$KV_CONFIG' > prefill_server.log 2>&1 < /dev/null &"

wait_healthy() {
  local node=$1 url=$2
  for _ in $(seq 1 120); do
    if ssh -o BatchMode=yes $P_SSH "curl -sf $url/health >/dev/null"; then echo "healthy $url"; return 0; fi
    if ! ssh -o BatchMode=yes "$node" "pgrep -f '[v]llm serve' >/dev/null"; then echo "SERVER_DIED on $node"; return 1; fi
    sleep 5
  done
  echo "TIMEOUT waiting for $url"; return 1
}
wait_healthy "$D_SSH" "http://$D_HOST:8200"
wait_healthy "$P_SSH" "http://$P_HOST:8100"

ssh -o BatchMode=yes $P_SSH "cd $REMOTE_RUN || exit 1; nohup env $COMMON_ENV python disagg_proxy.py --prefill http://$P_HOST:8100 --decode http://$D_HOST:8200 --port 8000 > proxy.log 2>&1 < /dev/null &"
for _ in $(seq 1 30); do ssh -o BatchMode=yes $P_SSH "curl -sf http://127.0.0.1:8000/health >/dev/null" && break; sleep 2; done
ssh -o BatchMode=yes $P_SSH "curl -sf http://127.0.0.1:8000/health >/dev/null" || { echo "PROXY_NOT_READY"; exit 1; }
echo "proxy ready"

for RATE in $RATES; do
  if ssh -o BatchMode=yes $P_SSH "cd $REMOTE_RUN && env $COMMON_ENV vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model $MODEL \
      --dataset-name random --random-input-len 512 --random-output-len 128 --random-range-ratio 0 \
      --ignore-eos --num-prompts 200 --request-rate $RATE --seed 0 --disable-tqdm \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir . --result-filename rate_${RATE}.json > bench_rate_${RATE}.log 2>&1"; then
    echo "RATE_DONE $RATE"
  else
    echo "RATE_FAILED $RATE"
  fi
done

scp -q "$P_SSH:inference-sim-lab/runs/disagg/rate_*.json" "$P_SSH:inference-sim-lab/runs/disagg/*.log" "$OUT_LOCAL/" || true
scp -q "$D_SSH:inference-sim-lab/runs/disagg/decode_server.log" "$OUT_LOCAL/" || true
echo "ALL_DONE results in $OUT_LOCAL"
