#!/usr/bin/env bash
# Run a lab spec with a node's GPU graphics clock locked, always resetting it.
# Needs the lab's passwordless sudo rule for `nvidia-smi -lgc` / `-rgc`.
#   bash run_with_gpu_clock.sh <ssh-host> <clock-mhz> <spec.toml> <runner-log>
set -euo pipefail
HOST=${1:?host}; MHZ=${2:?clock MHz}; SPEC=${3:?spec}; LOG=${4:?runner log}
HERE=$(cd "$(dirname "$0")/.." && pwd)

reset_clock() {
  ssh -o BatchMode=yes "$HOST" "sudo -n /usr/bin/nvidia-smi -rgc >/dev/null" \
    && echo "clock reset on $HOST: $(ssh -o BatchMode=yes "$HOST" 'nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader')" \
    || echo "WARNING: failed to reset GPU clock on $HOST; run: sudo nvidia-smi -rgc"
}
trap reset_clock EXIT

ssh -o BatchMode=yes "$HOST" "sudo -n /usr/bin/nvidia-smi -lgc $MHZ,$MHZ >/dev/null"
echo "clock locked on $HOST at $MHZ MHz"
cd "$HERE"
UV_CACHE_DIR=${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-cache} uv run --quiet lab.py run "$SPEC" > "$LOG" 2>&1
echo "run finished: $(tail -1 "$LOG" | cut -c1-160)"
