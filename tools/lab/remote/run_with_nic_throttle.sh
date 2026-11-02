#!/usr/bin/env bash
# Run a lab spec with bond0 egress throttled by a tbf qdisc on each listed
# host, always removing the qdisc afterwards. Needs the lab's passwordless
# sudo rule for `tc qdisc {show,add ... root *,del root} dev bond0`.
# Throttling bond0 slows every service on the host: keep runs short.
#   bash run_with_nic_throttle.sh <rate, e.g. 2gbit> <spec.toml> <runner-log> <ssh-host>...
set -euo pipefail
RATE=${1:?rate}; SPEC=${2:?spec}; LOG=${3:?runner log}; shift 3
HOSTS=("$@")
[ ${#HOSTS[@]} -gt 0 ] || { echo "need at least one ssh host"; exit 2; }
HERE=$(cd "$(dirname "$0")/.." && pwd)

remove_throttle() {
  for host in "${HOSTS[@]}"; do
    ssh -o BatchMode=yes "$host" "sudo -n /usr/sbin/tc qdisc del dev bond0 root 2>/dev/null; sudo -n /usr/sbin/tc qdisc show dev bond0" \
      && echo "throttle removed on $host" \
      || echo "WARNING: failed to check bond0 on $host; run: sudo tc qdisc del dev bond0 root"
  done
}
trap remove_throttle EXIT

for host in "${HOSTS[@]}"; do
  ssh -o BatchMode=yes "$host" "sudo -n /usr/sbin/tc qdisc add dev bond0 root tbf rate $RATE burst 4mb latency 100ms && sudo -n /usr/sbin/tc qdisc show dev bond0"
  echo "bond0 egress throttled on $host at $RATE"
done
cd "$HERE"
UV_CACHE_DIR=${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-cache} uv run --quiet lab.py run "$SPEC" > "$LOG" 2>&1
echo "run finished: $(tail -1 "$LOG" | cut -c1-160)"
