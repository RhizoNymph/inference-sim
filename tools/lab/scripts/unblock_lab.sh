#!/usr/bin/env bash
# Unblock inference-sim lab work on node0/node1/node2:
#   1. node2: clear re-downloadable caches (uv, pip, docker build cache) to free ~45 GB.
#   2. all nodes: install a narrow passwordless sudo rule limited to throttling
#      bond0 with tc and pinning GPU clocks with nvidia-smi.
#   3. verify both from the same ssh sessions.
#
# Run from the workstation that can `ssh node0/node1/node2`. sudo asks for
# your password once per node. Safe to re-run. To undo the sudo rule on a
# node: sudo rm /etc/sudoers.d/inference-sim-lab
set -euo pipefail

NODES=(node0 node1 node2)
CACHE_NODE=node2
RULE_PATH=/etc/sudoers.d/inference-sim-lab

RULE='# inference-sim lab: degraded-state experiments only (NIC throttling, GPU clocks)
nymph ALL=(root) NOPASSWD: /usr/sbin/tc qdisc show dev bond0, \
    /usr/sbin/tc qdisc add dev bond0 root *, \
    /usr/sbin/tc qdisc del dev bond0 root, \
    /usr/bin/nvidia-smi -lgc *, \
    /usr/bin/nvidia-smi -rgc'
# Shipped base64-encoded so no quoting survives the ssh hop.
RULE_B64=$(printf '%s\n' "$RULE" | base64 -w0)

echo "== $CACHE_NODE: clearing caches"
ssh "$CACHE_NODE" 'bash -s' <<'REMOTE'
set -u
echo "  before: $(df -h --output=avail / | tail -1 | tr -d " ") free"
if command -v uv >/dev/null; then uv cache clean >/dev/null 2>&1 && echo "  uv cache cleared" || echo "  uv cache clean failed (is a uv process running?)"; fi
python3 -m pip cache purge >/dev/null 2>&1 || rm -rf "$HOME/.cache/pip"
echo "  pip cache cleared"
docker builder prune -f >/dev/null 2>&1 && echo "  docker build cache cleared" || echo "  docker builder prune failed"
echo "  after:  $(df -h --output=avail / | tail -1 | tr -d " ") free"
REMOTE

failed=()
for node in "${NODES[@]}"; do
  echo "== $node: installing sudo rule (sudo may prompt for your password)"
  # -t gives sudo a terminal for the password prompt.
  if ssh -t "$node" "
    tmp=\$(mktemp) &&
    echo '$RULE_B64' | base64 -d > \"\$tmp\" &&
    sudo visudo -c -q -f \"\$tmp\" &&
    sudo install -m 0440 -o root -g root \"\$tmp\" '$RULE_PATH'
    rc=\$?; rm -f \"\$tmp\"
    [ \$rc -eq 0 ] || exit \$rc
    sudo -k
    sudo -n /usr/sbin/tc qdisc show dev bond0 >/dev/null && sudo -n -l /usr/bin/nvidia-smi -rgc >/dev/null
  "; then
    echo "  $node: rule installed and works without a password"
  else
    echo "  $node: FAILED (rule not installed or not passwordless)"
    failed+=("$node")
  fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "All done. Tell Claude the lab is unblocked."
else
  echo "Failed on: ${failed[*]}. Nothing was left half-installed: visudo checks the rule before it is copied."
  exit 1
fi
