#!/usr/bin/env bash
# SIGINT each Kinova GT trainer once its log shows epoch >= CUT.
set -euo pipefail
CUT="${CUT:-6000}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
declare -A PIDS=(
  [grasp]=1025597
  [reach_pick]=1025607
  [reach_place]=1025668
)

epoch_of() {
  local log="$ROOT/kinova/data/train/train_kinova_gt_$1.log"
  [[ -f "$log" ]] || { echo 0; return; }
  grep -oE "Training epoch [0-9]+" "$log" | tail -1 | awk '{print $3}'
}

while true; do
  alive=0
  for skill in "${!PIDS[@]}"; do
    pid="${PIDS[$skill]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    alive=1
    ep="$(epoch_of "$skill")"
    ep="${ep:-0}"
    if [[ "$ep" -ge "$CUT" ]]; then
      echo "$(date -Is) stopping $skill pid=$pid at epoch $ep"
      kill -INT "$pid" 2>/dev/null || true
      unset PIDS["$skill"]
    else
      echo "$(date -Is) $skill epoch=$ep (cut at $CUT)"
    fi
  done
  [[ "$alive" -eq 0 ]] && break
  [[ ${#PIDS[@]} -eq 0 ]] && break
  sleep 20
done
echo "cut-at-${CUT} watcher done"
