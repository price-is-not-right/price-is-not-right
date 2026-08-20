#!/usr/bin/env bash
# Run the Kinova YOLO eval on the deployed checkpoints.
# Robot, policies, regressors and n_act all come from kinova/configs/hanoi.yaml,
# selected by --robot Kinova3 alone.
#
# Usage (from neuro_symbolic_method/):
#   bash kinova/scripts/run_yolo_eval.sh              # 100 ep, seed 0
#   bash kinova/scripts/run_yolo_eval.sh 100 0        # explicit n_ep seed
#   bash kinova/scripts/run_yolo_eval.sh 100 0 kinova/data/train/eval_run.log
#   SYNC=1 bash kinova/scripts/run_yolo_eval.sh       # redeploy from training runs first
#   SYNC=1 CKPT_MODE=best bash kinova/scripts/run_yolo_eval.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SYNC="${SYNC:-0}"
CKPT_MODE="${CKPT_MODE:-latest}"
N_EP="${1:-100}"
SEED="${2:-0}"
LOG="${3:-kinova/data/train/eval_yolo_seed${SEED}_n${N_EP}.log}"

# Off by default: syncing overwrites the deployed checkpoints in
# kinova/models/policies/gt/ with whatever the training runs hold now.
if [[ "$SYNC" == "1" ]]; then
  bash kinova/scripts/sync_gt_policy_checkpoints.sh "$CKPT_MODE"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neurosym
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM || true
export MUJOCO_GL=osmesa
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

echo "=== Eval: Kinova3 + YOLO, n_ep=${N_EP} seed=${SEED} ==="
echo "Log: ${LOG}"
python -u experiments_neurosymbolic.py --env Hanoi \
  --robot Kinova3 --use_yolo \
  --n_ep "$N_EP" --seed "$SEED" 2>&1 | tee "$LOG"

python3 - <<PY
import re
from pathlib import Path
log = Path("$LOG").read_text()
rates = re.findall(r"Success rate:\s*([\d.]+)", log)
pp = re.findall(r"Successful pick_place:\s*\[([^\]]*)\]", log)
rate = float(rates[-1]) if rates else None
print(f"\n=== Final: episode_success_rate={rate} ===")
if pp:
    vals = [int(x.strip()) for x in pp[-1].split(",") if x.strip()]
    print(f"pick_place per episode (last line): {pp[-1][:120]}{'...' if len(pp[-1])>120 else ''}")
    print(f"episodes with full 7/7: {sum(1 for v in vals if v == 7)}/{len(vals)}")
PY
