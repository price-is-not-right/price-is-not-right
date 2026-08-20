#!/usr/bin/env bash
# Poll Kinova diffusion training; when all four jobs finish, copy ckpts and run eval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neurosym

LOG="$ROOT/kinova/data/train/monitor.log"
mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

SKILLS=(grasp drop reach_pick reach_place)
RUNS=(train_kinova_grasp train_kinova_drop train_kinova_reach_pick train_kinova_reach_place)
TARGET_EPOCH=4999

count_train_procs() {
  pgrep -af "train.py --config-name=train_kinova_" 2>/dev/null | grep -v monitor | wc -l || true
}

max_epoch_for() {
  local run=$1
  python3 - <<PY
import re, pathlib
p = pathlib.Path("$ROOT/kinova/data/train/$run/checkpoints")
epochs = []
for c in p.glob("epoch=*.ckpt"):
    m = re.search(r"epoch=(\d+)", c.name)
    if m:
        epochs.append(int(m.group(1)))
print(max(epochs) if epochs else -1)
PY
}

log "Monitor started (target epoch >= $TARGET_EPOCH or no train.py processes)."

while true; do
  nproc=$(count_train_procs)
  status=""
  all_done=true
  for i in "${!RUNS[@]}"; do
    run="${RUNS[$i]}"
    skill="${SKILLS[$i]}"
    ep=$(max_epoch_for "$run")
    latest="$ROOT/kinova/data/train/$run/checkpoints/latest.ckpt"
    if [[ ! -f "$latest" ]]; then
      all_done=false
    fi
    status+="${skill}=${ep} "
    if (( ep < TARGET_EPOCH )); then
      all_done=false
    fi
  done
  log "procs=$nproc | $status"
  if (( nproc == 0 )); then
    if $all_done; then
      log "All training jobs finished (target epoch reached)."
    else
      log "WARNING: no train.py processes but not all runs reached epoch $TARGET_EPOCH; using latest checkpoints."
    fi
    break
  fi
  sleep 120
done

log "Copying checkpoints to kinova/models/policies/"
python3 - <<'PY'
from pathlib import Path
import shutil
root = Path("kinova/data/train")
dst = Path("kinova/models/policies")
dst.mkdir(parents=True, exist_ok=True)
mapping = {
    "train_kinova_grasp": "grasp.ckpt",
    "train_kinova_drop": "drop.ckpt",
    "train_kinova_reach_pick": "reach_pick.ckpt",
    "train_kinova_reach_place": "reach_place.ckpt",
}
for run, name in mapping.items():
    ckpt_dir = root / run / "checkpoints"
    latest = ckpt_dir / "latest.ckpt"
    if not latest.exists():
        cands = sorted(ckpt_dir.glob("*.ckpt"))
        if not cands:
            raise SystemExit(f"No checkpoint for {run}")
        latest = cands[-1]
    shutil.copy2(latest, dst / name)
    print(f"copied {latest} -> {dst / name}")
PY

# Ensure mono regressor fallback exists for executor wrist refine.
if [[ ! -f kinova/models/regressors/hanoi_mono_regressor.pkl ]]; then
  cp kinova/models/regressors/hanoi_regressor.pkl \
     kinova/models/regressors/hanoi_mono_regressor.pkl
  log "Created hanoi_mono_regressor.pkl from dual regressor."
fi

unset LD_LIBRARY_PATH PYOPENGL_PLATFORM || true
export MUJOCO_GL=osmesa
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

log "Running GT smoke eval (1 episode)..."
python -u experiments_neurosymbolic.py --env Hanoi \
  --robot Kinova3 --n_ep 1 --seed 0 \
  2>&1 | tee -a "$LOG" | tail -20

log "Running YOLO eval (50 episodes)..."
python -u experiments_neurosymbolic.py --env Hanoi \
  --robot Kinova3 --use_yolo --n_ep 50 --seed 0 \
  2>&1 | tee "$ROOT/kinova/data/train/eval_yolo_seed0.log" | tail -30

log "Eval complete. See kinova/data/train/eval_yolo_seed0.log"
