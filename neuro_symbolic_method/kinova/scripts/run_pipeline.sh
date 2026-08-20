#!/usr/bin/env bash
# End-to-end Kinova3 Hanoi pipeline (commands only; long-running steps are sequential).
# Usage (from neuro_symbolic_method/):
#   bash kinova/scripts/run_pipeline.sh
# Or copy individual stages from kinova/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neurosym

unset LD_LIBRARY_PATH PYOPENGL_PLATFORM || true
export MUJOCO_GL=osmesa
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

SEED=0
EP_REG=30
EP_DEMO=30
YOLO=models/yolo/hanoi_yolo.pt

echo "=== 0) Link Hydra configs ==="
bash kinova/scripts/link_hydra_configs.sh

echo "=== 1) GT smoke (Kinova3, no vision) ==="
python -u experiments_neurosymbolic.py --env Hanoi \
  --robot Kinova3 --n_ep 1 --seed "$SEED"

echo "=== 2) Collect regressor CSVs (oracle labels on Kinova3) ==="
python -u auto_demo.py --env Hanoi --robot Kinova3 --train_yolo --rnd_reset \
  --episodes "$EP_REG" --dir ./kinova/data_reg/ --name reg --seed 42

echo "=== 3) Train dual + residual regressors ==="
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -u train_regressor.py \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --active kinova/models/regressors/hanoi_regressor.pkl
python -u train_residual_regressor.py \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --out kinova/models/regressors/hanoi_residual_regressor.pkl
# Optional: copy dual as mono seed if you do not train a cam1-only model yet
cp -n kinova/models/regressors/hanoi_regressor.pkl \
      kinova/models/regressors/hanoi_mono_regressor.pkl || true

echo "=== 4) Collect skill demos (vision + OSC_POSITION) ==="
unset LD_LIBRARY_PATH || true
python -u auto_demo.py --env Hanoi --robot Kinova3 --episodes "$EP_DEMO" \
  --use_yolo --action_split --ee --rnd_reset \
  --name hanoi_yolo --dir ./kinova/data/ --seed "$SEED" \
  --yolo_model "$YOLO" \
  --regressor_model kinova/models/regressors/hanoi_regressor.pkl

echo "=== 5) Preprocess → Zarr ==="
python data_processing/data_to_zarr.py \
  --data_dir kinova/data/Hanoi_seed_${SEED}/hanoi_yolo --filter_actions True

echo "=== 6) Train diffusion skills ==="
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib" WANDB_MODE=offline
cd diffusion_policy
for cfg in train_kinova_grasp train_kinova_drop train_kinova_reach_pick train_kinova_reach_place; do
  python train.py --config-name="$cfg" \
    training.device=cuda:0 logging.mode=offline \
    hydra.run.dir=../kinova/data/train/${cfg}
done
cd ..

echo "=== 7) Copy latest ckpts into kinova/models/policies ==="
# Adjust glob if your hydra output layout differs.
python - <<'PY'
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
    cands = sorted((root / run).rglob("*.ckpt"))
    if not cands:
        print(f"WARNING: no ckpt for {run}")
        continue
    # prefer checkpoints/latest.ckpt when present
    latest = None
    for c in cands:
        if c.name == "latest.ckpt":
            latest = c
            break
    src = latest or cands[-1]
    shutil.copy2(src, dst / name)
    print(f"copied {src} -> {dst / name}")
PY

echo "=== 8) Eval (GT then YOLO) ==="
unset LD_LIBRARY_PATH || true
python -u experiments_neurosymbolic.py --env Hanoi \
  --robot Kinova3 --n_ep 5 --seed "$SEED"
python -u experiments_neurosymbolic.py --env Hanoi \
  --robot Kinova3 --use_yolo --n_ep 5 --seed "$SEED"

echo "Pipeline finished."
