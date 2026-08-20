#!/usr/bin/env bash
# End-to-end Kinova3 Hanoi pipeline: demos -> policies -> regressors -> eval.
# Steps run sequentially and training takes about a day; prefer running the
# stages individually from kinova/README.md unless you want the whole thing.
#
# Usage (from neuro_symbolic_method/):
#   bash kinova/scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neurosym

unset LD_LIBRARY_PATH PYOPENGL_PLATFORM || true
export MUJOCO_GL=osmesa
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

SEED=0
EP_DEMO=30
EP_REG=30
EP_EVAL=100

echo "=== 1) Link Hydra configs ==="
bash kinova/scripts/link_hydra_configs.sh

echo "=== 2) Collect demonstrations (+ Zarr conversion) ==="
EPISODES="$EP_DEMO" SEED="$SEED" bash kinova/scripts/collect_gt_demos.sh

echo "=== 3) Train the four diffusion policies ==="
bash kinova/scripts/train_gt_policies.sh

echo "=== 4) Deploy checkpoints ==="
bash kinova/scripts/sync_gt_policy_checkpoints.sh latest

echo "=== 5) Collect regressor CSVs ==="
python -u auto_demo.py --env Hanoi --robot Kinova3 --train_yolo --rnd_reset \
  --episodes "$EP_REG" --dir ./kinova/data_reg/ --name reg --seed 42

echo "=== 6) Train the regressors ==="
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -u train_regressor.py \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --active kinova/models/regressors/hanoi_regressor.pkl
python -u train_regressor.py --mono \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --active kinova/models/regressors/hanoi_mono_regressor.pkl
python -u train_residual_regressor.py \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --out kinova/models/regressors/hanoi_residual_regressor.pkl
unset LD_LIBRARY_PATH || true

echo "=== 7) Eval (sim poses, then vision) ==="
python -u experiments_neurosymbolic.py --env Hanoi \
  --robot Kinova3 --n_ep "$EP_EVAL" --seed "$SEED"
python -u experiments_neurosymbolic.py --env Hanoi \
  --robot Kinova3 --use_yolo --n_ep "$EP_EVAL" --seed "$SEED"

echo "Pipeline finished."
