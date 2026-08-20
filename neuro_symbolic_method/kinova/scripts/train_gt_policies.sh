#!/usr/bin/env bash
# Train all 4 Kinova diffusion policies from the hanoi_gt zarr.
# max_train_episodes in the task yamls is an upper cap, so it adapts to the
# number of demos you collected.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/diffusion_policy"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neurosym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export WANDB_MODE=offline

ZARR_BASE="../kinova/data_gt/Hanoi_seed_0/hanoi_gt/hf_traj"
COMMON="training.device=cuda:0 logging.mode=offline task.dataset.max_train_episodes=98"

run_one() {
  local cfg=$1 skill=$2 zarr_sub=$3
  local run_dir="../kinova/data/train/train_kinova_gt_${skill}"
  echo "=== ${cfg} -> ${run_dir} ==="
  python train.py --config-name="$cfg" \
    task.dataset.zarr_path="${ZARR_BASE}/${zarr_sub}/keypoint/keypoint.zarr" \
    $COMMON \
    hydra.run.dir="$run_dir"
}

run_one train_kinova_grasp      grasp       pick
run_one train_kinova_drop       drop        place
run_one train_kinova_reach_pick reach_pick  reach_pick
run_one train_kinova_reach_place reach_place reach_place

echo "All 4 GT policy trainings finished."
