#!/usr/bin/env bash
# Relaunch the 4 Kinova GT diffusion trainings in the existing `kinova` tmux
# session (4 panes, same layout as the previous run).
#
# Old horizon=5 checkpoints are incompatible with the Panda-matched configs
# (horizon=16, n_action_steps=8), so previous run dirs are archived first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${SESSION:-kinova}"
TS="$(date +%Y%m%d_%H%M%S)"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' not found; creating 4-pane session"
  tmux new-session -d -s "$SESSION" -n train
  tmux split-window -t "$SESSION:0" -v
  tmux split-window -t "$SESSION:0.0" -v
  tmux split-window -t "$SESSION:0.1" -v
  tmux select-layout -t "$SESSION:0" even-vertical
  for i in 0 1 2 3; do
    tmux send-keys -t "$SESSION:0.$i" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate neurosym && cd \"$ROOT/diffusion_policy\" && export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\" && export WANDB_MODE=offline" C-m
  done
  sleep 2
fi

bash "$ROOT/kinova/scripts/link_hydra_configs.sh"

cd "$ROOT"
mkdir -p kinova/data/train
for skill in grasp drop reach_pick reach_place; do
  d="kinova/data/train/train_kinova_gt_${skill}"
  if [[ -d "$d" ]]; then
    mv "$d" "${d}_h5_${TS}"
    echo "archived $d -> ${d}_h5_${TS}"
  fi
done

ZARR_BASE="../kinova/data_gt/Hanoi_seed_0/hanoi_gt/hf_traj"
COMMON="training.device=cuda:0 logging.mode=offline training.resume=False task.dataset.max_train_episodes=98"

launch() {
  local pane=$1 cfg=$2 skill=$3 zarr_sub=$4
  local run_dir="../kinova/data/train/train_kinova_gt_${skill}"
  local log="../kinova/data/train/train_kinova_gt_${skill}.log"
  local cmd="cd \"$ROOT/diffusion_policy\" && export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\" && export WANDB_MODE=offline && python -u train.py --config-name=$cfg task.dataset.zarr_path=${ZARR_BASE}/${zarr_sub}/keypoint/keypoint.zarr $COMMON hydra.run.dir=$run_dir 2>&1 | tee $log"
  echo "pane $pane: $cfg"
  tmux send-keys -t "$SESSION:0.$pane" C-c
  sleep 0.2
  tmux send-keys -t "$SESSION:0.$pane" "$cmd" C-m
}

launch 0 train_kinova_grasp      grasp       pick
launch 1 train_kinova_drop       drop        place
launch 2 train_kinova_reach_pick reach_pick  reach_pick
launch 3 train_kinova_reach_place reach_place reach_place

echo "Launched 4 trainings in tmux session '$SESSION' (panes 0=grasp 1=drop 2=reach_pick 3=reach_place)"
tmux list-panes -t "$SESSION:0" -F 'pane=#{pane_index} cmd=#{pane_current_command}'
