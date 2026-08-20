#!/usr/bin/env bash
# Symlink Kinova Hydra configs into diffusion_policy's config tree so
# `python train.py --config-name=train_kinova_grasp` works.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/kinova/diffusion_config"
DP_CFG="$ROOT/diffusion_policy/diffusion_policy/config"

mkdir -p "$DP_CFG/task"
for f in "$SRC"/task/kinova_*.yaml; do
  base="$(basename "$f")"
  ln -sfn "$f" "$DP_CFG/task/$base"
  echo "linked task/$base"
done
for f in "$SRC"/train_kinova_*.yaml; do
  base="$(basename "$f")"
  ln -sfn "$f" "$DP_CFG/$base"
  echo "linked $base"
done
echo "Done. From diffusion_policy/: python train.py --config-name=train_kinova_grasp"
