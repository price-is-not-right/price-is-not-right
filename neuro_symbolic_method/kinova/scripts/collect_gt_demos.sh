#!/usr/bin/env bash
# Collect Kinova3 Hanoi skill demos with sim GT cube poses (no --use_yolo).
# Same expert as Panda: OSC_POSITION, --ee, --action_split, --object_centric, --rnd_reset.
# Output: kinova/data_gt/Hanoi_seed_<seed>/hanoi_gt/traces/{pick,place,reach_pick,reach_place}.zip
#
# Usage (from neuro_symbolic_method/):
#   bash kinova/scripts/collect_gt_demos.sh
#   EPISODES=100 SEED=0 bash kinova/scripts/collect_gt_demos.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neurosym

unset LD_LIBRARY_PATH PYOPENGL_PLATFORM || true
export MUJOCO_GL=osmesa
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-hanoi_gt}"
DIR="${DIR:-./kinova/data_gt/}"
LOG="${LOG:-kinova/data_gt/collect_${RUN_NAME}_seed${SEED}.log}"
CONVERT="${CONVERT:-1}"

OUT="${DIR}Hanoi_seed_${SEED}/${RUN_NAME}"
mkdir -p kinova/data_gt "$OUT"
# Fresh traces so this run does not mix with a previous collection.
rm -rf "${OUT}/traces"
mkdir -p "${OUT}/traces"

echo "Collecting ${EPISODES} GT episodes -> ${OUT}/"
echo "Log: ${LOG}"

python -u auto_demo.py --env Hanoi --robot Kinova3 \
  --episodes "$EPISODES" --seed "$SEED" \
  --object_centric --action_split --ee --rnd_reset \
  --name "$RUN_NAME" --dir "$DIR" 2>&1 | tee "$LOG"

echo "Done. Traces under ${OUT}/traces/"

if [[ "$CONVERT" == "1" ]]; then
  echo "Converting traces -> zarr"
  python -u data_processing/data_to_zarr.py \
    --data_dir "$OUT" --filter_actions True
  python -u kinova/scripts/diag_dataset.py --base "${OUT}/hf_traj"
fi
