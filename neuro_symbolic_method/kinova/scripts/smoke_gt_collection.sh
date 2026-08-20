#!/usr/bin/env bash
# Smoke-test that Kinova GT collection actually completed the Hanoi task.
# 1) log + zarr coherence on the 100-ep run
# 2) one live expert episode (same flags as collection)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neurosym
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM || true
export MUJOCO_GL=osmesa
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

BASE="${BASE:-kinova/data_gt/Hanoi_seed_0/hanoi_gt}"
LOG="${LOG:-kinova/data_gt/collect_hanoi_gt_seed0.log}"
FAIL=0

echo "========== 1) collection log =========="
n_ok=$(grep -c "Successful episode?:  True" "$LOG" || true)
n_bad=$(grep -c "Successful episode?:  False" "$LOG" || true)
n_rec=$(grep -c "Number of recorded episodes:" "$LOG" || true)
echo "Successful episode True=$n_ok  False=$n_bad  recorded_lines=$n_rec"
if [[ "$n_ok" -lt 100 || "$n_bad" -ne 0 ]]; then
  echo "FAIL: expected 100 successful / 0 failed episodes"
  FAIL=1
fi

echo "========== 2) zarr coherence =========="
if [[ ! -d "${BASE}/hf_traj" ]]; then
  echo "zarr missing; converting traces"
  python -u data_processing/data_to_zarr.py --data_dir "$BASE" --filter_actions True
fi
python -u kinova/scripts/diag_dataset.py --base "${BASE}/hf_traj"

echo "========== 2b) zarr shape checks =========="
import sys, numpy as np, zarr
from pathlib import Path
base = Path("${BASE}/hf_traj")
fail = 0
for skill, act_dim, obs_dim in [
    ("reach_pick", 3, 3), ("pick", 2, 2),
    ("reach_place", 3, 3), ("place", 2, 2),
]:
    p = base / skill / "keypoint" / "keypoint.zarr"
    r = zarr.open(str(p), "r")
    n = len(r["meta"]["episode_ends"])
    a = np.array(r["data"]["action"])
    s = np.array(r["data"]["state"])
    print(f"CHECK {skill}: episodes={n} action={a.shape} state={s.shape}")
    if n < 100:
        print(f"  FAIL: expected 100 episodes, got {n}")
        fail = 1
    if a.shape[1] != act_dim or s.shape[1] != obs_dim:
        print(f"  FAIL: expected act_dim={act_dim} obs_dim={obs_dim}")
        fail = 1
    if np.isnan(a).any() or np.isnan(s).any():
        print("  FAIL: NaNs in data")
        fail = 1
    if skill.startswith("reach"):
        ends = np.array(r["meta"]["episode_ends"])
        xy = np.array([np.linalg.norm(s[e-1, :2]) for e in ends])
        print(f"  terminal xy mm: mean={xy.mean()*1000:.2f} max={xy.max()*1000:.2f}")
        if xy.mean() > 0.02:
            print("  FAIL: mean terminal XY > 20mm (expert did not get over the target)")
            fail = 1
sys.exit(fail)
PY

echo "========== 3) live 1-episode expert execution =========="
SMOKE_DIR="./kinova/data_smoke/"
SMOKE_NAME="hanoi_gt_exec_check"
rm -rf "${SMOKE_DIR}Hanoi_seed_123/${SMOKE_NAME}"
python -u auto_demo.py --env Hanoi --robot Kinova3 \
  --episodes 1 --seed 123 \
  --object_centric --action_split --ee --rnd_reset \
  --name "$SMOKE_NAME" --dir "$SMOKE_DIR" \
  | tee kinova/data_smoke/exec_check.log

if ! grep -q "Successful episode?:  True" kinova/data_smoke/exec_check.log; then
  echo "FAIL: live Kinova expert episode did not succeed"
  FAIL=1
fi
if grep -q "Successful episode?:  False" kinova/data_smoke/exec_check.log; then
  echo "FAIL: live Kinova expert episode reported False"
  FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "SMOKE TEST FAILED"
  exit 1
fi
echo "SMOKE TEST PASSED: Kinova GT collection executes the task"
