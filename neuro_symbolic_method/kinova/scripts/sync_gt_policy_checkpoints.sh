#!/usr/bin/env bash
# Copy latest (or best-by-train_loss) GT diffusion checkpoints into
# kinova/models/policies/gt/ for eval via kinova/configs/hanoi_gt.yaml.
#
# Usage (from neuro_symbolic_method/):
#   bash kinova/scripts/sync_gt_policy_checkpoints.sh
#   bash kinova/scripts/sync_gt_policy_checkpoints.sh latest
#   bash kinova/scripts/sync_gt_policy_checkpoints.sh best
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-latest}"  # latest | best

python3 - <<PY
from pathlib import Path
import re
import shutil
import sys

root = Path("kinova/data/train")
dst = Path("kinova/models/policies/gt")
dst.mkdir(parents=True, exist_ok=True)
mode = "${MODE}"
if mode not in ("latest", "best"):
    raise SystemExit(f"Unknown mode {mode!r}; expected 'latest' or 'best'")

mapping = {
    "train_kinova_gt_grasp": "grasp.ckpt",
    "train_kinova_gt_drop": "drop.ckpt",
    "train_kinova_gt_reach_pick": "reach_pick.ckpt",
    "train_kinova_gt_reach_place": "reach_place.ckpt",
}

def pick_source(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing {ckpt_dir}")
    latest = ckpt_dir / "latest.ckpt"
    if mode == "latest":
        if latest.exists():
            return latest
        raise FileNotFoundError(f"No latest.ckpt in {ckpt_dir}")
    best_path, best_loss = None, float("inf")
    for p in ckpt_dir.glob("epoch=*-train_loss=*.ckpt"):
        m = re.search(r"train_loss=(\d+(?:\.\d+)?)", p.name)
        if not m:
            continue
        loss = float(m.group(1))
        if loss < best_loss:
            best_loss, best_path = loss, p
    if best_path is not None:
        return best_path
    if latest.exists():
        print(f"WARNING: no epoch=*-train_loss=*.ckpt in {ckpt_dir}; using latest.ckpt")
        return latest
    raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

for run, name in mapping.items():
    src = pick_source(root / run)
    out = dst / name
    shutil.copy2(src, out)
    m = re.search(r"epoch=(\d+)", src.name)
    ep = m.group(1) if m else "latest"
    print(f"{name}: {src} (epoch {ep}) -> {out}")
PY

echo "Synced GT policies to kinova/models/policies/gt/ (mode=${MODE})"
