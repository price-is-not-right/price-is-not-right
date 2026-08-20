#!/usr/bin/env bash
# Merge GT trace zips (e.g. 30 + 70 -> 100) into one traces/ folder.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT="${OUT:-kinova/data_gt/Hanoi_merged/hanoi_gt/traces}"
DIR1="${1:-kinova/data_gt/Hanoi_seed_0/hanoi_gt/traces}"
DIR2="${2:-kinova/data_gt/Hanoi_seed_1/hanoi_gt/traces}"

python3 kinova/scripts/merge_gt_traces.py "$DIR1" "$DIR2" --out "$OUT"
echo "Merged traces -> $OUT"
echo "Zarr: python data_processing/data_to_zarr.py --data_dir kinova/data_gt/Hanoi_merged/hanoi_gt --filter_actions True"
