
#!/usr/bin/env python3
"""
download_wandb_experiment_data.py
---------------------------------
Download **all runs** from a Weights & Biases project and save one **wide‑format**
CSV per metric key you request.

Each CSV has:
    • rows → relative time in seconds
    • cols → run names / ids

Example
-------
    python download_wandb_experiment_data.py \
        --args.project FINAL_Kinova3_Journal_Paper_CubeSorting \
        --args.metrics system/gpu.0.powerWatts

    python analysis/wandb_log_analysis/download_wandb_experiment_data.py \
        --project tim-duggan93-tufts-university/TRUE_FINAL_pi0_hanoi_300_one_task_3_blocks_random_selection \
        --metrics "*" \
        --out-dir analysis/wandb_log_analysis/data/experiments_exports/TRUE_FINAL_pi0_hanoi_300_one_task_3_blocks_random_selection

Arguments
---------
--project      entity/project (required)
--metrics      Comma‑separated list of metric keys,
               or '*' to grab every key present in any run.
--out-dir      Output directory (default: current working dir)

Tip
---
    python analysis/wandb_log_analysis/download_wandb_experiment_data.py --help
"""
from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
import tyro


# ----------------------------------------------------------------------------- #
def download_wandb_csvs(project: str,
                        metric_keys: List[str],
                        out_dir: Path) -> None:
    """Stream `metric_keys` for every run and write one wide CSV per metric."""
    try:
        import wandb
    except ModuleNotFoundError:
        raise SystemExit("❌  `wandb` not installed.  Run `pip install wandb` and retry.")

    api = wandb.Api()
    runs = api.runs(project)
    if not runs:
        raise SystemExit(f"❌  No runs found for project '{project}'")

    # Wildcard "*" → collect every key that appears in the runs
    if metric_keys == ["*"]:
        metric_set = set()
        for run in tqdm(runs, desc="Scanning runs", unit="run"):
            # scan_history(keys=[]) gives dicts of *all* keys
            for row in run.scan_history(keys=[]):
                metric_set.update([k for k in row.keys() if not k.startswith("_")])
        metric_keys = sorted(metric_set)
        print(f"Found {len(metric_keys)} distinct metric keys")

    # initialise one empty wide DF per metric
    dfs: Dict[str, pd.DataFrame] = {m: pd.DataFrame() for m in metric_keys}
    runtime_rows: list[dict[str, object]] = []

    print(f"Downloading {len(runs)} runs × {len(metric_keys)} metrics …")
    for run in tqdm(runs, desc="Runs", unit="run"):
        run_name = run.name or run.id
        series_by_metric: Dict[str, List[tuple[float, float]]] = {m: [] for m in metric_keys}
        start_ts = None
        max_runtime_seen = None
        for row in run.scan_history(keys=metric_keys + ["_runtime", "_timestamp"]):
            time_s = row.get("_runtime")
            if time_s is None:
                ts = row.get("_timestamp")
                if ts is not None:
                    if start_ts is None:
                        start_ts = ts
                    time_s = ts - start_ts
            if time_s is None:
                continue
            if max_runtime_seen is None or time_s > max_runtime_seen:
                max_runtime_seen = float(time_s)
            for m in metric_keys:
                if m in row:
                    series_by_metric[m].append((float(time_s), row[m]))

        runtime_s = run.summary.get("_runtime")
        if runtime_s is None:
            runtime_s = max_runtime_seen
        runtime_rows.append(
            {
                "run": run_name,
                "run_id": run.id,
                "runtime_s": runtime_s,
            }
        )

        for m, points in series_by_metric.items():
            if not points:
                continue
            times, values = zip(*points)
            run_series = pd.Series(values, index=times, name=run_name)
            run_series = run_series.sort_index()
            if dfs[m].empty:
                dfs[m] = run_series.to_frame()
            else:
                dfs[m] = dfs[m].join(run_series, how="outer")

    # sort index (relative time) numerically
    for df in dfs.values():
        df.sort_index(inplace=True)
        df.index.name = "Relative Time (Process)"

    out_dir.mkdir(parents=True, exist_ok=True)
    for m, df in dfs.items():
        if df.empty:
            print(f"⚠️  No data for key '{m}', skipping.")
            continue
        fn = out_dir / (m.replace("/", "_") + ".csv")
        df.to_csv(fn, index=True)
        print(f"✔️  Saved {fn}  (rows={df.shape[0]}, runs={df.shape[1]})")

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(out_dir / "run_runtime_seconds.csv", index=False)
    print(f"✔️  Saved {out_dir / 'run_runtime_seconds.csv'} (runs={len(runtime_rows)})")


# ----------------------------------------------------------------------------- #
@dataclasses.dataclass
class Args:
    project: str
    metrics: str = "system/gpu.0.powerWatts"
    out_dir: Path | None = None


def main(args: Args) -> None:
    metric_keys = [m.strip() for m in args.metrics.split(",")]
    out_dir = args.out_dir or Path(f"data/experiments_exports/{args.project}")
    download_wandb_csvs(args.project, metric_keys, out_dir)


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    tyro.cli(main)
