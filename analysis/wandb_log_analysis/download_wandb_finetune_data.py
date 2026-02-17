#!/usr/bin/env python3
"""
Download W&B metrics (system events or custom history) into CSVs.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
import difflib

try:
    import pandas as pd
    import wandb
    import tyro
except ImportError:
    print("❌ Missing dependencies. Run: pip install pandas wandb tyro")
    sys.exit(1)


@dataclasses.dataclass
class Args:
    project: str = "openpi"
    run_name: str = "hanoi_50_ja_no_mask_50ah"
    metrics: str = "system/gpu.0.powerWatts,cpu_power_watts"
    out_dir: Path = Path(f"data/finetuning_exports/{run_name}/")
    list_metrics: bool = False
    debug: bool = True


def _fetch_history(run, stream: str, samples: int) -> pd.DataFrame | None:
    try:
        return run.history(stream=stream, samples=samples)
    except Exception:
        return None


def _list_all_metrics(run, debug: bool) -> list[str]:
    events_df = _fetch_history(run, stream="events", samples=100000)
    default_df = _fetch_history(run, stream="default", samples=100000)
    columns: set[str] = set()
    if events_df is not None and not events_df.empty:
        columns.update([c for c in events_df.columns if not c.startswith("_")])
    if default_df is not None and not default_df.empty:
        columns.update([c for c in default_df.columns if not c.startswith("_")])
    if debug:
        print(f"  ℹ️  Found {len(columns)} metrics")
    return sorted(columns)


def _print_metric_hints(metric_key: str, columns: list[str], debug: bool) -> None:
    if not debug:
        return
    matches = difflib.get_close_matches(metric_key, columns, n=5, cutoff=0.4)
    print(f"  ⚠️  Metric '{metric_key}' (or variants) not found.")
    if matches:
        print("  ℹ️  Closest matches:")
        for m in matches:
            print(f"     - {m}")
    else:
        sample = ", ".join(columns[:20])
        if sample:
            print(f"  ℹ️  Sample available keys: {sample}")


def get_event_series(run, metric_key: str, debug: bool) -> pd.DataFrame | None:
    """
    Searches for a metric in the 'events' stream (system) AND 'default' stream (user logs).
    """
    # 1. Define where to look
    streams_to_check = ["events", "default"]
    
    df = None
    target_key = None
    
    # 2. Search both streams for the key
    for stream in streams_to_check:
        temp_df = _fetch_history(run, stream=stream, samples=100000)
        
        if temp_df is None or temp_df.empty:
            continue
            
        # Check 1: Exact Match
        if metric_key in temp_df.columns:
            df = temp_df
            target_key = metric_key
            if debug: print(f"  ℹ️  Found '{metric_key}' in '{stream}' stream.")
            break
            
        # Check 2: Auto-correct (System metrics often swap / for .)
        alt_key_dot = metric_key.replace("/", ".")
        if alt_key_dot in temp_df.columns:
            df = temp_df
            target_key = alt_key_dot
            if debug: print(f"  ℹ️  Mapping '{metric_key}' -> '{alt_key_dot}' (in '{stream}')")
            break
            
        alt_key_slash = metric_key.replace(".", "/")
        if alt_key_slash in temp_df.columns:
            df = temp_df
            target_key = alt_key_slash
            if debug: print(f"  ℹ️  Mapping '{metric_key}' -> '{alt_key_slash}' (in '{stream}')")
            break

    # 3. If still not found, show hints and exit
    if df is None or target_key is None:
        # For hints, we just grab columns from the default stream as a best guess
        hint_df = _fetch_history(run, stream="default", samples=1000)
        cols = list(hint_df.columns) if hint_df is not None else []
        _print_metric_hints(metric_key, cols, debug)
        return None

    # 4. Handle Time Alignment (Use _runtime if available)
    if "_runtime" in df.columns:
        df = df.sort_values("_runtime")
        time_index = pd.to_timedelta(df["_runtime"], unit="s")
    elif "_timestamp" in df.columns:
        df = df.sort_values("_timestamp")
        start_time = df["_timestamp"].min()
        rel_time = df["_timestamp"] - start_time
        time_index = pd.to_timedelta(rel_time, unit="s")
    else:
        print("  ❌ Error: No time column (_runtime or _timestamp) found.")
        return None

    df.index = time_index
    
    # 5. Clean Data
    series = df[target_key].dropna()
    series = pd.to_numeric(series, errors='coerce').dropna()

    if series.empty:
        return None

    # Return as DataFrame
    series = series.to_frame(name=metric_key)
    series.index.name = "Relative Time (Process)"
    
    return series


def download_wandb_history(
    project: str,
    metric_keys: list[str],
    out_dir: Path,
    run_name_filter: str,
    debug: bool,
    list_metrics: bool,
) -> None:
    
    api = wandb.Api()
    runs = api.runs(project)
    
    if run_name_filter:
        runs = [r for r in runs if run_name_filter in (r.name, r.id)]
    
    if not runs:
        print(f"❌ No runs found matching '{run_name_filter}' in '{project}'")
        return

    if list_metrics:
        for run in runs:
            r_name = run.name or run.id
            print(f"\n--- Metrics for Run: {r_name} ---")
            metrics = _list_all_metrics(run, debug)
            for m in metrics:
                print(m)
        return

    print(f"Found {len(runs)} runs. Processing metrics: {metric_keys}...")
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in metric_keys:
        print(f"\n--- Processing Metric: {metric} ---")
        all_run_dfs = []

        for run in runs:
            r_name = run.name or run.id
            if debug:
                print(f"Scanning run: {r_name}...")
            
            stats_df = get_event_series(run, metric, debug)
            
            if stats_df is not None:
                # Use the original requested name for columns to keep CSV pretty
                base_col = f"{r_name} - {metric}"
                stats_df.columns = [base_col]
                all_run_dfs.append(stats_df)
                print(f"  ✔️  Captured {len(stats_df)} samples")
            else:
                if debug:
                    print(f"  -> No data.")

        if not all_run_dfs:
            print("  ⚠️ No data captured for this metric.")
            continue

        # Merge and align
        merged_df = pd.concat(all_run_dfs, axis=1)
        
        # Convert Timedelta Index to Seconds (Float) for CSV
        merged_df.index = merged_df.index.total_seconds()
        merged_df.index.name = "Relative Time (Process)"
        merged_df = merged_df.sort_index()

        safe_name = metric.replace("/", "_") + ".csv"
        out_path = out_dir / safe_name
        merged_df.to_csv(out_path)
        print(f"✔️  Saved {safe_name} | Shape: {merged_df.shape}")


# Need to make args available to helper function, simple way:
args = None 

def main(args: Args) -> None:
    metric_keys = [m.strip() for m in args.metrics.split(",")]
    
    download_wandb_history(
        project=args.project,
        metric_keys=metric_keys,
        out_dir=args.out_dir,
        run_name_filter=args.run_name,
        debug=args.debug,
        list_metrics=args.list_metrics,
    )


if __name__ == "__main__":
    tyro.cli(main)
