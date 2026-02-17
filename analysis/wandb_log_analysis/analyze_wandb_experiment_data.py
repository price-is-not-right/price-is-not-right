#!/usr/bin/env python3
"""
Analyze W&B experiment CSVs for energy usage across runs.

    python analyze_wandb_experiment_data.py \
        --args.path data/experiments_exports/ICRA-3-Block-Hanoi-End-to-End_TRUE_FINAL
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Dict

try:
    import numpy as np
    import pandas as pd
    import tyro
except ImportError:
    print("❌ Missing dependencies. Run: pip install numpy pandas tyro")
    sys.exit(1)


@dataclasses.dataclass
class Args:
    path: Path = Path("data/experiments_exports/FINAL_Journal_Zero_Shot_2_Color_to_3_Color_NO_SWAP_AssemblyLineSorting")
    runtime_csv: Path | None = None
    boundary_mode: str = "hold"


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    time_col = "Relative Time (Process)"
    if time_col not in df.columns:
        time_col = df.columns[0]
    df = df.rename(columns={time_col: "time_s"})
    return df


def _prepare_series(time_s: np.ndarray, power_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Some exports contain duplicate timestamps; collapse them before integration.
    frame = pd.DataFrame({"time_s": time_s, "power_w": power_w}).dropna()
    frame = frame.groupby("time_s", as_index=False)["power_w"].mean().sort_values("time_s")
    return frame["time_s"].to_numpy(), frame["power_w"].to_numpy()


def _compute_energy_stats(
    time_s: np.ndarray,
    power_w: np.ndarray,
    runtime_s: float | None,
    boundary_mode: str,
) -> dict[str, float]:
    time_s, power_w = _prepare_series(time_s, power_w)
    if time_s.size < 2:
        return {"avg_power_w": float("nan"), "energy_j": float("nan"), "energy_wh": float("nan")}

    observed_start = float(time_s[0])
    observed_end = float(time_s[-1])
    observed_duration_s = observed_end - observed_start
    if observed_duration_s <= 0:
        return {"avg_power_w": float("nan"), "energy_j": float("nan"), "energy_wh": float("nan")}

    energy_j = float(np.trapezoid(power_w, time_s))

    runtime_is_valid = runtime_s is not None and np.isfinite(runtime_s) and runtime_s > 0
    missing_start_s = 0.0
    missing_end_s = 0.0
    duration_s = observed_duration_s

    if runtime_is_valid:
        runtime_s = float(runtime_s)
        missing_start_s = max(0.0, observed_start)
        missing_end_s = max(0.0, runtime_s - observed_end)
        if boundary_mode == "hold":
            energy_j += (power_w[0] * missing_start_s) + (power_w[-1] * missing_end_s)
            duration_s = runtime_s
        elif boundary_mode == "zero":
            duration_s = runtime_s
        else:
            duration_s = observed_duration_s

    energy_wh = energy_j / 3600.0
    avg_power_w = energy_j / duration_s if duration_s > 0 else float("nan")
    coverage = observed_duration_s / runtime_s if runtime_is_valid and runtime_s > 0 else 1.0
    return {
        "avg_power_w": avg_power_w,
        "energy_j": energy_j,
        "energy_wh": energy_wh,
        "duration_s": duration_s,
        "observed_duration_s": observed_duration_s,
        "missing_start_s": missing_start_s,
        "missing_end_s": missing_end_s,
        "coverage_ratio": coverage,
    }


def _load_runtime_overrides(runtime_csv: Path | None) -> Dict[str, float]:
    if runtime_csv is None or not runtime_csv.exists():
        return {}

    df = pd.read_csv(runtime_csv)
    if df.empty:
        return {}

    lower_cols = {c.lower(): c for c in df.columns}
    if "run" in lower_cols and "runtime_s" in lower_cols:
        run_col = lower_cols["run"]
        runtime_col = lower_cols["runtime_s"]
        out: Dict[str, float] = {}
        for _, row in df.iterrows():
            run = str(row[run_col]).strip()
            value = pd.to_numeric(row[runtime_col], errors="coerce")
            if run and np.isfinite(value) and value > 0:
                out[run] = float(value)
        return out

    # Fallback: wide CSV, one column per run.
    time_col = "Relative Time (Process)"
    if time_col not in df.columns:
        time_col = df.columns[0]
    run_cols = [c for c in df.columns if c != time_col]
    out: Dict[str, float] = {}
    for col in run_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        values = values[np.isfinite(values)]
        if values.size:
            out[col] = float(np.max(values))
    return out


def _analyze_metric_csv(
    csv_path: Path,
    runtime_by_run: Dict[str, float],
    boundary_mode: str,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    df = _load_csv(csv_path)
    if df.empty or "time_s" not in df.columns or len(df.columns) < 2:
        return [], {}

    df = df.sort_values("time_s")
    time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    results: list[dict[str, object]] = []

    for col in [c for c in df.columns if c != "time_s"]:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mask = np.isfinite(time_s) & np.isfinite(values)
        if mask.sum() < 2:
            continue
        stats = _compute_energy_stats(
            time_s[mask],
            values[mask],
            runtime_s=runtime_by_run.get(col),
            boundary_mode=boundary_mode,
        )
        results.append(
            {
                "file": csv_path.name,
                "run": col,
                "avg_power_w": stats["avg_power_w"],
                "energy_j": stats["energy_j"],
                "energy_wh": stats["energy_wh"],
                "duration_s": stats["duration_s"],
                "observed_duration_s": stats["observed_duration_s"],
                "missing_start_s": stats["missing_start_s"],
                "missing_end_s": stats["missing_end_s"],
                "coverage_ratio": stats["coverage_ratio"],
            }
        )

    if not results:
        return [], {}

    avg_power = float(np.mean([r["avg_power_w"] for r in results]))
    avg_energy_j = float(np.mean([r["energy_j"] for r in results]))
    avg_energy_wh = float(np.mean([r["energy_wh"] for r in results]))
    avg_duration = float(np.mean([r["duration_s"] for r in results]))
    total_energy_j = float(np.sum([r["energy_j"] for r in results]))
    total_duration_s = float(np.sum([r["duration_s"] for r in results]))
    pooled_avg_power = total_energy_j / total_duration_s if total_duration_s > 0 else float("nan")
    summary = {
        "avg_power_w": pooled_avg_power,
        "mean_run_avg_power_w": avg_power,
        "avg_energy_j": avg_energy_j,
        "avg_energy_wh": avg_energy_wh,
        "avg_duration_s": avg_duration,
        "total_energy_j": total_energy_j,
        "total_duration_s": total_duration_s,
        "runs": float(len(results)),
    }
    return results, summary


def main(args: Args) -> None:
    path = args.path
    if args.boundary_mode not in {"hold", "zero", "none"}:
        print("❌ boundary_mode must be one of: hold, zero, none")
        return

    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        output_dir = path
    elif path.is_file() and path.suffix.lower() == ".csv":
        csv_paths = [path]
        output_dir = path.parent
    else:
        print(f"❌ Path not found or not a CSV/directory: {path}")
        return

    if not csv_paths:
        print(f"⚠️  No CSV files found in: {path}")
        return

    runtime_csv = args.runtime_csv
    if runtime_csv is None and path.is_dir():
        for candidate in [
            path / "run_runtime_seconds.csv",
            path / "_runtime.csv",
            path / "runtime.csv",
        ]:
            if candidate.exists():
                runtime_csv = candidate
                break
    runtime_by_run = _load_runtime_overrides(runtime_csv)

    lines: list[str] = ["Experiment Energy Summary"]
    lines.append(f"boundary_mode: {args.boundary_mode}")
    if runtime_csv is not None and runtime_csv.exists():
        lines.append(f"runtime_source: {runtime_csv.name}")
    else:
        lines.append("runtime_source: none (using observed sample window only)")
    any_results = False

    for csv_path in csv_paths:
        per_run, summary = _analyze_metric_csv(csv_path, runtime_by_run=runtime_by_run, boundary_mode=args.boundary_mode)
        if not per_run:
            continue
        any_results = True
        lines.append("")
        lines.append(f"{csv_path.name}:")
        lines.append(f"  runs: {int(summary['runs'])}")
        lines.append(f"  avg_power_w:   {summary['avg_power_w']:.3f} W")
        lines.append(f"  mean_run_avg_power_w: {summary['mean_run_avg_power_w']:.3f} W")
        lines.append(f"  avg_energy_j:  {summary['avg_energy_j']:.3f} J")
        lines.append(f"  avg_energy_wh: {summary['avg_energy_wh']:.6f} Wh")
        lines.append(f"  avg_duration_s:{summary['avg_duration_s']:.3f} s")
        lines.append(f"  total_energy_j:{summary['total_energy_j']:.3f} J")
        lines.append(f"  total_duration_s:{summary['total_duration_s']:.3f} s")
        for r in per_run:
            lines.append(f"  {r['run']}")
            lines.append(f"    avg_power_w: {r['avg_power_w']:.3f} W")
            lines.append(f"    energy_j:    {r['energy_j']:.3f} J")
            lines.append(f"    energy_wh:   {r['energy_wh']:.6f} Wh")
            lines.append(f"    duration_s:  {r['duration_s']:.3f} s")
            lines.append(f"    observed_duration_s: {r['observed_duration_s']:.3f} s")
            lines.append(f"    missing_start_s: {r['missing_start_s']:.3f} s")
            lines.append(f"    missing_end_s: {r['missing_end_s']:.3f} s")
            lines.append(f"    coverage_ratio: {r['coverage_ratio']:.3f}")

    if not any_results:
        print("⚠️  No usable data found in CSVs.")
        return

    output_text = "\n".join(lines) + "\n"
    print(output_text, end="")

    output_path = output_dir / "experiment_energy_summary.txt"
    output_path.write_text(output_text)
    print(f"Saved summary: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
