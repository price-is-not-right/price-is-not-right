#!/usr/bin/env python3
"""
Analyze W&B-exported CSVs for average power and total energy.

Usage:
python analyze_wandb_finetune_data.py --args.path data/finetuning_exports/assembly_line_sorting
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import tyro
except ImportError:
    print("❌ Missing dependencies. Run: pip install numpy pandas tyro")
    sys.exit(1)


@dataclasses.dataclass
class Args:
    path: Path = Path("data/finetuning_exports/hanoi_3x3")


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    time_col = "Relative Time (Process)"
    if time_col not in df.columns:
        time_col = df.columns[0]
    df = df.rename(columns={time_col: "time_s"})
    return df


def _compute_energy_stats(time_s: np.ndarray, power_w: np.ndarray) -> dict[str, float]:
    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0:
        return {"avg_power_w": float("nan"), "energy_j": float("nan"), "energy_wh": float("nan")}
    energy_j = float(np.trapezoid(power_w, time_s))
    energy_wh = energy_j / 3600.0
    avg_power_w = energy_j / duration_s
    return {"avg_power_w": avg_power_w, "energy_j": energy_j, "energy_wh": energy_wh}


def _analyze_csv(csv_path: Path) -> list[dict[str, object]]:
    df = _load_csv(csv_path)
    if df.empty or "time_s" not in df.columns or len(df.columns) < 2:
        return []

    df = df.sort_values("time_s")
    time_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    results: list[dict[str, object]] = []

    for col in [c for c in df.columns if c != "time_s"]:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mask = np.isfinite(time_s) & np.isfinite(values)
        if mask.sum() < 2:
            continue
        stats = _compute_energy_stats(time_s[mask], values[mask])
        results.append(
            {
                "file": csv_path.name,
                "series": col,
                "avg_power_w": stats["avg_power_w"],
                "energy_j": stats["energy_j"],
                "energy_wh": stats["energy_wh"],
                "duration_s": float(time_s[mask][-1] - time_s[mask][0]),
            }
        )
    return results


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m {secs}s"


def main(args: Args) -> None:
    path = args.path
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

    all_results: list[dict[str, object]] = []
    for csv_path in csv_paths:
        all_results.extend(_analyze_csv(csv_path))

    if not all_results:
        print("⚠️  No usable data found in CSVs.")
        return

    lines = ["Average Power and Total Energy (per series)"]
    current_file = None
    for r in all_results:
        if current_file != r["file"]:
            if current_file is not None:
                lines.append("")
            current_file = r["file"]
            lines.append(f"{current_file}:")
        lines.append(f"  {r['series']}")
        lines.append(f"    avg_power_w: {r['avg_power_w']:.3f} W")
        lines.append(f"    energy_j:    {r['energy_j']:.3f} J")
        lines.append(f"    energy_wh:   {r['energy_wh']:.6f} Wh")
        lines.append(f"    duration_s:  {r['duration_s']:.3f} s")
        lines.append(f"    duration:    {_format_duration(r['duration_s'])}")

    output_text = "\n".join(lines) + "\n"
    print(output_text, end="")

    output_path = output_dir / "power_energy_summary.txt"
    output_path.write_text(output_text)
    print(f"Saved summary: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
