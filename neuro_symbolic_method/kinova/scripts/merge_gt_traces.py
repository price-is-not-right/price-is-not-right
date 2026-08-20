#!/usr/bin/env python3
"""Merge skill trace zips from multiple auto_demo collection runs."""
import argparse
import pickle
import zipfile
from pathlib import Path


def load_traces(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("data.pkl") as f:
            return pickle.load(f)


def save_traces(zip_path: Path, data):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(data)
    with zipfile.ZipFile(zip_path, "w") as zf:
        with zf.open("data.pkl", "w", force_zip64=True) as f:
            f.write(payload)


def main():
    parser = argparse.ArgumentParser(description="Merge pick/place/reach_* trace zips")
    parser.add_argument("trace_dirs", nargs="+", help="Directories containing *.zip traces")
    parser.add_argument("--out", required=True, help="Output directory for merged zips")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.trace_dirs]
    out = Path(args.out)
    skills = sorted({p.stem for d in dirs for p in d.glob("*.zip")})
    if not skills:
        raise SystemExit(f"No zip traces found under {args.trace_dirs}")

    for skill in skills:
        merged = []
        for d in dirs:
            zp = d / f"{skill}.zip"
            if not zp.exists():
                print(f"  skip missing {zp}")
                continue
            buf = load_traces(zp)
            merged.extend(buf)
            print(f"  + {len(buf)} from {zp}")
        save_traces(out / f"{skill}.zip", merged)
        print(f"{skill}: {len(merged)} trajectories -> {out / f'{skill}.zip'}")


if __name__ == "__main__":
    main()
