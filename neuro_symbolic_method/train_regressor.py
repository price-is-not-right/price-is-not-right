"""
Trains the Hanoi dual-camera + EE pixel-to-world regressor used by
Executor_Diffusion.pixel_to_world_dual.

Feature order (13 features):
    px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z
Targets: world_x, world_y, world_z

Writes a versioned file and optionally updates the active symlink/copy path.
Never overwrites an existing --out without --force; prefer versioned --out.

Usage:
    python train_regressor.py \
        --data_glob "data_tri_train/**/yolo_data/*.csv" "data/**/yolo_data/*.csv" \
        --out models/regressors/archive/hanoi_regressor_v2.pkl \
        --active models/regressors/hanoi_regressor.pkl
"""
import argparse
import glob
import os
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "px_cam1", "py_cam1", "w_cam1", "h_cam1", "conf_cam1",
    "px_cam2", "py_cam2", "w_cam2", "h_cam2", "conf_cam2",
    "ee_x", "ee_y", "ee_z",
]
TARGET_COLS = ["world_x", "world_y", "world_z"]


def load_data(patterns, max_rows=None, dedupe=True):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"No CSV files matched patterns: {patterns}")
    print(f"Loading {len(files)} CSV files...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  skip {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    # Require agentview detection (cam1); wrist may be missing (zeros).
    df = df[(df["w_cam1"] > 0) & (df["h_cam1"] > 0)]
    if dedupe:
        df = df.drop_duplicates(subset=FEATURE_COLS + TARGET_COLS)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0)
    print(f"Loaded {before} rows -> {len(df)} usable (dedupe={dedupe})")
    return df


def train(df, max_iter=400, max_depth=8, learning_rate=0.05, test_size=0.1, seed=0):
    X = df[FEATURE_COLS].astype(np.float64).values
    models = {}
    metrics = {}
    for axis, col in zip(["reg_x", "reg_y", "reg_z"], TARGET_COLS):
        y = df[col].astype(np.float64).values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            l2_regularization=1e-3,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = float(np.mean(np.abs(preds - y_test)))
        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        # Also report 3D-relevant scale in mm
        print(f"{axis} ({col}): MAE={mae:.5f}m ({mae*1000:.2f}mm)  RMSE={rmse:.5f}m")
        metrics[axis] = {"mae": mae, "rmse": rmse}
        models[axis] = model

    # Joint 3D error on held-out split (same indices via re-split with seed)
    _, X_te, yx_tr, yx_te = train_test_split(
        X, df["world_x"].values, test_size=test_size, random_state=seed
    )
    _, _, yy_tr, yy_te = train_test_split(
        X, df["world_y"].values, test_size=test_size, random_state=seed
    )
    _, _, yz_tr, yz_te = train_test_split(
        X, df["world_z"].values, test_size=test_size, random_state=seed
    )
    pred = np.stack([
        models["reg_x"].predict(X_te),
        models["reg_y"].predict(X_te),
        models["reg_z"].predict(X_te),
    ], axis=1)
    true = np.stack([yx_te, yy_te, yz_te], axis=1)
    dist = np.linalg.norm(pred - true, axis=1)
    print(f"3D error: mean={dist.mean()*1000:.2f}mm  median={np.median(dist)*1000:.2f}mm  "
          f"p90={np.percentile(dist,90)*1000:.2f}mm  p99={np.percentile(dist,99)*1000:.2f}mm")
    metrics["l2"] = {
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
    }
    return models, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_glob", type=str, nargs="+",
                        default=["data_tri_train/**/yolo_data/*.csv",
                                 "data/**/yolo_data/*.csv"])
    parser.add_argument("--out", type=str, default=None,
                        help="Versioned output path (default: archive/hanoi_regressor_<ts>.pkl)")
    parser.add_argument("--active", type=str,
                        default="models/regressors/hanoi_regressor.pkl",
                        help="Path to copy the new model to for live use (previous active is archived)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=400000)
    args = parser.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"models/regressors/archive/hanoi_regressor_{ts}.pkl"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out) and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.out} (pass --force)")

    df = load_data(args.data_glob, max_rows=args.max_rows)
    models, metrics = train(
        df,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        seed=args.seed,
    )
    joblib.dump({"reg_x": models["reg_x"], "reg_y": models["reg_y"], "reg_z": models["reg_z"],
                 "metrics": metrics, "feature_cols": FEATURE_COLS},
                args.out)
    print(f"Saved versioned regressor to {args.out}")

    if args.active:
        if os.path.exists(args.active):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = f"models/regressors/archive/hanoi_regressor_active_backup_{ts}.pkl"
            os.makedirs(os.path.dirname(bak), exist_ok=True)
            shutil.copy2(args.active, bak)
            print(f"Archived previous active model to {bak}")
        # Support both bare dict-of-regs and wrapped payload at load time in executor
        # Executor expects models_dual["reg_x"] etc. — dump compatible format:
        joblib.dump(models, args.active)
        print(f"Installed active regressor at {args.active}")


if __name__ == "__main__":
    main()
