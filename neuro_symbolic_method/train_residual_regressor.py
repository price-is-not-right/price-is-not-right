"""
Trains a lightweight residual-correction regressor that refines the analytic
stereo-triangulation estimate (from Executor_Diffusion.pixel_to_world_dual /
auto_demo.py's triangulate_dual) into the true 3D world position.

Motivation: analytic triangulation using exact camera geometry is extremely
accurate (<1mm) when given the *true* 3D centroid's pixel projection, but
YOLO's bounding-box center is only an approximation of that projection
(worse under occlusion/perspective), leaving a systematic residual error of
1-3cm. Rather than learning the full pixel->3D mapping from scratch (which
needs to implicitly learn 3D projective geometry and is what caused the
original ~1cm+ regressor error), this model only learns the small residual
correction, using BOTH camera detections (bbox center/size/confidence) and
the end-effector position as inputs alongside the triangulated estimate.

Feature order (16 features):
    tri_x, tri_y, tri_z,
    px_cam1, py_cam1, w_cam1, h_cam1, conf_cam1,
    px_cam2, py_cam2, w_cam2, h_cam2, conf_cam2,
    ee_x, ee_y, ee_z
Targets: world_x, world_y, world_z  (regressor predicts the residual
world - tri, which is added back to `tri` at inference time)

Usage:
    python train_residual_regressor.py --data_glob "data_tri_train/**/yolo_data/*.csv" \
        --out models/regressors/hanoi_residual_regressor.pkl
"""
import argparse
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

TRI_COLS = ["tri_x", "tri_y", "tri_z"]
BBOX_COLS = [
    "px_cam1", "py_cam1", "w_cam1", "h_cam1", "conf_cam1",
    "px_cam2", "py_cam2", "w_cam2", "h_cam2", "conf_cam2",
]
EE_COLS = ["ee_x", "ee_y", "ee_z"]
FEATURE_COLS = TRI_COLS + BBOX_COLS + EE_COLS
TARGET_COLS = ["world_x", "world_y", "world_z"]


def load_data(patterns, max_rows=None, dedupe=True, seed=0):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"No CSV files matched patterns: {patterns}")
    print(f"Loading {len(files)} CSV files...")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    before = len(df)
    # Only keep rows where triangulation actually succeeded (both cameras saw the object).
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    if dedupe:
        df = df.drop_duplicates(subset=FEATURE_COLS + TARGET_COLS)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    print(f"Loaded {before} rows, {len(df)} usable (both-camera + triangulated) rows, from {len(files)} files")
    return df


def train(df, n_estimators=300, max_depth=3, learning_rate=0.05, test_size=0.1, seed=0):
    X = df[FEATURE_COLS].astype(np.float64).values
    tri = df[TRI_COLS].astype(np.float64).values
    models = {}
    metrics = {}
    for i, (axis, col) in enumerate(zip(["res_x", "res_y", "res_z"], TARGET_COLS)):
        y_true = df[col].astype(np.float64).values
        y_resid = y_true - tri[:, i]
        X_train, X_test, yr_train, yr_test, tri_train, tri_test = train_test_split(
            X, y_resid, tri[:, i], test_size=test_size, random_state=seed
        )
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        model.fit(X_train, yr_train)
        resid_pred = model.predict(X_test)
        final_pred = tri_test + resid_pred
        final_true = tri_test + yr_test
        mae = np.mean(np.abs(final_pred - final_true))
        rmse = np.sqrt(np.mean((final_pred - final_true) ** 2))
        raw_mae = np.mean(np.abs(tri_test - final_true))  # error of triangulation alone (no correction)
        metrics[axis] = {"mae": mae, "rmse": rmse, "raw_tri_mae": raw_mae}
        print(f"{axis} ({col}): corrected MAE={mae:.5f}  RMSE={rmse:.5f}  "
              f"| raw triangulation MAE={raw_mae:.5f}  (n_train={len(X_train)}, n_test={len(X_test)})")
        models[axis] = model
    return models, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_glob", type=str, nargs="+",
                         default=["data_tri_train/**/yolo_data/*.csv"],
                         help="Glob pattern(s) for input CSV files (supports ** with recursive)")
    parser.add_argument("--out", type=str, default="models/regressors/hanoi_residual_regressor.pkl")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=200000,
                        help="Subsample cap (GradientBoosting is slow on >1M rows)")
    args = parser.parse_args()

    df = load_data(args.data_glob, max_rows=args.max_rows, seed=args.seed)
    models, metrics = train(
        df,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        seed=args.seed,
    )
    joblib.dump({"models": models, "feature_cols": FEATURE_COLS, "metrics": metrics}, args.out)
    print(f"Saved residual regressor to {args.out}")


if __name__ == "__main__":
    main()
