"""Diagnostic: does a trained Kinova GT policy reproduce its own training data?

Prints dataset obs/action statistics and the open-loop action error of the
checkpoint when replayed on observations drawn from the training zarr.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diffusion_policy"))

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402
import dill  # noqa: E402
import hydra  # noqa: E402


def stats(name, arr):
    arr = np.asarray(arr)
    print(f"{name}: shape={arr.shape}")
    print(f"  min  {np.round(arr.min(axis=0), 4)}")
    print(f"  max  {np.round(arr.max(axis=0), 4)}")
    print(f"  mean {np.round(arr.mean(axis=0), 4)}")
    print(f"  std  {np.round(arr.std(axis=0), 4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    root = zarr.open(args.zarr, mode="r")
    obs = np.array(root["data"]["keypoint"]) if "keypoint" in root["data"] else np.array(root["data"]["obs"])
    act = np.array(root["data"]["action"])
    ends = np.array(root["meta"]["episode_ends"])
    print(f"=== zarr {args.zarr}")
    print(f"episodes={len(ends)} steps={len(act)}")
    obs = obs.reshape(len(obs), -1)
    stats("dataset obs", obs)
    stats("dataset action", act)

    payload = torch.load(args.ckpt, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    print(f"\n=== ckpt {args.ckpt}")
    print(f"epoch={payload.get('_output_dir', '')} keys={list(payload.keys())}")
    print(f"cfg obs_dim={cfg.task.get('obs_dim')} action_dim={cfg.task.get('action_dim')} "
          f"horizon={cfg.horizon} n_obs={cfg.n_obs_steps} n_act={cfg.n_action_steps}")

    ws_cls = hydra.utils.get_class(cfg._target_)
    ws = ws_cls(cfg)
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if cfg.training.get("use_ema", False) else ws.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()

    n_obs = cfg.n_obs_steps
    n_act = cfg.n_action_steps
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(obs) - (n_obs + n_act) - 1, size=args.n)

    print("\n=== open-loop replay on training data")
    errs = []
    for s in starts:
        o = obs[s:s + n_obs][None].astype(np.float32)
        with torch.no_grad():
            pred = policy.predict_action({"obs": torch.from_numpy(o).to(device)})
        pred = pred["action"].cpu().numpy()[0]
        gt = act[s + n_obs - 1: s + n_obs - 1 + len(pred)]
        m = min(len(pred), len(gt))
        err = np.abs(pred[:m] - gt[:m]).mean()
        errs.append(err)
        print(f"  t={s:6d} obs={np.round(o[0, -1], 4)}")
        print(f"        pred[0]={np.round(pred[0], 4)}  gt[0]={np.round(gt[0], 4)}  |mae|={err:.4f}")
    print(f"\nmean |action| in data: {np.abs(act).mean():.4f}")
    print(f"mean MAE pred vs gt   : {np.mean(errs):.4f}")


if __name__ == "__main__":
    main()
