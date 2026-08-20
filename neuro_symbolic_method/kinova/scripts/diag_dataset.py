"""Coherence checks on a collected Kinova skill dataset (zarr).

For each skill we assert the demonstrations actually do what their name says:
reach skills should drive the relative target offset to ~0, and grasp/drop
skills should end with the gripper closed/open respectively.
"""
import argparse
import pathlib

import numpy as np
import zarr


def episode_slices(ends):
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts, ends))


def describe(skill, path, obs_labels, act_labels):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"\n### {skill}: MISSING {path}")
        return
    r = zarr.open(str(p), mode="r")
    state = np.array(r["data"]["state"])
    action = np.array(r["data"]["action"])
    ends = np.array(r["meta"]["episode_ends"])
    eps = episode_slices(ends)
    lens = np.array([e - s for s, e in eps])

    print(f"\n### {skill}  ({path})")
    print(f"episodes={len(eps)}  steps={len(state)}  "
          f"len min/med/max = {lens.min()}/{int(np.median(lens))}/{lens.max()}")
    print(f"obs dims  {obs_labels}")
    print(f"  start mean {np.round(np.array([state[s] for s, _ in eps]).mean(0), 4)}")
    print(f"  end   mean {np.round(np.array([state[e-1] for _, e in eps]).mean(0), 4)}")
    print(f"  end   |max| {np.round(np.abs(np.array([state[e-1] for _, e in eps])).max(0), 4)}")
    print(f"act dims  {act_labels}")
    print(f"  min {np.round(action.min(0), 3)}  max {np.round(action.max(0), 3)}")
    print(f"  nan/inf in state={np.isnan(state).any() or np.isinf(state).any()} "
          f"action={np.isnan(action).any() or np.isinf(action).any()}")

    finals = np.array([state[e-1] for _, e in eps])
    if skill in ("reach_pick", "reach_place"):
        dist = np.linalg.norm(finals[:, :3], axis=1)
        xy = np.linalg.norm(finals[:, :2], axis=1)
        print(f"  final |offset| xyz: mean={dist.mean():.4f} max={dist.max():.4f}")
        print(f"  final |offset| xy : mean={xy.mean():.4f} max={xy.max():.4f} "
              f"| frac under 5mm = {(xy < 0.005).mean():.2f}")
    else:
        print(f"  final dz    : mean={finals[:, 0].mean():.4f} max|{np.abs(finals[:, 0]).max():.4f}|")
        print(f"  final apert : mean={finals[:, 1].mean():.4f} "
              f"min={finals[:, 1].min():.4f} max={finals[:, 1].max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to .../hf_traj")
    args = ap.parse_args()
    base = pathlib.Path(args.base)
    specs = [
        ("reach_pick", "reach_pick", ["dx", "dy", "dz"], ["ax", "ay", "az"]),
        ("pick", "pick", ["dz_obj", "aperture"], ["az", "grip"]),
        ("reach_place", "reach_place", ["dx", "dy", "dz"], ["ax", "ay", "az"]),
        ("place", "place", ["dz_drop", "aperture"], ["az", "grip"]),
    ]
    for skill, sub, ol, al in specs:
        describe(skill, base / sub / "keypoint" / "keypoint.zarr", ol, al)


if __name__ == "__main__":
    main()
