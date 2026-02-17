#!/usr/bin/env python3
import sys, io, pickle
from itertools import islice
from pathlib import Path
import numpy as np
import numpy.lib.format as nf
from PIL import Image
import os

INSPECTION_DIR = Path(__file__).parent / "inspection_videos/npy"
INSPECTION_DIR.mkdir(exist_ok=True)

class NumpyCoreRedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)
    

def save_frames_as_gif(frames, out_path, duration=100):
    """frames: list of np.ndarray, out_path: str"""
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )

def load_npy_object_array(path):
    with open(path, 'rb') as f:
        version = nf.read_magic(f)
        shape, fortran_order, dtype = nf._read_array_header(f, version)
        return NumpyCoreRedirectUnpickler(f).load()

def find_image_key(step):
    # Patch: recursively search for array images inside dicts
    def search_dict(d, prefix=''):
        if not isinstance(d, dict):
            return None
        for k, v in d.items():
            if hasattr(v, 'shape') and len(v.shape) >= 2 and np.issubdtype(v.dtype, np.number):
                return prefix + k
            if isinstance(v, dict):
                found = search_dict(v, prefix + k + '.')
                if found: return found
        return None
    return search_dict(step)

def get_image_by_key(step, image_key):
    # Support 'foo.bar' dot-paths
    keys = image_key.split('.')
    v = step
    for k in keys:
        v = v[k]
    return v

def save_episode_video(steps, image_key, save_video_path, max_frames=30):
    try:
        from PIL import Image
    except ImportError:
        print("  [WARN] PIL not installed, skipping gif save.")
        return
    frames = []
    for i, step in enumerate(steps[:max_frames]):
        img = get_image_by_key(step, image_key)
        if img.ndim == 2:
            out_img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            out_img = np.repeat(img, 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            out_img = img[..., :3]
        else:
            print(f"  Skipping weird image shape at step {i}: {img.shape}")
            continue
        if np.issubdtype(out_img.dtype, np.floating) and out_img.max() <= 1.01:
            out_img = (out_img * 255).astype(np.uint8)
        else:
            out_img = out_img.astype(np.uint8)
        frames.append(out_img)
    if not frames:
        print("  [WARN] No frames to save for gif.")
        return
    # Always save to inspection_videos with the same episode basename
    basename = Path(save_video_path).stem  # e.g. 'episode_0'
    gif_path = INSPECTION_DIR / f"{basename}.gif"
    try:
        save_frames_as_gif(frames, gif_path, duration=100)
        print(f"  [Saved gif to {gif_path}]")
    except Exception as e:
        print(f"  [ERROR] Could not create gif file: {e}")


def inspect_episode(path):
    print(f"\n=== Inspecting {path.name} ===")
    try:
        ep = load_npy_object_array(path)
        if isinstance(ep, np.ndarray) and ep.shape == ():
            ep = ep.item()
    except Exception as e:
        print(f"Failed to load {path.name}: {e}")
        return

    # Handle RLDS dict or list
    if isinstance(ep, dict) and "steps" in ep:
        print(f"  Format: dict with {len(ep['steps'])} steps")
        steps = ep["steps"]
        step0 = steps[0]
    elif isinstance(ep, (list, np.ndarray)):
        print(f"  Format: {type(ep)} with {len(ep)} steps")
        steps = ep
        step0 = ep[0]
    else:
        print(f"  Unknown episode structure: {type(ep)}")
        return

    # Print per-key shape/dtype/min/max for step 0
    print("  Step 0 keys/stats:")
    if isinstance(step0, dict):
        for k, v in step0.items():
            if hasattr(v, 'shape'):
                stats = f"shape={v.shape}, dtype={v.dtype}"
                if np.issubdtype(v.dtype, np.number):
                    stats += f", min={np.min(v):.2f}, max={np.max(v):.2f}"
                print(f"    {k!r}: {stats}")
            else:
                print(f"    {k!r}: type={type(v)}  {v}")
    else:
        print(f"    [WARN] Step 0 is not a dict, type={type(step0)}")

    # Summary for all steps to verify RLDS conversion correctness:
    expected_keys = {'observation', 'action', 'discount', 'reward', 'is_first', 'is_last', 'is_terminal', 'language_instruction'}
    all_keys_found = set()
    missing_keys_per_step = []
    print("\n  Summary for first 5 steps:")
    for i, step in enumerate(steps[:5]):
        if not isinstance(step, dict):
            print(f"    Step {i}: Not a dict, type={type(step)}")
            missing_keys_per_step.append(expected_keys)
            continue
        step_keys = set(step.keys())
        all_keys_found.update(step_keys)
        missing_keys = expected_keys - step_keys
        missing_keys_per_step.append(missing_keys)
        print(f"    Step {i}: keys = {sorted(step_keys)}")
        # For observation key, print sub-keys and their array shape/dtype
        if 'observation' in step and isinstance(step['observation'], dict):
            print(f"      observation sub-keys:")
            for ok in sorted(step['observation'].keys()):
                ov = step['observation'][ok]
                if hasattr(ov, 'shape') and hasattr(ov, 'dtype'):
                    print(f"        {ok}: shape={ov.shape}, dtype={ov.dtype}")
                else:
                    print(f"        {ok}: type={type(ov)}")
        # For other keys, print their type/value for first 2 steps
        for k in step_keys:
            if k == 'observation':
                continue
            if i < 2:
                v = step[k]
                if hasattr(v, 'shape') and hasattr(v, 'dtype'):
                    print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"      {k}: type={type(v)}, value={v}")
    # For steps beyond first 5, just check keys presence
    for i, step in enumerate(steps[5:]):
        if not isinstance(step, dict):
            missing_keys_per_step.append(expected_keys)
            continue
        step_keys = set(step.keys())
        all_keys_found.update(step_keys)
        missing_keys = expected_keys - step_keys
        missing_keys_per_step.append(missing_keys)

    # Warning if any step missing expected keys
    any_missing = any(len(missing) > 0 for missing in missing_keys_per_step)
    if any_missing:
        print("\n  [WARN] Missing expected keys in some steps:")
        for i, missing in enumerate(missing_keys_per_step):
            if missing:
                print(f"    Step {i}: missing keys = {sorted(missing)}")
    else:
        print("\n  All steps contain the expected keys.")

    # Print if 'steps' present at top level (and length), or if top-level is list/array
    if isinstance(ep, dict) and 'steps' in ep:
        print(f"\n  Top-level 'steps' key present with length {len(ep['steps'])}.")
    else:
        print(f"\n  Top-level is a {type(ep)} with length {len(ep) if hasattr(ep, '__len__') else 'N/A'}.")

    # Summary section showing all unique keys found across steps
    print("\n  Summary of all unique keys found across steps:")
    for k in sorted(all_keys_found):
        print(f"    {k}")

    # Patch: Find nested image key and save gif
    image_key = find_image_key(step0)
    print(f"image_key: {image_key}")
    image_key='observation.image'
    if image_key is not None:
        print(f"  Using image key '{image_key}'. Saving first 50 frames as gif ...")
        save_episode_video(steps, image_key, save_video_path=path.with_suffix(".gif"), max_frames=50)
        print(f"  First 5 frames stats for '{image_key}':")
        for i, step in enumerate(steps[:5]):
            img = get_image_by_key(step, image_key)
            print(f"    Frame {i}: shape={img.shape}, dtype={img.dtype}, min={np.min(img):.2f}, max={np.max(img):.2f}")
    else:
        print("  [WARN] No image key found (even inside nested dicts).")

if __name__ == "__main__":
    # Point to either a directory or a file
    arg = sys.argv[1] if len(sys.argv) > 1 else "/home/hrilab/Documents/.vlas/cycliclxm-slim/CyclicLxM/rlds_dataset_builder/hanoi_300/data/train"
    data_dir = Path(arg)
    if data_dir.is_file():
        files = [data_dir]
    else:
        files = sorted(data_dir.glob("episode_*.npy"))
    if not files:
        print(f"No files found in {data_dir}")
        sys.exit(1)
    for i, path in enumerate(files):
        if i >= 5: break  # just inspect first 5
        inspect_episode(path)