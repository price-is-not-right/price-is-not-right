import os
from pathlib import Path
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from itertools import islice

DATASET_PATH = "/home/hrilab/tensorflow_datasets/hanoi300/1.0.0"
SAVE_GIF_DIR = Path("inspection_videos/rlds")
SAVE_GIF_DIR.mkdir(exist_ok=True)

def save_frames_as_gif(frames, out_path, duration=100):
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )

def main():
    builder = tfds.builder_from_directory(DATASET_PATH)
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train", as_supervised=False)
    print(f"Loaded {builder.name} with {builder.info.splits['train'].num_examples} train episodes")

    for i, ex in enumerate(tfds.as_numpy(ds)):
        # ------------------------------------------------------------------
        # Grab the first 60 steps so that the GIF starts at step 0.
        # ------------------------------------------------------------------
        steps_iter = iter(ex["steps"])
        first_60 = list(islice(steps_iter, 60))  # steps 0‑59
        if not first_60:
            continue

        step0 = first_60[0]
        print("  Step 0 keys:", list(step0.keys()))
        obs0 = step0["observation"]
        print("  Step 0 observation keys:", list(obs0.keys()))

        # Check images in the very first step
        for cam_key in ["image", "wrist_image"]:
            img = obs0[cam_key]
            print(
                f"    {cam_key}: shape={img.shape}, dtype={img.dtype}, "
                f"min={np.min(img)}, max={np.max(img)}"
            )

        # ------------------------------------------------------------------
        # Save a GIF with the first 60 RGB frames for the first two episodes
        # ------------------------------------------------------------------
        if i < 2:
            frames = []
            for s in first_60:
                img = s["observation"]["image"]
                if img.max() <= 1.01:          # if normalised to [0,1]
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                frames.append(img)

            gif_path = SAVE_GIF_DIR / f"rlds_episode_{i}.gif"
            save_frames_as_gif(frames, gif_path)
            print(f"    [Saved gif to {gif_path}]")

        if i >= 5:
            break


def inspect_for_consistency():
    builder = tfds.builder_from_directory(DATASET_PATH)
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train", as_supervised=False)
    print(f"Loaded {builder.name} with {builder.info.splits['train'].num_examples} train episodes")

    n_checked = 0
    n_broken = 0
    error_counts = {}

    for i, ex in enumerate(tfds.as_numpy(ds)):
        steps = ex.get("steps", None)
        if steps is None:
            print(f"Episode {i} is missing 'steps' key!")
            n_broken += 1
            error_counts["missing_steps"] = error_counts.get("missing_steps", 0) + 1
            continue

        step_shapes = None
        for j, step in enumerate(steps):
            obs = step.get("observation", None)
            if obs is None:
                print(f"Episode {i}, step {j}: missing 'observation'")
                n_broken += 1
                error_counts["missing_observation"] = error_counts.get("missing_observation", 0) + 1
                continue

            # Check expected keys
            for k in ["image", "wrist_image", "state"]:
                if k not in obs:
                    print(f"Episode {i}, step {j}: missing observation key '{k}'")
                    n_broken += 1
                    error_counts[f"missing_{k}"] = error_counts.get(f"missing_{k}", 0) + 1

            for k in ["action", "discount", "reward", "is_first", "is_last", "is_terminal", "language_instruction"]:
                if k not in step:
                    print(f"Episode {i}, step {j}: missing step key '{k}'")
                    n_broken += 1
                    error_counts[f"missing_{k}"] = error_counts.get(f"missing_{k}", 0) + 1

            # Check shapes and dtypes
            try:
                img = obs["image"]
                wimg = obs["wrist_image"]
                state = obs["state"]
                action = step["action"]

                # Only check shapes for first step, then compare rest
                shapes = {
                    "image": img.shape,
                    "wrist_image": wimg.shape,
                    "state": state.shape,
                    "action": action.shape,
                }
                dtypes = {
                    "image": img.dtype,
                    "wrist_image": wimg.dtype,
                    "state": state.dtype,
                    "action": action.dtype,
                }

                # Define the expected shapes and types for your dataset
                expected_shapes = {
                    "image": (256, 256, 3),
                    "wrist_image": (256, 256, 3),
                    "state": (8,),
                    "action": (7,),
                }
                expected_dtypes = {
                    "image": np.uint8,
                    "wrist_image": np.uint8,
                    "state": np.float32,
                    "action": np.float32,
                }

                # Compare to expected
                for k in expected_shapes:
                    if shapes[k] != expected_shapes[k]:
                        print(f"Episode {i}, step {j}: '{k}' shape {shapes[k]} != expected {expected_shapes[k]}")
                        n_broken += 1
                        error_counts[f"bad_shape_{k}"] = error_counts.get(f"bad_shape_{k}", 0) + 1
                    if dtypes[k] != expected_dtypes[k]:
                        print(f"Episode {i}, step {j}: '{k}' dtype {dtypes[k]} != expected {expected_dtypes[k]}")
                        n_broken += 1
                        error_counts[f"bad_dtype_{k}"] = error_counts.get(f"bad_dtype_{k}", 0) + 1

                # Optionally check for nan/inf in state and action
                if np.any(np.isnan(state)) or np.any(np.isnan(action)):
                    print(f"Episode {i}, step {j}: NaN found in 'state' or 'action'")
                    n_broken += 1
                    error_counts["nan"] = error_counts.get("nan", 0) + 1

                if np.any(np.isinf(state)) or np.any(np.isinf(action)):
                    print(f"Episode {i}, step {j}: Inf found in 'state' or 'action'")
                    n_broken += 1
                    error_counts["inf"] = error_counts.get("inf", 0) + 1

                # Check consistency of shapes within episode
                if step_shapes is None:
                    step_shapes = shapes
                else:
                    for k in shapes:
                        if shapes[k] != step_shapes[k]:
                            print(f"Episode {i}: Inconsistent shape for '{k}' in step {j}: {shapes[k]} vs {step_shapes[k]}")
                            n_broken += 1
                            error_counts[f"inconsistent_{k}"] = error_counts.get(f"inconsistent_{k}", 0) + 1

            except Exception as e:
                print(f"Episode {i}, step {j}: Exception during checking: {e}")
                n_broken += 1
                error_counts["exception"] = error_counts.get("exception", 0) + 1

        n_checked += 1

    print(f"\nDone. Checked {n_checked} episodes. Found {n_broken} issues.")
    if n_broken:
        print("Error counts:")
        for k, v in error_counts.items():
            print(f"  {k}: {v}")

def validate_for_lerobot():
    builder = tfds.builder_from_directory(DATASET_PATH)
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train", as_supervised=False)
    print(f"Loaded {builder.name} with {builder.info.splits['train'].num_examples} train episodes")

    n_checked = 0
    n_broken = 0
    error_counts = {}

    required_obs_keys = ["image", "wrist_image", "state"]
    required_step_keys = ["action"]

    expected_shapes = {
        "image": (256, 256, 3),
        "wrist_image": (256, 256, 3),
        "state": (8,),
        "action": (7,),
    }
    expected_dtypes = {
        "image": np.uint8,
        "wrist_image": np.uint8,
        "state": np.float32,
        "action": np.float32,
    }

    for i, ex in enumerate(tfds.as_numpy(ds)):
        steps = ex.get("steps", None)
        if steps is None:
            print(f"Episode {i} is missing 'steps' key!")
            n_broken += 1
            error_counts["missing_steps"] = error_counts.get("missing_steps", 0) + 1
            continue

        for j, step in enumerate(steps):
            obs = step.get("observation", None)
            if obs is None:
                print(f"Episode {i}, step {j}: missing 'observation'")
                n_broken += 1
                error_counts["missing_observation"] = error_counts.get("missing_observation", 0) + 1
                continue

            # Check required observation keys
            for k in required_obs_keys:
                if k not in obs:
                    print(f"Episode {i}, step {j}: missing observation key '{k}'")
                    n_broken += 1
                    error_counts[f"missing_{k}"] = error_counts.get(f"missing_{k}", 0) + 1

            # Check required step keys
            for k in required_step_keys:
                if k not in step:
                    print(f"Episode {i}, step {j}: missing step key '{k}'")
                    n_broken += 1
                    error_counts[f"missing_{k}"] = error_counts.get(f"missing_{k}", 0) + 1

            # Check shapes and dtypes for only required fields
            try:
                for k in expected_shapes:
                    value = obs[k] if k in obs else step[k]
                    if value.shape != expected_shapes[k]:
                        print(f"Episode {i}, step {j}: '{k}' shape {value.shape} != expected {expected_shapes[k]}")
                        n_broken += 1
                        error_counts[f"bad_shape_{k}"] = error_counts.get(f"bad_shape_{k}", 0) + 1
                    if value.dtype != expected_dtypes[k]:
                        print(f"Episode {i}, step {j}: '{k}' dtype {value.dtype} != expected {expected_dtypes[k]}")
                        n_broken += 1
                        error_counts[f"bad_dtype_{k}"] = error_counts.get(f"bad_dtype_{k}", 0) + 1

                # Check for NaN/Inf in state/action
                for k in ["state", "action"]:
                    arr = obs[k] if k in obs else step[k]
                    if np.any(np.isnan(arr)):
                        print(f"Episode {i}, step {j}: NaN found in '{k}'")
                        n_broken += 1
                        error_counts[f"nan_{k}"] = error_counts.get(f"nan_{k}", 0) + 1
                    if np.any(np.isinf(arr)):
                        print(f"Episode {i}, step {j}: Inf found in '{k}'")
                        n_broken += 1
                        error_counts[f"inf_{k}"] = error_counts.get(f"inf_{k}", 0) + 1

                # After dtype and shape checks, add:
                for k in expected_shapes:
                    value = obs[k] if k in obs else step[k]
                    if np.count_nonzero(value) == 0:
                        print(f"Episode {i}, step {j}: '{k}' is all zeros!")
                        n_broken += 1
                        error_counts[f"all_zero_{k}"] = error_counts.get(f"all_zero_{k}", 0) + 1
            except Exception as e:
                print(f"Episode {i}, step {j}: Exception during checking: {e}")
                n_broken += 1
                error_counts["exception"] = error_counts.get("exception", 0) + 1

            n_checked += 1

    print(f"\nDone. Checked {n_checked} episodes. Found {n_broken} issues.")
    if n_broken:
        print("Error counts:")
        for k, v in error_counts.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
    # validate_for_lerobot()
    # inspect_for_consistency()
