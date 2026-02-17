import os
from pathlib import Path
import tensorflow_datasets as tfds
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

DATASET_PATH = "/home/hrilab/tensorflow_datasets/hanoi300/1.0.0"  # Change if needed!
REPO_NAME = "tduggan93/hanoi_300_lerobot"                          # Update to your HuggingFace user/repo
ROBOT_TYPE = "panda"                                                 # Update as needed

def convert_rlds_to_lerobot():
    # Set up LeRobot output dataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type=ROBOT_TYPE,
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            # Add extra fields if you need (e.g., "task": {"dtype": ...})
        },
        image_writer_threads=8,
        image_writer_processes=2,
    )

    # Load RLDS dataset
    builder = tfds.builder_from_directory(DATASET_PATH)
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train", as_supervised=False)
    print(f"Loaded {builder.name} with {builder.info.splits['train'].num_examples} train episodes")

    # Convert each episode
    for epi_idx, ex in enumerate(tfds.as_numpy(ds)):
        steps = ex.get("steps", None)
        if steps is None:
            print(f"Episode {epi_idx} missing steps, skipping.")
            continue

        for j, step in enumerate(steps):
            obs = step["observation"]
            # Optional: handle decoding of bytes objects for strings, etc.

            # Optionally: preprocess images if not uint8 (shouldn't be needed if already validated)
            frame_dict = {
                "image": obs["image"],
                "wrist_image": obs["wrist_image"],
                "state": obs["state"],
                "actions": step["action"],
                # "task": step["language_instruction"].decode()  # Uncomment if you want task/language
            }
            task_str = step["language_instruction"].decode() if isinstance(step["language_instruction"], bytes) else str(step["language_instruction"])
            dataset.add_frame(frame_dict, task=task_str)
        dataset.save_episode()  # End of episode
        if epi_idx % 10 == 0:
            print(f"Converted episode {epi_idx}")

    print("Done converting RLDS dataset to LeRobot format!")

if __name__ == "__main__":
    convert_rlds_to_lerobot()