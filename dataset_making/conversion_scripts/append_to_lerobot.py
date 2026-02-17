from pathlib import Path
import tensorflow_datasets as tfds
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_PATH = "/home/hrilab/tensorflow_datasets/hanoi300/1.0.0"  # Change if needed!
REPO_NAME = "tduggan93/hanoi_300_lerobot"  

# 1) reopen the existing repo (don’t use create again!)
dataset = LeRobotDataset(REPO_NAME)      # falls back to local repo path

builder = tfds.builder_from_directory(DATASET_PATH)

for split in ["val", "test"]:                      # only the pieces you still need
    ds = builder.as_dataset(split=split, as_supervised=False)
    print(f"Adding {split}: {builder.info.splits[split].num_examples} episodes")

    for epi_idx, ex in enumerate(tfds.as_numpy(ds)):
        steps = ex["steps"]
        for step in steps:
            obs = step["observation"]
            frame = {
                "image":        obs["image"],
                "wrist_image":  obs["wrist_image"],
                "state":        obs["state"],
                "actions":      step["action"],
            }
            lang = step["language_instruction"]
            task_str = lang.decode() if isinstance(lang, bytes) else str(lang)
            # you can keep a split tag if you like:
            dataset.add_frame(frame, task=task_str)      # `split` not supported by this version

        dataset.save_episode()                     # closes the episode

    print(f"✓ finished {split}")

# 2) push / finalize if you use the HF hub
dataset.push_to_hub()         # or dataset.finalize() for purely local usage