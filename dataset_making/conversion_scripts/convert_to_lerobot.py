#!/usr/bin/env python3
import pickle
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME


def main(data_dir: str):
    # 1) Where to write the LeRobot-format dataset
    repo_id     = "tduggan93/hanoi_dataset"   # or whatever you choose
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        # clear out any previous run
        import shutil; shutil.rmtree(output_path)

    # 2) Create an *empty* LeRobotDataset (v2 API) with your four features
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=HF_LEROBOT_HOME,
        robot_type="kinova3",
        fps=10,
        features={
            "image":       {"dtype":"image",   "shape":(256,256,3)},
            "wrist_image": {"dtype":"image",   "shape":(256,256,3)},
            "state":       {"dtype":"float32", "shape":(8,)},
            "actions":     {"dtype":"float32", "shape":(7,)},
            # "task":        {"dtype":"string",  "shape":(),   "names":["task"]},
        },
    )  # this sets up metadata, TFRecord writer threads, etc. :contentReference[oaicite:2]{index=2}

    # 3) Load your episodes and feed them in
    raw_dir = Path(data_dir)
    for npy_path in sorted(raw_dir.glob("episode_*.npy")):
        # load with pickle-compatible array-of-dicts
        ep: list = list(np.load(npy_path, allow_pickle=True))
        current_task = ep[0]["language_instruction"]
        for step in ep:
            # whenever the instruction changes, close out the previous episode
            instr = step["language_instruction"]
            if instr != current_task:
                dataset.save_episode()
                print(f"{instr} in {npy_path} completed")
                current_task = instr
            dataset.add_frame({
                "image":       step["image"],
                "wrist_image": step["wrist_image"],
                "state":       step["state"],
                "actions":     step["action"],
                "task":        instr.decode() if isinstance(instr, bytes) else instr, #step["language_instruction"],
            })
        # attach the per-episode instruction exactly as LeRobot expects
        dataset.save_episode()
        print(f"{npy_path} completed")

    # 4) Write out TFRecords + metadata index
    dataset.consolidate(run_compute_stats=False)  # now you have a loader-friendly dataset :contentReference[oaicite:3]{index=3}


main("data/hanoi_dataset/data")

# Old code
# import sys, pickle
# from pathlib import Path
# import numpy as np
# import numpy.lib.format as nf

# #–– your redirecting Unpickler from before ––
# class NumpyCoreRedirectUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module.startswith("numpy._core"):
#             module = module.replace("numpy._core", "numpy.core")
#         return super().find_class(module, name)

# def load_npy_object_array(path):
#     with open(path, 'rb') as f:
#         version = nf.read_magic(f)
#         shape, fortran_order, dtype = nf._read_array_header(f, version)
#         return NumpyCoreRedirectUnpickler(f).load()

# if __name__ == "__main__":
#     src_dir  = Path("data/hanoi_dataset/data")   # where your episode_*.npy live
#     dst_dir  = Path("data/hanoi_dataset/lerobot")  # new lerobot‐style folder
#     dst_dir.mkdir(parents=True, exist_ok=True)

#     for path in sorted(src_dir.glob("episode_*.npy")):
#         ep_objarr = load_npy_object_array(path)
#         # turn it into a plain Python list of dicts
#         ep_list = list(ep_objarr) if isinstance(ep_objarr, np.ndarray) else ep_objarr

#         # write out as pickle (so lerobot can load via standard pickle.load)
#         out_path = dst_dir / path.with_suffix(".pkl").name
#         with open(out_path, "wb") as f:
#             pickle.dump(ep_list, f, protocol=pickle.HIGHEST_PROTOCOL)

#         print(f"Converted {path.name} → {out_path.name} ({len(ep_list)} steps)")
