import zipfile, pickle, numpy as np, os, cv2
from datasets import Dataset
from huggingface_hub import HfApi

def load_demos(traces_dir):
    with zipfile.ZipFile(os.path.join(traces_dir, "main.zip")) as zf:
        with zf.open("data.pkl") as f:
            raw = pickle.load(f)
    episodes = []
    for traj_list, task_buffer in raw:
        actions = [traj_list[i]     for i in range(0, len(traj_list), 2)]
        obs     = [traj_list[i + 1] for i in range(0, len(traj_list), 2)]
        episodes.append({"actions": actions, "obs": obs, "tasks": task_buffer})
    return episodes

def encode_jpeg(img_array):
    """Compress a (256,256,3) BGR numpy array to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()

def episodes_to_rows(episodes):
    rows = []
    for ep_idx, ep in enumerate(episodes):
        T = len(ep["actions"])
        task_str = " | ".join(ep["tasks"])
        print(f"  Episode {ep_idx+1}/{len(episodes)}  ({T} steps)")
        for t, (action, obs) in enumerate(zip(ep["actions"], ep["obs"])):
            rows.append({
                "episode_index": ep_idx,
                "frame_index":   t,
                "timestamp":     t / 20.0,
                "task":          task_str,
                "action":        action.tolist(),
                "proprio":       obs["proprio"].tolist(),
                "objects_pos":   obs["objects_pos"],
                "agentview":     encode_jpeg(obs["agentview"]),    # bytes, ~5-15 KB each
                "wrist_image":   encode_jpeg(obs["wrist_image"]),  # bytes, ~5-15 KB each
                "is_terminal":   t == T - 1,
            })
    return rows

def push_to_hub(traces_dir, repo_id):
    print("Loading demos...")
    episodes = load_demos(traces_dir)
    print(f"Loaded {len(episodes)} episodes\n")

    print("Converting to rows...")
    rows = episodes_to_rows(episodes)
    print(f"\nTotal steps: {len(rows)}")

    print("Building dataset...")
    dataset = Dataset.from_list(rows)

    print(f"Pushing to {repo_id}...")
    dataset.push_to_hub(repo_id, private=True)
    print("Done!")

if __name__ == "__main__":
    push_to_hub(
        traces_dir = "data/Hanoi_seed_0/Matthias/traces",
        repo_id    = "lorangpi/hanoi-demos"
    )