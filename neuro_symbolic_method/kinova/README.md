# Kinova3 Hanoi — neuro-symbolic vision pipeline

Runs the Hanoi tower task on a Kinova3 with a Robotiq85 gripper: YOLO detects the
cubes, regressors turn the detections into 3D poses, a PDDL planner orders the
moves, and four diffusion policies execute them.

## Get the code

Install git-lfs *before* cloning, otherwise the weights arrive as small text
pointers and loading them fails.

```bash
git lfs install
git clone --recurse-submodules https://github.com/price-is-not-right/price-is-not-right.git
cd price-is-not-right
```

## Environment

Once, from the repository root:

```bash
conda create -n neurosym python=3.10 -y
conda activate neurosym
pip install -r requirements.txt

# local packages
cd robosuite && pip install -e . && cd ..
cd robosuite-task-zoo && pip install -e . && cd ..
cd neuro_symbolic_method/diffusion_policy && pip install -e . && cd ../..

# model weights, then the Metric-FF planner
git lfs pull
sudo apt install bison flex
cd neuro_symbolic_method
wget https://fai.cs.uni-saarland.de/hoffmann/ff/Metric-FF-v2.1.tgz
tar -xzvf Metric-FF-v2.1.tgz && cd Metric-FF-v2.1 && make && cd ../..
```

Then in every shell, from `neuro_symbolic_method/`:

```bash
conda activate neurosym
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
```

`--robot Kinova3` is the only robot flag you need — it selects
`kinova/configs/hanoi.yaml`, which holds the Kinova policies, regressors, skill
horizons and `n_act`. Omit it to run the Panda baseline.

---

## 1. Link the Hydra configs (once)

```bash
bash kinova/scripts/link_hydra_configs.sh
```

## 2. Collect demonstrations

An expert scripted from sim poses records 30 episodes and converts the traces to
Zarr in the same step.

```bash
EPISODES=30 SEED=0 bash kinova/scripts/collect_gt_demos.sh
```

Output: `kinova/data_gt/Hanoi_seed_0/hanoi_gt/hf_traj/{pick,place,reach_pick,reach_place}/keypoint/keypoint.zarr`

## 3. Train the four diffusion policies

Roughly a day on one GPU; run it under `tmux`.

```bash
bash kinova/scripts/train_gt_policies.sh
```

Output: `kinova/data/train/train_kinova_gt_{grasp,drop,reach_pick,reach_place}/`

## 4. Deploy the checkpoints

```bash
bash kinova/scripts/sync_gt_policy_checkpoints.sh latest   # or: best
```

Copies into `kinova/models/policies/gt/` as `grasp.ckpt` (Pick), `drop.ckpt`
(Drop), `reach_pick.ckpt` (ReachPick) and `reach_place.ckpt` (ReachDrop).

## 5. Collect vision data and train the regressors

The cube detector is shared with Panda, so only the pixel-to-world regressors are
retrained — the Kinova wrist camera sits differently.

```bash
python -u auto_demo.py --env Hanoi --robot Kinova3 --train_yolo --rnd_reset \
  --episodes 30 --dir ./kinova/data_reg/ --name reg --seed 42

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u train_regressor.py \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --active kinova/models/regressors/hanoi_regressor.pkl

python -u train_regressor.py --mono \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --active kinova/models/regressors/hanoi_mono_regressor.pkl

python -u train_residual_regressor.py \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --out kinova/models/regressors/hanoi_residual_regressor.pkl

unset LD_LIBRARY_PATH
```

The dual regressor handles both cameras, the mono one covers frames where the
wrist camera sees nothing, and the residual model corrects stereo triangulation.

## 6. Evaluate

```bash
# Vision (YOLO + regressors)
python -u experiments_neurosymbolic.py --env Hanoi --robot Kinova3 --use_yolo --n_ep 100 --seed 0

# Sim poses instead of vision, to isolate the policies
python -u experiments_neurosymbolic.py --env Hanoi --robot Kinova3 --n_ep 100 --seed 0

# Panda baseline
python -u experiments_neurosymbolic.py --env Hanoi --n_ep 100 --seed 0
```

Wrappers that log to a file and print a success summary (100 episodes, seed 0 by
default):

```bash
bash kinova/scripts/run_yolo_eval.sh
bash kinova/scripts/run_gt_eval.sh
```

---

## Layout

```
kinova/
  configs/hanoi.yaml     # robot, policy + regressor paths, horizons, n_act
  models/                # regressors and policy checkpoints (Git LFS)
  data_gt/               # demonstrations and Zarr datasets
  data_reg/              # regressor CSVs
  diffusion_config/      # Hydra task and training yamls
  scripts/               # the commands above, wrapped
```

## Notes

- Keep the `OSC_POSITION` controller for both demos and eval; `default_kinova3`
  is joint-velocity and will not match the trained action space.
- Peg poses come from sim fixtures and the symbolic predicates come from the
  detector. Only the cube poses go through vision.
- Panda checkpoints and regressors will not transfer — the wrist geometry and
  gripper differ.
- If `reset_gripper` stalls, retune `reset_gripper_pos` in `kinova/configs/hanoi.yaml`.
- Everything runs end to end with `bash kinova/scripts/run_pipeline.sh`, but that
  takes about a day; prefer the steps above.
