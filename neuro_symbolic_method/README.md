# Neuro-Symbolic Method

MuJoCo / Robosuite Hanoi with Metric-FF planning and diffusion policies.
Vision at eval uses YOLO + pixel→3D regressor (`--use_yolo`).

------------------------------------------------------------------------

# Setup

```bash
conda create -n neurosym python=3.8 && conda activate neurosym
sudo apt install bison flex   # for Metric-FF
```

```bash
cd neuro_symbolic_method
wget https://fai.cs.uni-saarland.de/hoffmann/ff/Metric-FF-v2.1.tgz
tar -xzvf Metric-FF-v2.1.tgz && cd Metric-FF-v2.1 && make && cd ..

cd ../robosuite && git checkout teach && pip install -r requirements.txt && pip install -e .
cd ../robosuite-task-zoo && pip install -e .
cd ../neuro_symbolic_method/diffusion_policy && pip install -e . && cd ..

pip install gym joblib pyyaml h5py gymnasium matplotlib tarski dill torch \
  diffusers hydra-core wandb tqdm einops zarr pandas ultralytics scikit-learn
```

Weights expected by `configs/hanoi.yaml`:

- `models/yolo/hanoi_yolo.pt`
- `models/regressors/hanoi_regressor.pkl`
- `models/policies/gt/{grasp,drop,reach_pick,reach_place}.ckpt`

WSL2 / headless:

```bash
# sim / eval
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa

# sklearn training
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

------------------------------------------------------------------------

# Hanoi: collect → preprocess → train → execute (vision)

All commands from `neuro_symbolic_method/` with `conda activate neurosym`.

## 1. Collect regressor CSVs

```bash
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM && export MUJOCO_GL=osmesa

python -u auto_demo.py --env Hanoi --train_yolo --use_yolo --rnd_reset \
  --episodes 30 --dir ./data_reg/ --name reg --seed 42
```

Writes `data_reg/Hanoi_seed_42/reg/yolo_data/*.csv`.

## 2. Train pixel→world regressor

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u train_regressor.py \
  --data_glob "data_reg/**/yolo_data/*.csv" \
  --active models/regressors/hanoi_regressor.pkl
```

Optional stereo residual:

```bash
python -u train_residual_regressor.py \
  --data_glob "data_reg/**/yolo_data/*.csv" \
  --out models/regressors/hanoi_residual_regressor.pkl
```

## 3. Collect skill demos (vision obs)

Same perception as eval (`--use_yolo`). Use `--ee` so actions match OSC_POSITION eval.

```bash
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM && export MUJOCO_GL=osmesa

python -u auto_demo.py --env Hanoi --episodes 30 --use_yolo --action_split --ee \
  --name hanoi_yolo --dir ./data/ --seed 0
```

Traces: `data/Hanoi_seed_0/hanoi_yolo/traces/{pick,place,reach_pick,reach_place}.zip`.

Oracle demos (sim poses) instead: drop `--use_yolo`, add `--object_centric`, name e.g. `hanoi_gt`.

## 4. Preprocess → Zarr

```bash
python data_processing/data_to_zarr.py \
  --data_dir data/Hanoi_seed_0/hanoi_yolo --filter_actions True
```

## 5. Train diffusion policies

Hydra configs live under `diffusion_policy/diffusion_policy/config/`.
Each skill needs a **task** yaml (dims + dataset) and a **train** yaml that defaults to it.

Hanoi relative-obs dims (must match zarr / executor):

| Skill | Task config | `obs_dim` | `action_dim` |
|-------|-------------|-----------|--------------|
| pick | `task/grasp_lowdim.yaml` | 2 | 2 |
| place | `task/drop_lowdim.yaml` | 2 | 2 |
| reach_pick | `task/reach_pick_lowdim.yaml` | 3 | 3 |
| reach_place | `task/reach_place_lowdim.yaml` | 3 | 3 |

Example task file (`task/grasp_lowdim.yaml`):

```yaml
name: grasp_lowdim
obs_dim: 2
action_dim: 2
keypoint_dim: 0

env_runner:
  _target_: diffusion_policy.env_runner.pick_place_env_runner.PickPlaceEnvRunner

dataset:
  _target_: diffusion_policy.dataset.pick_place_dataset.PickPlaceLowdimDataset
  zarr_path: ../data/Hanoi_seed_0/hanoi_yolo/hf_traj/pick/keypoint/keypoint.zarr
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
  max_train_episodes: 90
```

Train entry (`train_diffusion_transformer_lowdim_grasp.yaml`) only needs:

```yaml
defaults:
  - _self_
  - task: grasp_lowdim
```

(same pattern: `drop_lowdim`, `reach_pick_lowdim`, `reach_place_lowdim`).

Or override the zarr path on the CLI without editing files:

```bash
cd diffusion_policy
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib" WANDB_MODE=offline

python train.py --config-name=train_diffusion_transformer_lowdim_grasp \
  task.dataset.zarr_path=../data/Hanoi_seed_0/hanoi_yolo/hf_traj/pick/keypoint/keypoint.zarr \
  training.device=cuda:0 logging.mode=offline hydra.run.dir=data/train_yolo/grasp
# reach_pick example:
# python train.py --config-name=train_diffusion_transformer_lowdim_reach_pick \
#   task.dataset.zarr_path=../data/Hanoi_seed_0/hanoi_yolo/hf_traj/reach_pick/keypoint/keypoint.zarr \
#   training.device=cuda:0 logging.mode=offline hydra.run.dir=data/train_yolo/reach_pick
```

Copy checkpoints into `models/policies/gt/` (or update `policies:` in `configs/hanoi.yaml`).
Delete the run dir before a clean retrain (`training.resume: True`).

## 6. Execute

```bash
cd ..   # back to neuro_symbolic_method/
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM && export MUJOCO_GL=osmesa

python -u experiments_neurosymbolic.py --env Hanoi --use_yolo --n_ep 5
# optional: --render  |  --debug
# oracle (no vision): omit --use_yolo
```

------------------------------------------------------------------------
