# Neuro-Symbolic Method Setup

WARNING: This repository is currently not up to date. An updated version will be delivered before the paper publication.

This repository contains the implementation for a **neuro-symbolic
learning framework** built on top of **MuJoCo**, **Robosuite**, and
**Diffusion Policies**.

The project integrates: - Robotic simulation environments - Symbolic
planning with Metric-FF - Diffusion-based policy learning -
Demonstration generation pipelines

------------------------------------------------------------------------

# 1. Requirements

-   Python **3.8**
-   **Conda**
-   **MuJoCo 2.1**
-   System packages:
    -   `bison`
    -   `flex`

Example installation (Ubuntu):

``` bash
sudo apt install bison flex
```

------------------------------------------------------------------------

# 2. Create Conda Environment

``` bash
conda create -n neurosym python=3.8
conda activate neurosym
```

------------------------------------------------------------------------

# 3. Install Metric-FF Planner

Metric-FF is required for symbolic planning.

``` bash
cd neuro_symbolic_method

wget https://fai.cs.uni-saarland.de/hoffmann/ff/Metric-FF-v2.1.tgz
tar -xzvf Metric-FF-v2.1.tgz

cd Metric-FF-v2.1
make
```

If compilation fails, ensure `bison` and `flex` are installed.

Alternatively, manually move the `Metric-FF-v2.1` folder into:

    neuro_symbolic_method/

------------------------------------------------------------------------

# 4. Install Robosuite (Teach Branch)

``` bash
cd ../../robosuite

git checkout teach

pip install -r requirements.txt
pip install -e .
```

------------------------------------------------------------------------

# 5. Install Robosuite Task Zoo

``` bash
cd ../robosuite-task-zoo
pip install -e .
```

------------------------------------------------------------------------

# 6. Install Python Dependencies

``` bash
pip install gym joblib robosuite pyyaml h5py gymnasium matplotlib tarski dill torch diffusers
```

Additional dependencies:

``` bash
pip install hydra-core wandb tqdm einops zarr pandas ultralytics scikit-learn datasets imitation
```

------------------------------------------------------------------------

# 7. Install Diffusion Policy

``` bash
cd ../diffusion_policy
pip install -e .
```

------------------------------------------------------------------------

# 8. Full Hanoi pipeline (collect → train → execute)

This is the supported end-to-end flow for **Panda / Towers of Hanoi**.
All commands run from `neuro_symbolic_method/` unless noted.

**Recommended recipe that reaches full 7/7 pick–place with YOLO at eval:**

1. Collect dual-camera CSV labels for the **pixel→3D regressor**
2. Train a **versioned** regressor (and optionally the stereo residual)
3. Collect **ground-truth** skill demos and train the four diffusion policies
4. Execute with `--use_yolo` (YOLO + regressor for perception; GT-trained policies)

On WSL2 / headless MuJoCo:

```bash
# Simulation / eval (OSMesa): clear conflicting GL libs
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa

# sklearn / scipy training: put conda libs first
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Activate the env once:

```bash
source ~/miniconda3/etc/profile.d/conda.sh   # or your conda path
conda activate neurosym
cd neuro_symbolic_method
```

## 8.1 Collect regressor training data (YOLO bboxes + GT 3D)

`--train_yolo` writes CSVs under `<dir>/Hanoi_seed_<seed>/<name>/yolo_data/`.
Use `--use_yolo` so detections come from the detector; `--rnd_reset` diversifies
peg/cube layouts (important for peg2/peg3).

```bash
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa

python -u auto_demo.py \
  --env Hanoi \
  --train_yolo --use_yolo \
  --rnd_reset \
  --episodes 30 \
  --dir ./data_reg_v2/ \
  --name reg_v2 \
  --seed 42
```

Optional viewer / bbox window: add `--render`.

CSVs land in e.g. `data_reg_v2/Hanoi_seed_42/reg_v2/yolo_data/*.csv`.

## 8.2 Train the pixel→world regressor (versioned)

Never overwrite archives blindly. `train_regressor.py` writes a timestamped
file under `models/regressors/archive/` and can install the active path used by
`configs/hanoi.yaml` (`models/regressors/hanoi_regressor.pkl`).

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u train_regressor.py \
  --data_glob "data_reg_v2/**/yolo_data/*.csv" "data_rnd/**/yolo_data/*.csv" \
  --out "models/regressors/archive/hanoi_regressor_$(date +%Y%m%d_%H%M%S).pkl" \
  --active models/regressors/hanoi_regressor.pkl
```

Optional stereo residual (refines dual-camera triangulation when both cams see
the cube):

```bash
python -u train_residual_regressor.py \
  --data_glob "data_tri_train/**/yolo_data/*.csv" \
  --out models/regressors/hanoi_residual_regressor.pkl
```

Requires a YOLO weights file at `models/yolo/hanoi_yolo.pt` (see config).

## 8.3 Collect skill demonstrations for diffusion policies

Train policies on **simulator ground truth** (`--object_centric`). At eval,
those policies consume relative obs built from YOLO+regressor estimates
(`--use_yolo`). That combination is what we used for full Hanoi success.

```bash
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa

# Ground-truth demos -> data/Hanoi_seed_0/hanoi_gt/traces/{pick,place,reach_pick,reach_place}.zip
python -u auto_demo.py \
  --env Hanoi \
  --episodes 30 \
  --object_centric \
  --action_split \
  --name hanoi_gt \
  --dir ./data/ \
  --seed 0
```

Optional: also collect YOLO-observation demos (`--use_yolo --action_split`,
name e.g. `hanoi_yolo`) if you want policies trained under perception noise.
GT-trained policies performed better for grasp in our runs.

Key flags:

| Flag | Role |
|------|------|
| `--episodes N` | Number of demos |
| `--action_split` | Split into per-skill zips for diffusion training |
| `--object_centric` | Relative GT object obs (oracle demos) |
| `--use_yolo` | YOLO+regressor obs (perception demos / CSV labeling with `--train_yolo`) |
| `--train_yolo` | Write bbox→3D CSVs for regressor training |
| `--rnd_reset` | Randomize Hanoi resets |
| `--render` | MuJoCo viewer + YOLO bbox window when applicable |

## 8.4 Convert traces → Zarr

```bash
python data_processing/data_to_zarr.py \
  --data_dir data/Hanoi_seed_0/hanoi_gt --filter_actions True
```

Produces `data/Hanoi_seed_0/hanoi_gt/hf_traj/<skill>/keypoint/keypoint.zarr`
for `pick`, `place`, `reach_pick`, `reach_place`.

## 8.5 Train the four diffusion policies

| `--config-name` | Skill folder |
|-----------------|--------------|
| `train_diffusion_transformer_lowdim_grasp` | `pick` |
| `train_diffusion_transformer_lowdim_drop` | `place` |
| `train_diffusion_transformer_lowdim_reach_pick` | `reach_pick` |
| `train_diffusion_transformer_lowdim_reach_place` | `reach_place` |

```bash
cd diffusion_policy
conda activate neurosym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export WANDB_MODE=offline

python train.py \
  --config-name=train_diffusion_transformer_lowdim_grasp \
  task.dataset.zarr_path=../data/Hanoi_seed_0/hanoi_gt/hf_traj/pick/keypoint/keypoint.zarr \
  training.device=cuda:0 logging.mode=offline \
  hydra.run.dir=data/train_gt/grasp
# repeat for drop / reach_pick / reach_place with the matching zarr paths
```

Configs use `training.resume: True`. To retrain cleanly, remove the old run dir
first (`rm -rf data/train_gt/grasp`).

Point `configs/hanoi.yaml` `policies:` at the resulting checkpoints (defaults
already target `diffusion_policy/data/train_gt/...`).

### tmux (one window per skill)

```bash
SETUP='source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate neurosym && cd '"$PWD"' && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib" WANDB_MODE=offline'

tmux new-session -d -s hanoi_train_gt -n grasp
tmux send-keys -t hanoi_train_gt:grasp "$SETUP && python train.py --config-name=train_diffusion_transformer_lowdim_grasp task.dataset.zarr_path=../data/Hanoi_seed_0/hanoi_gt/hf_traj/pick/keypoint/keypoint.zarr training.device=cuda:0 logging.mode=offline hydra.run.dir=data/train_gt/grasp" C-m
# ... drop, reach_pick, reach_place similarly ...
tmux attach -t hanoi_train_gt
```

## 8.6 Execute / evaluate

```bash
cd neuro_symbolic_method   # if still in diffusion_policy/
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa

# Oracle (simulator object poses)
python -u experiments_neurosymbolic.py --env Hanoi --n_ep 5 --debug

# YOLO perception (7/7 pick–place on the standard Hanoi plan when regressor+YOLO are healthy)
python -u experiments_neurosymbolic.py --env Hanoi --use_yolo --n_ep 1 --debug

# Optional: MuJoCo viewer + YOLO bbox window ("YOLO Detections")
python -u experiments_neurosymbolic.py --env Hanoi --use_yolo --render --n_ep 1 --debug
```

Flags:

- `--use_yolo` — YOLO + dual-cam/EE regressor (and stereo when both cams see the object)
- `--render` — viewer + bbox overlay (snapshot after each gripper reset)
- `--n_ep` — evaluation episodes
- `--debug` — POS / predicate logging

Perception details (current executor): color-mapped single-frame detection after
each gripper reset; stereo triangulation (+ residual) when both cameras see a
cube; otherwise the learned regressor. Policies in `hanoi.yaml` use relative
observations (`oracle: true`).

------------------------------------------------------------------------

# Notes

- Ensure **MuJoCo 2.1** is installed and OSMesa works for headless runs.
- Do not commit large model weights (`.pkl` / `.pt` / policy `.ckpt`) or demo
  datasets; keep versioned regressors under `models/regressors/archive/` locally.
- Metric-FF must be built (section 3) before planning works at eval time.

------------------------------------------------------------------------
