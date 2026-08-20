# Kinova3 Hanoi — neuro-symbolic vision pipeline

Self-contained layout to **replicate the Panda Hanoi YOLO experiment on Kinova3**
(Robotiq85Gripper). PDDL, cubes, and pegs are unchanged; robot, data, regressors,
and policies are Kinova-specific.

All commands below assume:

```bash
conda activate neurosym
cd neuro_symbolic_method
unset LD_LIBRARY_PATH PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
```

---

## Layout

```
kinova/
  configs/hanoi.yaml          # robot=Kinova3 + Kinova weight paths
  models/{yolo,regressors,policies}/
  data/                       # skill demos + zarr + train runs
  data_reg/                   # regressor CSV collection (created by you)
  diffusion_config/           # Hydra task + train yamls
  scripts/
    link_hydra_configs.sh     # symlink configs into diffusion_policy/
    run_pipeline.sh           # optional end-to-end driver
  README.md                   # this file
```

Shared with Panda (do not retrain unless cubes change):

- `models/yolo/hanoi_yolo.pt` — cube detector
- `planning/PDDL/hanoi/` — planner domain/problem

Must be **Kinova-specific** (recollect / retrain):

- `kinova/models/regressors/hanoi_{regressor,residual,mono}_regressor.pkl`
- `kinova/models/policies/{grasp,drop,reach_pick,reach_place}.ckpt`

Infrastructure already wired for Kinova:

- `--robot Kinova3` on `auto_demo.py` / `experiments_neurosymbolic.py` — this is the
  only flag needed; eval resolves `kinova/configs/hanoi.yaml` from
  `ROBOT_CONFIG_REGISTRY`. Omit it (or pass `--robot Panda`) to run `configs/hanoi.yaml`.
  `--config` still overrides the resolved path.
- Finger bodies auto-detect Robotiq85 (`left_inner_finger` / `right_inner_finger`)
- Residual/mono loaders also search `kinova/models/regressors/`

---

## 0. One-time Hydra link

```bash
bash kinova/scripts/link_hydra_configs.sh
```

---

## 1. GT smoke (no vision)

Confirms Kinova3 + OSC_POSITION + PDDL + **Panda** policies will *not* transfer —
expect failure until you retrain policies. Useful to verify the arm resets and
`open()` / `reset_gripper` work:

```bash
python -u experiments_neurosymbolic.py --env Hanoi --robot Kinova3 --n_ep 1 --seed 0
```

If `reset_gripper` stalls, retune `reset_gripper_pos` in `kinova/configs/hanoi.yaml`.

---

## 2. Collect regressor CSVs

Oracle CSV labels (`--train_yolo`); robot must be Kinova3 so wrist FOV matches eval.

```bash
python -u auto_demo.py --env Hanoi --robot Kinova3 --train_yolo --rnd_reset \
  --episodes 30 --dir ./kinova/data_reg/ --name reg --seed 42
```

Writes `kinova/data_reg/Hanoi_seed_42/reg/yolo_data/*.csv`.

---

## 3. Train pixel→world regressors

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u train_regressor.py \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --active kinova/models/regressors/hanoi_regressor.pkl

python -u train_residual_regressor.py \
  --data_glob "kinova/data_reg/**/yolo_data/*.csv" \
  --out kinova/models/regressors/hanoi_residual_regressor.pkl
```

Optional mono (agentview-only) fallback — train a cam1-only regressor the same way
as Panda, or temporarily copy the dual model:

```bash
cp kinova/models/regressors/hanoi_regressor.pkl \
   kinova/models/regressors/hanoi_mono_regressor.pkl
```

---

## 4. Collect skill demos (vision + OSC_POSITION)

Use the Kinova regressor; `--ee` matches eval action space.

```bash
unset LD_LIBRARY_PATH
python -u auto_demo.py --env Hanoi --robot Kinova3 --episodes 30 \
  --use_yolo --action_split --ee --rnd_reset \
  --name hanoi_yolo --dir ./kinova/data/ --seed 0 \
  --yolo_model models/yolo/hanoi_yolo.pt \
  --regressor_model kinova/models/regressors/hanoi_regressor.pkl
```

Traces: `kinova/data/Hanoi_seed_0/hanoi_yolo/traces/{pick,place,reach_pick,reach_place}.zip`.

---

## 5. Preprocess → Zarr

```bash
python data_processing/data_to_zarr.py \
  --data_dir kinova/data/Hanoi_seed_0/hanoi_yolo --filter_actions True
```

---

## 6. Train diffusion policies

```bash
bash kinova/scripts/link_hydra_configs.sh
cd diffusion_policy
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib" WANDB_MODE=offline

python train.py --config-name=train_kinova_grasp \
  training.device=cuda:0 logging.mode=offline \
  hydra.run.dir=../kinova/data/train/train_kinova_grasp

python train.py --config-name=train_kinova_drop \
  training.device=cuda:0 logging.mode=offline \
  hydra.run.dir=../kinova/data/train/train_kinova_drop

python train.py --config-name=train_kinova_reach_pick \
  training.device=cuda:0 logging.mode=offline \
  hydra.run.dir=../kinova/data/train/train_kinova_reach_pick

python train.py --config-name=train_kinova_reach_place \
  training.device=cuda:0 logging.mode=offline \
  hydra.run.dir=../kinova/data/train/train_kinova_reach_place

cd ..
```

Copy checkpoints into `kinova/models/policies/` as:

| File | Skill |
|------|--------|
| `grasp.ckpt` | Pick |
| `drop.ckpt` | Drop |
| `reach_pick.ckpt` | ReachPick |
| `reach_place.ckpt` | ReachDrop |

(`kinova/configs/hanoi.yaml` already points here.)

---

## 7. Evaluate

```bash
unset LD_LIBRARY_PATH
# Oracle poses (sanity)
python -u experiments_neurosymbolic.py --env Hanoi --robot Kinova3 --n_ep 5 --seed 0

# YOLO vision
python -u experiments_neurosymbolic.py --env Hanoi --robot Kinova3 --use_yolo --n_ep 50 --seed 0

# Panda (unchanged baseline): omit --robot
python -u experiments_neurosymbolic.py --env Hanoi --n_ep 5 --seed 0
```

---

## One-shot driver

```bash
bash kinova/scripts/run_pipeline.sh
```

Edit `EP_REG` / `EP_DEMO` / `SEED` at the top of that script as needed.

---

## Notes

- Prefer **Kinova3** over Jaco (three-finger gripper needs more detector work).
- Keep `OSC_POSITION` for eval and `--ee` demos; do not use `default_kinova3` (joint velocity).
- Peg poses still come from sim fixtures; planning/termination still use GT detector predicates (same as Panda vision runs).
- Do not reuse Panda `models/policies/gt/*.ckpt` or Panda regressors on Kinova — wrist geometry differs.
