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

# 8. Generate Automatic Demonstrations

``` bash
python auto_demo.py --seed 10 --use_yolo --ee --rnd_reset --one_operation
```

Options:

-   `--render` : visualize the simulation
-   `--rnd_reset` : randomize environment resets

------------------------------------------------------------------------

# Notes

-   Ensure **MuJoCo 2.1** is correctly installed and configured.
-   The framework relies on **Robosuite environments** and **diffusion
    policy learning** to generate demonstrations and train policies.

------------------------------------------------------------------------
