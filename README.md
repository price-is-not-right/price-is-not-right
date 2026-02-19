# Price is not Right: Neuro-Symbolic Methods Outperform VLAs on Structured Long-Horizon Manipulation Tasks with Significantly Lower Energy Consumption

## Setup
```bash
git submodule update --init --recursive
```

## Evaluation

### VLA

#### Download one of our models
Navigate to the checkpoints directory
```bash
cd openpi/checkpoints
```

Clone the model(s) you want to evaluate

End-to-End Model
```bash
git clone https://huggingface.co/tduggan93/pi0-hanoi-end-to-end
```

Planner-Guided Model
```bash
git clone https://huggingface.co/tduggan93/pi0-hanoi-planner-guided
```

#### Set Server Arguments for the pi0 model in config.yml
In openpi/examples/robotsuite/config.yml lines 51 - 52, you will see the following below:
```yml
- SERVER_ARGS=policy:checkpoint --policy.config pi0_hanoi_end_to_end --policy.dir /app/checkpoints/pi0-hanoi-end-to-end
# - SERVER_ARGS=policy:checkpoint --policy.config pi0_hanoi_planner_guided --policy.dir /app/checkpoints/pi0-hanoi-planner-guided
```

Uncomment the model you want to use and comment out the one you don't want to use. Also feel free to add your own model if you train one.

#### Set the evaluation arguments
In openpi/examples/robotsuite/main.py lines 59 - 112, you will see the following below:

The most important important arguments for you will be:
1. env_name - Hanoi or Hanoi4x3
2. env - Hanoi or Hanoi4x3
3. use_sequential_tasks - True for Planner Guided, False for End-to-End
4. random_block_selection - For 3 block hanoi randomly use 3 of the 4 available block types each episode
5. episodes - Number of episodes to run
6. wandb_project_name - the name of your wandb project

```python
@dataclasses.dataclass
class Args:
    """Arguments for running Robosuite with OpenPI Websocket Policy and multi-config support"""
    # --- Server Connection ---
    host: str = "127.0.0.1"         # Hostname of the OpenPI policy server
    port: int = 8000                # Port of the OpenPI policy server

    # --- planner ---
    planner:str = "pddl"            # Planner to use: 'pddl' or 'gpt-5'

    # --- Policy Interaction ---
    resize_size: int = 224               # Target size for image resizing (must match model training)
    replan_steps: int = 50               # Number of steps per action chunk from policy server
    use_sequential_tasks: bool =False    # If True, use sequential task prompts; if False, use single prompt
    time_based_progression: bool = False # If True, advance to next task after task_timeout steps regardless of completion
    task_timeout: int = 750              # Number of steps to wait before timing out a task

    # --- Robosuite Environment ---
    env_name: str = "Hanoi4x3" 
    env: str = "Hanoi4x3"                # Environment name for RecordDemos compatibility
    robots: str = "Panda"                # Robot model to use
    controller: str = "OSC_POSE"         # Robosuite controller name
    horizon: int = 7050                  # Max steps per episode
    skip_steps: int = 50                 # Number of initial steps to skip (wait for objects to settle)

    # --- Multi-configuration support ---
    random_block_placement: bool = False   # Place blocks on pegs randomly according to Towers of Hanoi rules
    random_block_selection: bool = False   # Randomly select 3 out of 4 blocks
    cube_init_pos_noise_std: float = 0.01  # Std dev for XY jitter of initial tower position

    # --- Rendering & Video ---
    render_mode: str = "headless"                 # Rendering mode: 'headless' (save video) or 'human' (live view)
    video_out_path: str = "data/robosuite_videos" # Directory to save videos
    camera_names: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Cameras for observation/video
    camera_height: int = 256  # Rendered camera height (before potential resize)
    camera_width: int = 256   # Rendered camera width (before potential resize)
    
    # --- Full Resolution Video Recording ---
    save_full_res_video: bool = True  # Save full resolution videos alongside model observations
    full_res_height: int = 480        # Full resolution video height
    full_res_width: int = 640         # Full resolution video width
    required_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
        ) # Required cameras for OpenPI preprocessing

    # --- Misc ---
    seed: int = 3           #: Random seed
    episodes: int = 50      #: How many episodes to run back-to-back

    # --- Logging ---
    wandb_project: str = "Your wandb project name"  #: W&B project name
    log_every_n_seconds: float = 0.5                #: W&B system metric sampling interval (seconds)
```

#### Run the experiments
From the openpi directory:

```bash
docker compose -f examples/robosuite/compose.yml up
```