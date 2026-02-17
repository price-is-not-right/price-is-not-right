import dataclasses
import os
import atexit
import signal
import subprocess
from typing import *
# add planning to path
import sys
import numpy as np
import wandb
import weave
import time
import json
import PIL.Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers.image_utils import load_image
# from gpt import query_gpt
from qwen import query_qwen
from paligemma import query_paligemma

planning_prompt="""You are a helpful robot planner playing towers of hanoi. The colored cubes represent the disks and the three rectangular areas (the left area, the middle area, the right area) represent the three pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. The blue cube is smaller than the red cube. The red cube is smaller than the green cube. The green cube is smaller than the yellow cube. However, only three of the four cubes are present. You should pay attention to which three colors are present. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube in either an empty area or on top of a cube that is bigger. You should plan a sequence of actions to achieve the goal shown in the second image from the initial state shown in the first image. You are able to execute the following actions: 
pick the {color} cube
place the {color} cube in the {left|middle|right} area. 
place the {color} cube on top of the {color} cube.
Please provide a step-by-step plan where each action step is separated by a newline to achieve the goal. Let's think step by step. Output your plan in the following example format:
pick the blue cube
place the blue cube in the right area
"""

four_cube_planning_prompt="""You are a helpful robot planner playing towers of hanoi. The colored cubes represent the disks and the three rectangular areas (the left area, the middle area, the right area) represent the three pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. The blue cube is smaller than the red cube. The red cube is smaller than the green cube. The green cube is smaller than the yellow cube. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube in either an empty area or on top of a cube that is bigger. You should plan a sequence of actions to achieve the goal shown in the second image from the initial state shown in the first image. You are able to execute the following actions: 
pick the {color} cube
place the {color} cube in the {left|middle|right} area. 
place the {color} cube on top of the {color} cube.
Please provide a step-by-step plan where each action step is separated by a newline to achieve the goal. Let's think step by step. Output your plan in the following example format:
pick the blue cube
place the blue cube in the right area
"""

@dataclasses.dataclass
class Args:
    model: str = 'paligemma'  # gpt-4, gpt-5, qwen, paligemma, dummy
    path_to_images: str = '/home/train/vlas/icra/CyclicLxM/openpi/src/openpi/planning/init_goal_images_per_config_640x480'

@weave.op(name="query_model")
def query_model(init_image: Union[os.PathLike, np.array], goal_image: Union[os.PathLike, np.array], prompt=planning_prompt, model='gpt-5') -> List[str]:
    """
    Query the model with the given prompt and images.
    Args:
        prompt (str): The planning prompt.
        init_image (str): Path to the initial state image.
        goal_image (str): Path to the goal state image.
    Returns:
        list of str: The plan as a list of steps.
    """
    if model.startswith('gpt'):
         pass
        #  out = query_gpt(prompt, init_image, goal_image, model=model)
    elif model == 'qwen':
        out = query_qwen(prompt, init_image, goal_image)
    elif model == 'paligemma':
        out = query_paligemma(prompt, init_image, goal_image)
    elif model == 'dummy':
        # doesn't call any model, just return a dummy plan to get a baseline for the inference time and power consumption
        out = """pick the blue cube
        place the blue cube in the right area
        pick the red cube
        place the red cube in the middle area
        pick the blue cube
        place the blue cube on top of the red cube
        pick the green cube
        place the green cube in the right area
        pick the blue cube
        place the blue cube in the left area
        pick the red cube
        place the red cube on top of the green cube
        pick the blue cube
        place the blue cube on top of the red cube"""
    return process_output(out)

def load_image_pairs(path_to_images: str) -> List[Tuple[str, str]]:
    """
    Load image pairs from the given path.
    Each image pair consists of an initial state image and a goal state image.
    Args:
        path_to_images (str): Path to the directory containing images.
    Returns:
        list of tuples: Each tuple contains (init_image_path, goal_image_path).
    """
    image_pairs = {}
    for config_dir in os.listdir(path_to_images):
        config_path = os.path.join(path_to_images, config_dir)
        if not os.path.isdir(config_path):
            continue
        image_pairs[config_dir] = []
        for filename in os.listdir(config_path):
            # replace 'end' with 'init'
            if filename.endswith('init.png'):
                init_image_path = os.path.join(config_path, filename)
                goal_image_path = os.path.join(config_path, filename.replace('init', 'goal'))
                if os.path.exists(goal_image_path):
                    image_pairs[config_dir].append((init_image_path, goal_image_path))
    return image_pairs

def rename_images(path_to_images: str):
    """
    Rename images in the given path to ensure consistent naming.
    Args:
        path_to_images (str): Path to the directory containing images.
    """
    for config_dir in os.listdir(path_to_images):
        config_path = os.path.join(path_to_images, config_dir)
        if not os.path.isdir(config_path):
            continue
        for filename in os.listdir(config_path):
            # replace 'end' with 'init' and 'start' with 'goal'. The images are reversed. We want to have init and goal.
            if 'end' in filename:
                new_filename = filename.replace('end', 'init')
                os.rename(os.path.join(config_path, filename), os.path.join(config_path, new_filename))
            elif 'start' in filename:
                new_filename = filename.replace('start', 'goal')
                os.rename(os.path.join(config_path, filename), os.path.join(config_path, new_filename))
    

def flip_resize_image(image: os.PathLike) -> PIL.Image.Image:
    """Flip the input image horizontally and resize to 224 x 224.
    Args:
        image: the input image, either as a file path
    Returns:
        the flipped PIL.Image.Image object
    """
    # check if image is a path, it could be a string path
    if not isinstance(image, (os.PathLike, str)):
        raise TypeError("Input must be a file path.")
    image_pil: PIL.Image.Image = load_image(image)
    # flip from left to right
    flipped = image_pil.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    # resize to 224 x 224
    resized = flipped.resize((224, 224))
    # save the flipped in the original path as a png
    resized.save(image)
    return resized

def process_output(output_text: str) -> List[str]:
    plan = output_text.split("\n")
    plan = [step.strip() for step in plan]
    return plan


def _cpu_power_output_dir() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
    return os.path.join(repo_root, "analysis")


def _start_cpu_power_monitor() -> subprocess.Popen | None:
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_power.py")
    output_dir = _cpu_power_output_dir()
    cmd = [sys.executable, script_path, "--until-interrupt", "--output-dir", output_dir]
    try:
        proc = subprocess.Popen(cmd)
        print(f"Started CPU power monitor (pid={proc.pid}) -> {output_dir}")
        return proc
    except Exception as e:
        print(f"Warning: failed to start CPU power monitor: {e}")
        return None


def _stop_cpu_power_monitor(proc: subprocess.Popen | None) -> None:
    if not proc:
        return
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        return
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()



if __name__ == "__main__":
    args = Args()
    model = args.model
    pairs = load_image_pairs(args.path_to_images)
    cpu_power_proc = _start_cpu_power_monitor()
    if cpu_power_proc:
        atexit.register(_stop_cpu_power_monitor, cpu_power_proc)
    # start a wandb run to measure the inference time, the power consumption, the energy consumption, etc.
    # --- Init Weave (tracing) ---
    # Tip: set WANDB_ENTITY via env or pass entity="your_team"
    weave.init(
        project_name=f"hanoi_{model}_planner",
        autopatch_settings={"openai": True},  # auto-trace OpenAI calls made inside query_model
    )

    # --- Init Weights & Biases (logging) ---
    wandb.init(
        project=f"hanoi_{model}_planner",
        name=f"hanoi_{model}_planner_run",
        config=dataclasses.asdict(args),
    )
    table = wandb.Table(columns=[
        "step", "env config", "init_image", "goal_image", "action", "num_actions", "model", "latency", "status", "init_image_path", "goal_image_path"
    ])
    project_start_time = time.time()
    for config in pairs:
        for i, (init_image, goal_image) in enumerate(pairs[config]):
            # query each model N times per image pair
            
            query_start_time = time.time()

            # >>> traced call (shows up in Weave Traces)
            #     If query_model internally calls OpenAI, those calls are auto-traced too.
            try:
                if len(config) == 4:
                    plan = query_model(init_image, goal_image, prompt=four_cube_planning_prompt, model=model)
                else:
                    plan = query_model(init_image, goal_image, prompt=planning_prompt, model=model)
                status = "ok"
            except Exception as e:
                # Weave will capture the exception; you can also log a placeholder plan
                plan = {"error": repr(e)}
                status = "error"

            query_end_time = time.time()
            infer_time = query_end_time - query_start_time

            # Log the plan and the inference time (keep your table)
            table.add_data(
                i, # step
                config,
                wandb.Image(init_image),
                wandb.Image(goal_image),
                json.dumps(plan),
                len(plan) if isinstance(plan, list) else 0,
                model,
                infer_time,
                status,
                init_image,
                goal_image,
            )
            

    project_end_time = time.time()
    wandb.log({"total_time": project_end_time - project_start_time})
    num_image_pairs = sum(len(pairs[config]) for config in pairs)
    print(f"Total time for {num_image_pairs} image pairs: {project_end_time - project_start_time:.2f} seconds")
    # Print the number of image pairs processed
    print(f"Number of image pairs processed: {num_image_pairs}")

    # Emit the table at the end (or log periodically inside the loop if you prefer)
    wandb.log({"vlm queries": table})
    _stop_cpu_power_monitor(cpu_power_proc)


