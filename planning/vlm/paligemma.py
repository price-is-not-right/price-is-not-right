import sys
import os

import weave
# add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import PIL.Image
from transformers.image_utils import load_image
import numpy as np
from typing import *
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/paligemma-3b-mix-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

def preprocess_image(image: Union[os.PathLike, str, PIL.Image.Image]) -> PIL.Image.Image:
    """Preprocess the input image to resize. 
    Args:
        image: the input image, either as a file path or a PIL.Image.Image object
    Returns:
        the resized PIL.Image.Image object
    """
    if isinstance(image, (os.PathLike, str)):
        # load the image from the path with PIL
        image = PIL.Image.open(image).convert("RGB")
    if not isinstance(image, PIL.Image.Image):
        raise TypeError("Input must be a PIL.Image.Image or a file path.")
    return image

@weave.op(name="query_paligemma")
def query_paligemma(prompt:str, init_image: Union[os.PathLike, PIL.Image.Image], goal_image: Union[os.PathLike, PIL.Image.Image]) -> str:
    """Query the PaliGemma model with a prompt and two images (initial and goal state).
    Args:
        prompt: the text prompt to provide to the model
        init_image: the initial state image (path or array)
        goal_image: the goal state image (path or array)
    Returns:
        the model's output text
    """
    init_image = preprocess_image(init_image)
    goal_image = preprocess_image(goal_image)
    # add two <image> tokens to the beginning of the text
    prompt = "<image> <image> " + prompt
    inputs = processor(text=[prompt], images=[[init_image, goal_image]],
                      padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
    model.to(device)
    inputs = inputs.to(dtype=model.dtype)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5000)
        output = output[:, inputs["input_ids"].shape[1]:]
    return processor.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # test the query_paligemma function
    init_image_path = "/home/train/VLA-probing/init_goal_images_per_config_640x480/bry/episode_007_agentview_init.png"
    goal_image_path = "/home/train/VLA-probing/init_goal_images_per_config_640x480/bry/episode_007_agentview_goal.png"
    planning_prompt = """You are a helpful robot planner playing towers of hanoi. The differently colored cubes represent the differently sized disks and the three rectangular areas (the left area, the middle area, the right area) represent the three pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube in either an empty area or on top of a cube that is bigger. You should plan a sequence of actions to achieve the goal shown in the second image from the initial state shown in the first image. You are able to execute the following actions: 
    pick the {color} cube
    place the {color} cube in the {left|middle|right} area. 
    place the {color} cube on top of the {color} cube.
    Please provide a step-by-step plan where each action step is separated by a newline to achieve the goal. Let's think step by step. Output your plan in the following example format:
    pick the blue cube
    place the blue cube in the right area
    """
    out = query_paligemma(planning_prompt, init_image_path, goal_image_path)
    print(out)