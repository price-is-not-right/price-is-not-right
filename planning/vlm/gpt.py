import base64
from io import BytesIO
import time
import os
import cv2
import PIL.Image
import weave
from typing import *
import numpy as np
from openai import OpenAI
client = OpenAI()


# Function to create a file with the Files API
def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# Function to encode the image
# image can be a path or arraylike
def encode_image(image: Union[os.PathLike, np.ndarray]) -> str:
    # if the image is a numpy array, convert it to base64
    if isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode("utf-8")
    if isinstance(image, PIL.Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@weave.op(name="query_gpt")
def query_gpt(prompt, init_image: Union[os.PathLike, np.ndarray], goal_image: Union[os.PathLike, np.ndarray], model="gpt-5") -> str:
    # image can be a path or an array
    base64_image_1 = encode_image(init_image)
    base64_image_2 = encode_image(goal_image)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image_1}"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image_2}"
                    }
                ]
            }
        ]
    )
    out =  response.output_text
    return out


if __name__ == "__main__":
    # test the query_gpt function
    init_image_path = "/home/train/VLA-probing/init_goal_images_per_config_640x480/bgy/episode_001_agentview_init.png"
    goal_image_path = "/home/train/VLA-probing/init_goal_images_per_config_640x480/bgy/episode_001_agentview_goal.png"
    planning_prompt="""You are a helpful robot planner playing towers of hanoi. The colored cubes represent the disks and the three rectangular areas (the left area, the middle area, the right area) represent the three pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. The blue cube is smaller than the red cube. The red cube is smaller than the green cube. The green cube is smaller than the yellow cube. However, only three of the four cubes are present. You should identify which three colors are present. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube in either an empty area or on top of a cube that is bigger. You should plan a sequence of actions to achieve the goal shown in the second image from the initial state shown in the first image. You are able to execute the following actions: 
    pick the {color} cube
    place the {color} cube in the {left|middle|right} area. 
    place the {color} cube on top of the {color} cube.
    Please provide a step-by-step plan where each action step is separated by a newline to achieve the goal. Let's think step by step. Output your plan in the following example format:
    pick the blue cube
    place the blue cube in the right area
"""
    #init_image_path = "/home/train/VLA-probing/init_goal_images_per_config_640x480/brgy/episode_000_agentview_init.png"
    #goal_image_path = "/home/train/VLA-probing/init_goal_images_per_config_640x480/brgy/episode_000_agentview_goal.png"

    planning_prompt="""You are a helpful robot planner playing towers of hanoi. The colored cubes represent the disks and the three rectangular areas (the left area, the middle area, the right area) represent the three pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. The blue cube is smaller than the red cube. The red cube is smaller than the green cube. The green cube is smaller than the yellow cube. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube in either an empty area or on top of a cube that is bigger. You should plan a sequence of actions to achieve the goal shown in the second image from the initial state shown in the first image. You are able to execute the following actions: 
    pick the {color} cube
    place the {color} cube in the {left|middle|right} area. 
    place the {color} cube on top of the {color} cube.
    Please provide a step-by-step plan where each action step is separated by a newline to achieve the goal. Let's think step by step. Output your plan in the following example format:
    pick the blue cube
    place the blue cube in the right area
"""
    out = query_gpt(planning_prompt, init_image_path, goal_image_path, model="gpt-5")
    print(out)