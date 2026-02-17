from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import weave

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2", 
#     device_map="auto",
# )
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)


# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


# Messages containing multiple images and a text query
def construct_messages(prompt, init_image_path, goal_image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file:///{init_image_path}"},
                {"type": "image", "image": f"file:///{goal_image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


@weave.op(name="query_qwen")
def query_qwen(prompt, init_image, goal_image) -> str:
    # Preparation for inference
    messages = construct_messages(prompt, init_image, goal_image)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=5000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text


if __name__ == "__main__":
    # test the query_qwen function
    init_image_path = "/home/train/VLA-probing/init_goal_images_per_config_640x480/bry/episode_007_agentview_init.png"
    goal_image_path = "/home/train/VLA-probing/init_goal_images_per_config_640x480/bry/episode_007_agentview_goal.png"
    planning_prompt = """You are a helpful robot planner playing towers of hanoi. The differently colored cubes represent the differently sized disks and the three rectangular areas (the left area, the middle area, the right area) represent the three pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube in either an empty area or on top of a cube that is bigger. You should plan a sequence of actions to achieve the goal shown in the second image from the initial state shown in the first image. You are able to execute the following actions: 
    pick the {color} cube
    place the {color} cube in the {left|middle|right} area. 
    place the {color} cube on top of the {color} cube.
    Please provide a step-by-step plan where each action step is separated by a newline to achieve the goal. Let's think step-by-step. Output your plan in the following example format:
    pick the blue cube
    place the blue cube in the right area
    """
    out = query_qwen(planning_prompt, init_image_path, goal_image_path)
    print(out)
        
