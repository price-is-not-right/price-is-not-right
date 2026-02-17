import PIL.Image
import os

# load /home/train/VLA-probing/init_goal_images_per_config_640x480/brg/episode_029_agentview_goal.png and flip left right
image = PIL.Image.open("/home/train/VLA-probing/init_goal_images_per_config_640x480/brg/episode_029_agentview_goal.png")
image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
image.save("/home/train/VLA-probing/init_goal_images_per_config_640x480/brg/episode_029_agentview_goal.png")