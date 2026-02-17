import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file containing multiple images
images = np.load('/cyclic_lxm/datasets/Hanoi_univla_seed_0/2025-07-28_19:10:07/data/episode_0.npy', allow_pickle=True)

# Assuming images.shape is (N, H, W, C)
# for i in range(len(images)):
plt.figure()
plt.imshow(images[0]['image'])
plt.title(f'Image {0}')
plt.axis('off')
plt.savefig('output_image.png')
