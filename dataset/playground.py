from dataset.generate_dataset import (load_dataset, split_into_episodes)
# from src.dataset.generate_dataset import generate_dataset

import matplotlib.pyplot as plt


num_frames = 100_000
path = "storage/breakout_test/"
#greyscale_buffer, rgb_buffer, action_buffer, reward_buffer, done_buffer = generate_dataset(num_frames, path)


dataset = load_dataset(path, should_split_into_episodes=False)
episodes = split_into_episodes(dataset)

greyscale_buffer, rgb_buffer, action_buffer, reward_buffer, done_buffer = dataset
# Assuming greyscale_buffer has the shape (num_envs, frames_per_env, height, width)
for i in range(10_000, 10_005):
    plt.figure(figsize=(10, 5))

    # Plot greyscale image
    plt.subplot(1, 2, 1)
    plt.imshow(greyscale_buffer[0, i, :, :], cmap='gray')
    plt.title(f'Greyscale Image {i+1}')

    # Plot RGB image
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_buffer[0, i, :, :])
    plt.title(f'RGB Image {i+1}')

    plt.show()