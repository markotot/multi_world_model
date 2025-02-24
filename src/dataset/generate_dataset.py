import gym
from huggingface_sb3 import load_from_hub

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm


class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, rgb_shape, greyscale_shape):
        super().__init__(env)
        self.rgb_shape = rgb_shape
        self.greyscale_shape = greyscale_shape
        self.observation_space = gym.spaces.Dict({
            'rgb': gym.spaces.Box(low=0, high=255, shape=rgb_shape, dtype=np.uint8),
            'greyscale': gym.spaces.Box(low=0, high=255, shape=greyscale_shape, dtype=np.uint8)
        })

    def observation(self, obs):
        rgb = cv2.resize(obs, (self.rgb_shape[0], self.rgb_shape[1]), interpolation=cv2.INTER_AREA)
        greyscale = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        greyscale = cv2.resize(greyscale, (self.greyscale_shape[0], self.greyscale_shape[1]), interpolation=cv2.INTER_AREA)
        greyscale = np.expand_dims(greyscale, axis=-1)
        return {'rgb': rgb, 'greyscale': greyscale}

def make_atari_env(env_id, rgb_shape, greyscale_shape):
    env = gym.make(env_id)
    env = MaxAndSkipEnv(env, skip=4)

    return CustomObservationWrapper(env, rgb_shape, greyscale_shape)

# Generates dataset for BreakoutNoFrameskip-v4
# Dataset contains: grey_obs_buffer[n+1], rgb_obs_buffer[n+1], actions_buffer[n], rewards[n], dones[n]
def generate_dataset(num_frames, save_path):
    # Load the model
    checkpoint = load_from_hub("ThomasSimonini/ppo-BreakoutNoFrameskip-v4", "ppo-BreakoutNoFrameskip-v4.zip")
    model = PPO.load(checkpoint)

    # Setup the environment
    env_id = "BreakoutNoFrameskip-v4"
    num_envs = 8
    rgb_shape = (64, 64, 3)
    greyscale_shape = (84, 84, 1)
    envs = DummyVecEnv([lambda: make_atari_env(env_id, rgb_shape, greyscale_shape) for _ in range(num_envs)])

    # Evaluation loop
    frames_per_env = num_frames // num_envs
    initial_steps = 5
    frame_stack = 4

    grey_obs_buffer = np.empty(shape=(num_envs, frames_per_env + 1, *greyscale_shape[:-1]), dtype=np.uint8)
    rgb_obs_buffer = np.empty(shape=(num_envs, frames_per_env + 1, *rgb_shape), dtype=np.uint8)
    actions_buffer = np.empty(shape=(num_envs, frames_per_env), dtype=np.uint8)
    rewards_buffer = np.empty(shape=(num_envs, frames_per_env), dtype=np.float32)
    dones_buffer = np.empty(shape=(num_envs, frames_per_env), dtype=np.bool_)

    obs = envs.reset()
    grey_obs_buffer[:, 0, :, :] = obs['greyscale'].squeeze(-1)
    rgb_obs_buffer[:, 0, :, :, :] = obs['rgb']

    total_rewards = []
    rewards_per_episode = np.zeros(num_envs)
    for step in tqdm(range(frames_per_env), desc="Generating dataset"):

        # Make a random action for the first few steps
        if step < initial_steps:
            action = np.random.randint(0, 4, num_envs)
        else:
            obs_stack = grey_obs_buffer[:, step - frame_stack:step, :, :] # Assume frame_stack < initial_steps
            action, _ = model.predict(obs_stack)

        obs, rewards, dones, infos = envs.step(action)
        rewards_per_episode += rewards
        for i, done in enumerate(dones):
            if done:
                total_rewards.append(rewards_per_episode[i])
                rewards_per_episode[i] = 0

        grey_obs_buffer[:, step + 1, :, :] = obs['greyscale'].squeeze(-1)
        rgb_obs_buffer[:, step + 1, :, :, :] = obs['rgb']
        actions_buffer[:, step] = action
        rewards_buffer[:, step] = rewards
        dones_buffer[:, step] = dones


    np.save(f"{save_path}/grey_obs_buffer.npy", grey_obs_buffer)
    np.save(f"{save_path}/rgb_obs_buffer.npy", rgb_obs_buffer)
    np.save(f"{save_path}/actions_buffer.npy", actions_buffer)
    np.save(f"{save_path}/rewards.npy", rewards_buffer)
    np.save(f"{save_path}/dones.npy", dones_buffer)

    return grey_obs_buffer, rgb_obs_buffer, actions_buffer, rewards_buffer, dones_buffer

def load_dataset(path):
    grey_obs_buffer = np.load(f"{path}/grey_obs_buffer.npy")
    rgb_obs_buffer = np.load(f"{path}/rgb_obs_buffer.npy")
    actions_buffer = np.load(f"{path}/actions_buffer.npy")
    rewards = np.load(f"{path}/rewards.npy")
    dones = np.load(f"{path}/dones.npy")
    return grey_obs_buffer, rgb_obs_buffer, actions_buffer, rewards, dones