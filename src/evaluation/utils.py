import math

import numpy as np
import torch

from functools import partial
from pathlib import Path

from einops import einops
from hydra import initialize, compose
from hydra.utils import instantiate
from matplotlib import pyplot as plt

from src.iris.envs import SingleProcessEnv

from src.iris.envs.world_model_env import WorldModelEnv as IrisWorldModelEnv
from src.iris.models.actor_critic import ActorCritic as IrisActorCritic
from src.iris.models.world_model import WorldModel as IrisWorldModel
from src.iris.agent import Agent as IrisAgent

from src.diamond.models.diffusion import Denoiser as DiamondDenoiser
from src.diamond.models.actor_critic import ActorCritic as DiamondActorCritic
from src.diamond.models.diffusion import DiffusionSampler as DiamondDiffusionSampler
from src.diamond.models.rew_end_model import RewEndModel as DiamondRewEndModel


def get_checkpoint_path_from_env_id(env_id: str, encoder_type: str):

    env_ckpt_name = env_id.split("NoFrameskip")[0].lower()
    model_path = f"checkpoints/{encoder_type}/{env_ckpt_name}.pt"
    config_path = f"checkpoints/{encoder_type}/config/"

    return model_path, config_path

def setup_gym(env_id, encoder_type):

    _, config_path = get_checkpoint_path_from_env_id(env_id, encoder_type)

    with initialize(config_path=f"../../{config_path}"):
        cfg = compose(config_name="trainer")
    env_fn = partial(instantiate, config=cfg.env.test)
    return SingleProcessEnv(env_fn)


def setup_iris(env_id, encoder_type, device):

    model_path, model_cfg = get_checkpoint_path_from_env_id(env_id, encoder_type)
    model_cfg = f"../../{model_cfg}"
    with initialize(config_path=model_cfg):
        cfg = compose(config_name="trainer")

    cfg.env.test.id = env_id  # override iris config to the LightZero config

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    tokenizer = instantiate(cfg.tokenizer)
    world_model = IrisWorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions,
                             config=instantiate(cfg.world_model))

    actor_critic = IrisActorCritic(**cfg.actor_critic, act_vocab_size=test_env.num_actions)
    actor_critic.reset(1)
    agent = IrisAgent(tokenizer, world_model, actor_critic).to(device)

    # so many ../.. because we are in "./LightZero/zoo/atari/entry/outputs/2024-12-19/12-39-16"
    agent.load(Path(f"../../{model_path}"), device)

    world_model_env = IrisWorldModelEnv(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                    env=env_fn())
    return agent, world_model_env


def  run_gym_env(env, max_steps, agent, device):

    obs = env.reset() # (1, 64, 64, 3) dtype=uint8
    obs = einops.rearrange(obs.squeeze(), 'h w c -> c h w') # (1, 64, 64, 3) -> (3, 64, 64)
    obs = obs / 255.0 # normalize [0, 255] -> [0, 1]
    env_observations = [obs]
    actions = []
    total_reward = 0
    num_steps = 0

    initial_lives = 5
    lost_lives_timesteps = []
    while num_steps < max_steps:

        obs_tensor = torch.from_numpy(obs).to(torch.float32).to(device)
        ac_output = agent.act(obs_tensor.unsqueeze(0), should_sample=False, temperature=1)
        action = ac_output.detach().cpu().numpy()

        obs_dict, rew, done, info = env.step(action)
        env_observations.append(obs)
        actions.append(action)
        total_reward += rew

        if info['lives'] < initial_lives:
            lost_lives_timesteps.append(num_steps)
            initial_lives = info['lives']
        num_steps += 1
        if num_steps % 1000 == 0:
            print(num_steps)
        if done:
            print(num_steps)
            break

    return env_observations, actions, lost_lives_timesteps
# def setup_diamond(env_id, num_actions, device):
#
#     path = "../../../diamond/config/"
#     with initialize(config_path=path, job_name="test_app"):
#         cfg = compose(config_name="trainer")
#
#     cfg.env.test.id = env_id  # override iris config to the LightZero config
#     model_path = diamond_config.get_model_path_from_env_id(env_id)
#
#     with open_dict(cfg):
#         cfg.agent.denoiser.inner_model.num_actions = num_actions
#         cfg.agent.rew_end_model.num_actions = num_actions
#         cfg.agent.actor_critic.num_actions = num_actions
#
#     denoiser = DiamondDenoiser(cfg.agent.denoiser)
#     rew_end_model = DiamondRewEndModel(cfg.agent.rew_end_model)
#     actor_critic = DiamondActorCritic(cfg.agent.actor_critic)
#
#     sd = torch.load(Path(f"../../../{model_path}"), map_location=device)
#     sd = {k: extract_state_dict(sd, k) for k in ("denoiser", "rew_end_model", "actor_critic")}
#     denoiser.load_state_dict(sd["denoiser"])
#     rew_end_model.load_state_dict(sd["rew_end_model"])
#     actor_critic.load_state_dict(sd["actor_critic"])
#
#     denoiser.to(device)
#     rew_end_model.to(device)
#     actor_critic.to(device)
#
#     sampler = DiamondDiffusionSampler(denoiser, cfg.world_model_env.diffusion_sampler)
#     return sampler, rew_end_model, actor_critic

def pad_images(images: np.ndarray, top=0, bottom=0, left=0, right=0, constant=0) -> np.ndarray:
    assert len(images.shape) == 4, "not a batch of images!"
    return np.pad(images, ((0, 0), (top, bottom), (left, right), (0, 0)), mode="constant", constant_values=constant)

def plot_images(images, current_step, num_steps, transpose, title):

    images = images[:num_steps]
    if transpose:
        images = [np.transpose(obs, (1, 2, 0)) for obs in images]

    empty = np.array(images[0].copy())
    empty.fill(0)

    cols = math.sqrt(num_steps)
    if np.floor(cols) < cols:
        cols = math.floor(cols) + 1
    else:
        cols = math.floor(cols)  # for some reason this is needed

    rows = math.ceil(num_steps / cols)

    images.extend(((cols * rows) - len(images)) * [empty])

    padded_images = pad_images(np.array(images), top=4, bottom=4, left=4, right=4)
    image_rows = []
    resize_factor = 1
    for i in range(rows):
        image_slice = padded_images[i * cols: (i + 1) * cols]
        image_row = np.concatenate(image_slice, 1)
        x, y, _ = image_row.shape
        image_row_resized = image_row[::resize_factor, ::resize_factor]
        image_rows.append(image_row_resized)

    image = np.concatenate(image_rows, 0)

    plt.figure(dpi=300)
    plt.imshow(image)
    plt.axis('off')  # Optional: Turn off the axis
    plt.title(f"{title} - Step {current_step}")
    plt.show()
