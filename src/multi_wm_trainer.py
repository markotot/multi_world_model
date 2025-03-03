from functools import partial
from pathlib import Path

import wandb
import torch
from hydra.utils import instantiate
from mccabe import PathGraph

from omegaconf import DictConfig, OmegaConf
from iris.utils import set_seed, EpisodeDirManager
from src.iris.collector import Collector
from src.iris.envs import MultiProcessEnv, SingleProcessEnv
from iris.dataset import load_dataset

class Trainer:
    def __init__(self, iris_cfg: DictConfig, diamond_cfg: DictConfig):

        wandb.init(
            config=OmegaConf.to_container(iris_cfg, resolve=True),
            reinit=True,
            resume=False,
            **iris_cfg.wandb
        )

        if iris_cfg.common.seed is not None:
            set_seed(iris_cfg.common.seed)


        self.iris_cfg = iris_cfg
        self.diamond_cfg = diamond_cfg

        self.start_epoch = 1
        self.device = torch.device(iris_cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if not iris_cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=True, parents=False)
            wandb.save(str(config_path))
            self.ckpt_dir.mkdir(exist_ok=True, parents=False)
            self.media_dir.mkdir(exist_ok=True, parents=False)
            self.episode_dir.mkdir(exist_ok=True, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=True, parents=False)

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train',
                                                  max_num_episodes=iris_cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test',
                                                 max_num_episodes=iris_cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination',
                                                             max_num_episodes=iris_cfg.evaluation.actor_critic.num_episodes_to_save)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        self.episodes_dataset = load_dataset("./dataset/storage/breakout_test", should_split_into_episodes=True)


        if iris_cfg.training.should:
            train_env = create_env(iris_cfg.env.train, iris_cfg.collection.train.num_envs)
            self.train_dataset = instantiate(iris_cfg.datasets.train)
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)

            # If using a pregenerated dataset
            self.train_collector.add_pregenerated_experience_to_dataset(self.episodes_dataset)

        if iris_cfg.evaluation.should:
            test_env = create_env(iris_cfg.env.test, iris_cfg.collection.test.num_envs)
            self.test_dataset = instantiate(iris_cfg.datasets.test)
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

        assert iris_cfg.training.should or iris_cfg.evaluation.should
        env = train_env if iris_cfg.training.should else test_env

        tokenizer = instantiate(iris_cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
        self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)
        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
        self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=cfg.training.learning_rate)

        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint()

    def run(self):
        print("Running Trainer")