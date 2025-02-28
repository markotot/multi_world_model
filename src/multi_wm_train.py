import os

print(os.getcwd())

from hydra import initialize, compose
from iris.trainer import Trainer

def main():

    with initialize(config_path="./iris/config"):
        iris_cfg = compose(config_name="trainer")

        if iris_cfg.env.train.id is None:
            iris_cfg.env.train.id = "BreakoutNoFrameskip-v4"
        iris_cfg.common.device = "cuda:0"
        iris_cfg.wandb.mode = "online"

    with initialize(config_path="./diamond/config"):
        diamond_cfg = compose(config_name="trainer")

        if diamond_cfg.env.train.id is None:
            diamond_cfg.env.train.id = "BreakoutNoFrameskip-v4"
        diamond_cfg.common.device = "cuda:0"
        diamond_cfg.wandb.mode = "online"

    trainer = Trainer(iris_cfg)
    trainer.run()


if __name__ == "__main__":
    main()