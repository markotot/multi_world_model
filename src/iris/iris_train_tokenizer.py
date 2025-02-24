from hydra import initialize, compose
from trainer import Trainer

def main():

    with initialize(config_path="config"):
        cfg = compose(config_name="trainer")

        if cfg.env.train.id is None:
            cfg.env.train.id = "BreakoutNoFrameskip-v4"
        cfg.common.device = "cuda:0"
        cfg.wandb.mode = "online"

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()