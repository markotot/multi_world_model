from hydra import initialize, compose
from iris.trainer import Trainer

import sys; print("Exec:\t\t" + sys.executable)
import os; print("CWD:\t\t" + os.getcwd())
print("SYS_PATH:\t\t" + str(sys.path))

def main():

    with initialize(config_path="iris/config"):
        cfg = compose(config_name="trainer")

        if cfg.env.train.id is None:
            cfg.env.train.id = "BreakoutNoFrameskip-v4"
        cfg.common.device = "cuda:0"
        cfg.wandb.mode = "online"

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()