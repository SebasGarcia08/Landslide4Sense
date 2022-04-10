import os

from landslide4sense.config import Config

from hydra.core.config_store import ConfigStore
import hydra

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: Config):
    print(cfg.data.dir)
    with open(cfg.data.train_list, "r") as f:
        lines = f.readlines()
        print(lines)


if __name__ == "__main__":
    main()
