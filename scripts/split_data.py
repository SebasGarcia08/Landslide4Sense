import os
import logging
import random

from landslide4sense.config import Config
from landslide4sense.utils import set_deterministic

from hydra.core.config_store import ConfigStore
import hydra

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

logger = logging.getLogger(__name__)

NUM_VALIDATION_SAMPLES = 300


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: Config):
    set_deterministic(cfg.train.seed)

    with open(cfg.data.train_list, "r") as f:
        train_imgs_list = f.readlines()

    logger.info(f"Total number of training images: {len(train_imgs_list)}")
    random.shuffle(train_imgs_list)

    val_imgs_list = train_imgs_list[:NUM_VALIDATION_SAMPLES]

    logger.info("Selected training images for validation:")
    logger.info(val_imgs_list)

    for val_img_path in val_imgs_list:
        train_imgs_list.remove(val_img_path)

    new_train_list_path = cfg.data.train_list.replace(".txt", "") + "_split_train.txt"
    with open(new_train_list_path, "w") as f:
        for train_img_path in train_imgs_list:
            f.write(train_img_path)

    new_val_list_path = cfg.data.train_list.replace(".txt", "") + "_split_val.txt"
    with open(new_val_list_path, "w") as f:
        for val_img_path in val_imgs_list:
            f.write(val_img_path)


if __name__ == "__main__":
    main()
