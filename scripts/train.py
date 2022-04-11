import os
import typing as ty
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from landslide4sense.config import Config
from landslide4sense.data import LandslideDataSet
from landslide4sense.training import ModelTrainer
from landslide4sense.training.base_callbacks import Callback
from landslide4sense.utils import import_name, set_deterministic
from landslide4sense.training.callbacks import (
    EarlyStopping,
    ModelCheckpointer,
    ProgressPrinter,
    WandbCallback,
)

import hydra
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

name_classes = ["Non-Landslide", "Landslide"]


def setup_callbacks(cfg: Config) -> ty.List[Callback]:
    wandb_callback = WandbCallback({"config": cfg, **cfg.train.callbacks.wandb})

    early_stopper = EarlyStopping(**cfg.train.callbacks.early_stopping)

    model_checkpointer = ModelCheckpointer(
        os.path.join(cfg.train.snapshot_dir, wandb_callback.run.name), early_stopper
    )

    return [
        early_stopper,
        model_checkpointer,
        ProgressPrinter(),
        wandb_callback,
    ]


def setup_datasets(
    cfg: Config,
) -> ty.Tuple[DataLoader[LandslideDataSet], ty.List[DataLoader[LandslideDataSet]]]:
    train_loader: DataLoader[LandslideDataSet] = DataLoader(
        LandslideDataSet(
            cfg.data.dir,
            cfg.data.train_list,
            max_iters=cfg.train.num_steps_stop * cfg.train.batch_size,
            set="labeled",
        ),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    eval_set_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    eval_sets: ty.List[DataLoader[LandslideDataSet]] = [
        DataLoader(
            LandslideDataSet(cfg.data.dir, eval_list_path, set="labeled"),
            **eval_set_kwargs,
        )
        for eval_list_path in cfg.data.eval_lists_paths
    ]

    return train_loader, eval_sets


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: Config):
    set_deterministic(cfg.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.train.gpu_id)
        cudnn.enabled = True
        cudnn.benchmark = True

    w, h = map(int, cfg.model.input_size.split(","))
    input_size = (w, h)

    # Instantiate model
    model_cls = import_name(cfg.model.module, cfg.model.name)
    model = model_cls(n_classes=cfg.model.num_classes)
    if cfg.train.restore_from:
        logger.info(f"Restoring moddel from checkpoint: {cfg.train.restore_from}...")
        saved_state_dict = torch.load(cfg.train.restore_from)
        model.load_state_dict(saved_state_dict)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)
    train_loader, eval_sets = setup_datasets(cfg)
    callbacks = setup_callbacks(cfg)

    trainer = ModelTrainer(
        model,
        optimizer,
        cross_entropy_loss,
        train_set=train_loader,
        eval_sets=eval_sets,
        eval_names=cfg.data.eval_names,
        input_size=input_size,
        num_classes=cfg.model.num_classes,
        device=device,
    )

    trainer.train(
        max_epochs=cfg.train.num_steps_stop // cfg.train.steps_per_epoch,
        steps_per_epoch=cfg.train.steps_per_epoch,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
