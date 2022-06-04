import os
import typing as ty
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from landslide4sense.config import Config, AugmentationConfig
from landslide4sense.data import LandslideDataSet, Transformation
from landslide4sense.training.base_callbacks import Callback
from landslide4sense.utils import import_name, optimizer_to
from landslide4sense.training.callbacks import (
    EarlyStopping,
    Checkpointer,
    ProgressPrinter,
    WandbCallback,
)


logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_callbacks(cfg: Config) -> ty.List[Callback]:
    wandb_callback = WandbCallback({"config": dict(cfg), **cfg.train.callbacks.wandb})

    early_stopper = EarlyStopping(**cfg.train.callbacks.early_stopping)

    model_checkpointer = Checkpointer(
        os.path.join(cfg.train.snapshot_dir, wandb_callback.run.name), early_stopper
    )

    return [
        early_stopper,
        model_checkpointer,
        ProgressPrinter(),
        wandb_callback,
    ]


def setup_augmentations(
    cfg: ty.Optional[AugmentationConfig],
) -> ty.Optional[Transformation]:
    logger.info("Setting up augmentations")
    if cfg is None:
        logger.info("No augmentations specified")
        return None
    transforms = import_name(cfg.module, cfg.name)
    return Transformation(transforms)


def setup_datasets(
    cfg: Config,
) -> ty.Tuple[DataLoader[LandslideDataSet], ty.List[DataLoader[LandslideDataSet]]]:
    transform = setup_augmentations(cfg.data.augmentation)
    logger.info("Setting up datasets")

    train_dataset = LandslideDataSet(
        cfg.data.dir,
        cfg.data.train_list,
        max_iters=cfg.train.steps_per_epoch * cfg.train.batch_size,
        set="labeled",
        transform=transform,
    )

    logger.info(
        f"Training with {len(train_dataset)} samples from {cfg.data.train_list}"
    )

    train_loader: DataLoader[LandslideDataSet] = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    eval_set_kwargs = dict(
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    eval_sets: ty.List[DataLoader[LandslideDataSet]] = []
    for name, eval_list_path in zip(cfg.data.eval_names, cfg.data.eval_lists_paths):
        eval_set = LandslideDataSet(cfg.data.dir, eval_list_path, set="labeled")
        logger.info(
            f"Setting up {name} set with {len(eval_set)} samples from {eval_list_path}"
        )
        eval_sets.append(DataLoader(eval_set, **eval_set_kwargs))

    return train_loader, eval_sets


def setup_model(cfg: Config) -> nn.Module:
    # Instantiate model
    logger.info("Model setup")
    logger.info(f"Instantiating {cfg.model.module}.{cfg.model.name}...")
    model_cls = import_name(cfg.model.module, cfg.model.name)
    model = model_cls(**cfg.model.args)
    if cfg.model.restore_from:
        logger.info(f"Restoring moddel from checkpoint: {cfg.model.restore_from}...")
        loaded_state_dict = torch.load(
            cfg.model.restore_from, map_location=torch.device(device)
        )
        model.load_state_dict(loaded_state_dict)
    return model


def setup_optimizer(cfg: Config, model: nn.Module) -> optim.Optimizer:
    logger.info("Optimizer setup")
    logger.info(f"Instantiating {cfg.optimizer.module}.{cfg.optimizer.name}...")
    optimizer_cls = import_name(cfg.optimizer.module, cfg.optimizer.name)
    optimizer: optim.Optimizer = optimizer_cls(model.parameters(), **cfg.optimizer.args)
    if cfg.optimizer.restore_from:
        logger.info(
            f"Restoring optimizer from checkpoint: {cfg.optimizer.restore_from}..."
        )
        loaded_state_dict = torch.load(
            cfg.optimizer.restore_from, map_location=torch.device(device)
        )
        optimizer.load_state_dict(loaded_state_dict)

    optimizer_to(optimizer, device)

    if cfg.optimizer.scheduler is not None:
        scheduler_cls = import_name(
            cfg.optimizer.scheduler.module, cfg.optimizer.scheduler.name
        )
        # Wraps optimizer
        optimizer = scheduler_cls(optimizer, **cfg.optimizer.scheduler.args)

    return optimizer


def setup_loss_fn(cfg: Config) -> nn.Module:
    logger.info("Loss function setup")
    logger.info(f"Instantiating {cfg.loss.module}.{cfg.loss.name}...")
    loss_fn_cls = import_name(cfg.loss.module, cfg.loss.name)
    loss_args = dict(cfg.loss.args.copy())
    if "weight" in loss_args:
        loss_args["weight"] = torch.tensor(loss_args["weight"], device=device).float()
    loss_fn = loss_fn_cls(**loss_args)

    return loss_fn
