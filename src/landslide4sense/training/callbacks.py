from .base_callbacks import Callback

import typing as ty
import logging
from dataclasses import dataclass
import copy
import time

import os
import torch
from torch import Tensor
import numpy as np
import wandb

OptionalDict = ty.Optional[ty.Dict[str, ty.Any]]
logger = logging.getLogger(__name__)


@dataclass
class ProgressPrinter(Callback):
    epoch_name: str = "Epoch"
    start_time: float = None
    end_time: float = None

    def on_epoch_begin(self, epoch: int, logs: OptionalDict = None) -> None:
        print(f"{self.epoch_name}: {epoch}")
        self.start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: OptionalDict = None) -> None:
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        msg = f"\t{self.epoch_name} summary\n"
        msg += f"\tTime took: {elapsed:.2f}s\n"
        for k, v in logs.items():
            msg += f"\t\t{k}: {v}\n"
        print(msg)


@dataclass
class EarlyStopping(Callback):
    monitor: str
    mode: str = "min"
    patience: int = 10
    best_epoch: int = 0
    best_result: float = np.inf
    stopped_epoch: int = 0
    _best_weights: ty.Optional[ty.Dict[str, Tensor]] = None
    _wait: int = 0

    def __post_init__(self):
        assert self.mode in ["min", "max"], "Mode must be one of ['min', 'max']"
        self.best_result = -np.inf if self.mode == "max" else np.inf
        super().__init__()

    def on_epoch_end(self, epoch: int, logs: OptionalDict = None) -> None:
        epoch_res = logs.get(self.monitor)
        if epoch_res is None:
            logger.warning(f"Tried to search for {self.monitor} in logs, but found: {logs}")
            return
        
        if self.mode == "min":
            improved: bool = epoch_res < self.best_result
        else:
            improved: bool = epoch_res > self.best_result

        if improved:
            self.best_epoch = epoch
            self._best_weights = copy.deepcopy(self.trainer.model.state_dict())
            self.best_result = epoch_res
            self._wait = 1
        else:
            if self._wait >= self.patience:
                self.trainer.stop_training = True
                self.stopped_epoch = epoch
            self._wait += 1

    def on_train_end(self, logs: OptionalDict = None) -> None:
        self.trainer.best_epoch = self.best_epoch
        self.trainer.best_result = self.best_result
        self.trainer.monitor_metric = self.monitor

        if self._best_weights is not None:
            self.trainer.model.load_state_dict(self._best_weights)

        if self.stopped_epoch > 0:
            msg = f"\nEarly stopping occurred at epoch {self.stopped_epoch}"
            msg += (
                f" with best_epoch = {self.best_epoch} and "
                + f"best_{self.monitor} = {round(self.best_result, 5)}"
            )
            logger.info(msg)
        else:
            msg = (
                f"Stop training because you reached max_epochs = {self.trainer.max_epochs}"
                + f" with best_epoch = {self.best_epoch} and "
                + f"best_{self.monitor} = {round(self.best_result, 5)}"
            )
            logger.info(msg)
        wrn_msg = "Best weights from best epoch are automatically used!"
        logger.warning(wrn_msg)


@dataclass
class ModelCheckpointer(Callback):
    save_dir: str
    eval_set: str

    def __post_init__(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_epoch_end(self, epoch: int, logs: OptionalDict = None) -> None:
        snapshot_name = "epoch" + repr(epoch) + ".pth"
        model_path = os.path.join(self.save_dir, snapshot_name)
        torch.save(self.trainer.model.state_dict(), model_path)


@dataclass
class WandbCallback(Callback):
    wandb_init_kwargs: ty.Dict[str, ty.Any]
    run: ty.Optional[ty.Any] = None
    log_freq: str = "batch"

    def __post_init__(self):
        self.run = wandb.init(**self.wandb_init_kwargs)

    def on_batch_end(self, batch: int, logs: OptionalDict = None) -> None:
        wandb.log(logs)

    def on_epoch_end(self, epoch: int, logs: OptionalDict = None) -> None:
        wandb.log(logs)

    def on_train_end(self, logs: OptionalDict = None) -> None:
        self.run.finish()
