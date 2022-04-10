from dataclasses import dataclass
import typing as ty
from abc import abstractmethod

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm

from ..data.dataset import LandslideDataSet, LabeledDatasetIterable
from .base_callbacks import Callback, CallbackContainer, OptionalDict
from ..data.dataset import LabeledDatasetIterable


@dataclass
class Trainer:
    model: Module
    optimizer: Optimizer

    loss_fn: ty.Callable[[Tensor, Tensor], Tensor]

    train_set: DataLoader[LandslideDataSet]
    eval_sets: ty.List[DataLoader[LandslideDataSet]]
    eval_names: ty.List[str]
    device: str

    stop_training: bool = False
    callback_container: ty.Optional[CallbackContainer] = None

    @abstractmethod
    def train_step(
        self,
        batch_id: int,
        batch: LabeledDatasetIterable,
        batch_logs: ty.Dict[str, ty.Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        eval_name: str,
        eval_set: DataLoader[LandslideDataSet],
        batch_logs: ty.Dict[str, ty.Any],
    ) -> None:
        raise NotImplementedError

    def train(
        self,
        max_epochs: int = 10,
        steps_per_epoch: int = 500,
        callbacks: ty.Optional[ty.List[Callback]] = None,
    ) -> None:
        callbacks = [] or callbacks
        self.max_epochs = max_epochs
        self.callback_container = CallbackContainer(callbacks)
        self.callback_container.set_trainer(self)

        self.callback_container.on_train_begin(dict())

        for epoch in range(max_epochs):
            epoch_logs: OptionalDict = dict()
            self.callback_container.on_epoch_begin(epoch, epoch_logs)

            for batch_id, batch in tqdm(
                enumerate(self.train_set),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}",
            ):
                batch_logs: OptionalDict = dict()
                self.callback_container.on_batch_begin(batch_id, batch_logs)
                self.train_step(batch_id, batch, batch_logs=batch_logs)
                self.callback_container.on_batch_end(batch_id, batch_logs)
                if (batch_id + 1) == steps_per_epoch:
                    break

            for eval_name, eval_set in zip(self.eval_names, self.eval_sets):
                eval_batch_logs: OptionalDict = dict()
                self.callback_container.on_epoch_begin(0, eval_batch_logs)
                self.eval(eval_name, eval_set, eval_batch_logs)
                self.callback_container.on_epoch_end(0, eval_batch_logs)

            self.callback_container.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        self.callback_container.on_train_end(dict())
