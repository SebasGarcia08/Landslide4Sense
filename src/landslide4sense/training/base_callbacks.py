import typing as ty
from dataclasses import dataclass


OptionalDict = ty.Optional[ty.Dict[str, ty.Any]]


class Callback(object):

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, logs: OptionalDict = None) -> None:
        pass

    def on_train_end(self, logs: OptionalDict = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: OptionalDict = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: OptionalDict = None) -> None:
        pass

    def on_batch_begin(self, batch: int, logs: OptionalDict = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: OptionalDict = None) -> None:
        pass


@dataclass
class CallbackContainer(Callback):
    callbacks: ty.List[Callback]

    def set_trainer(self, trainer):
        for c in self.callbacks:
            c.set_trainer(trainer)

    def on_train_begin(self, logs: OptionalDict = None) -> None:
        logs = logs or dict()
        for c in self.callbacks:
            c.on_train_begin(logs)

    def on_train_end(self, logs: OptionalDict = None) -> None:
        logs = logs or dict()
        for c in self.callbacks:
            c.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: OptionalDict = None) -> None:
        logs = logs or dict()
        for c in self.callbacks:
            c.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: OptionalDict = None) -> None:
        logs = logs or dict()
        for c in self.callbacks:
            c.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: OptionalDict = None) -> None:
        logs = logs or dict()
        for c in self.callbacks:
            c.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: OptionalDict = None) -> None:
        logs = logs or dict()
        for c in self.callbacks:
            c.on_batch_end(batch, logs)
