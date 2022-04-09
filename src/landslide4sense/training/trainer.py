from ast import Mod
from dataclasses import dataclass
import typing as ty

from torch.nn import Module
from torch.optim import Optimizer
from torch.data import DataLaoder


@dataclass
class Trainer:
    model: Module
    optimizer: Optimizer
    train_set: DataLaoder
    eval_sets: ty.List[DataLaoder]
    