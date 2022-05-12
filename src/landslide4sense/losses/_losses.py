from turtle import forward
import torch
from torch.nn import Module
from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss
import typing as ty
from ..config import LossConfig
from ..utils import import_name

class LogCoshDiceLoss(Module):
    def __init__(self, *args, **kwargs):
        super(LogCoshDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(*args, **kwargs)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.cosh(self.dice_loss(y_pred, y_true)))

class LogCoshTverskyLoss(Module):
    def __init__(self, *args, **kwargs):
        super(LogCoshTverskyLoss, self).__init__()
        self.dice_loss = TverskyLoss(*args, **kwargs)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.cosh(self.dice_loss(y_pred, y_true)))



class Sum(Module):
    def __init__(self, losses_cfg: ty.List[LossConfig], weights: ty.List[float]):
        super(Sum, self).__init__()
        if len(weights) != len(losses_cfg):
            raise ValueError("weights must be the same length as losses_cfg")
        self.losses_cfg = losses_cfg
        self.loss_fns: ty.List[Module] = []
        for cfg in losses_cfg:
            loss_cls = import_name(cfg.module, cfg.name)
            loss_fn = loss_cls(**cfg.args)
            self.loss_fns.append(loss_fn)
        self.weights = weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_fn, weight in zip(self.loss_fns, self.weights):
            loss += weight * loss_fn(y_pred, y_true)
        return loss
