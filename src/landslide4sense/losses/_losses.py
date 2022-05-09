import torch
from torch.nn import Module
from segmentation_models_pytorch.losses import DiceLoss


class LogCoshDiceLoss(Module):
    def __init__(self, *args, **kwargs):
        self.dice_loss = DiceLoss(*args, **kwargs)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.cosh(self.dice_loss(y_pred, y_true)))
