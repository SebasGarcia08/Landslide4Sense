from ..data.dataset import LabeledDatasetIterable
from .base_trainer import Trainer
from ..utils.tools import eval_image

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from dataclasses import dataclass
import typing as ty

epsilon = 1e-14


@dataclass
class ModelTrainer(Trainer):
    input_size: ty.Tuple[int, int] = (128, 128)
    num_classes: int = 2

    def __post_init__(self):
        self.interp = nn.Upsample(
            size=(self.input_size[1], self.input_size[0]), mode="bilinear"
        )

    def train_step(
        self,
        batch_id: int,
        batch: LabeledDatasetIterable,
        batch_logs: ty.Dict[str, ty.Any],
    ) -> None:
        self.model.train()
        self.model = self.model.to(self.device)
        self.optimizer.zero_grad()

        images, labels, _, _ = batch
        images = images.to(self.device)
        pred = self.model(images)

        pred_interp = self.interp(pred)

        labels = labels.to(self.device).long()
        loss = self.loss_fn(pred_interp, labels)
        _, predict_labels = torch.max(pred_interp, 1)
        predict_labels = predict_labels.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        batch_oa = np.sum(predict_labels == labels) * 1.0 / len(labels.reshape(-1))

        batch_logs["overall_accuracy"] = batch_oa
        batch_logs["loss"] = loss.item()

        loss.backward()
        self.optimizer.step()

    def eval(
        self,
        eval_name: str,
        eval_set: LabeledDatasetIterable,
        batch_logs: ty.Dict[str, ty.Any],
    ) -> None:
        self.model.eval()
        TP_all = np.zeros((self.num_classes, 1))
        FP_all = np.zeros((self.num_classes, 1))
        TN_all = np.zeros((self.num_classes, 1))
        FN_all = np.zeros((self.num_classes, 1))
        n_valid_sample_all = 0
        F1 = np.zeros((self.num_classes, 1))

        for _, batch in tqdm(enumerate(eval_set), desc="Evaluating...", total=len(eval_set)):
            image, label, _, name = batch
            label = label.squeeze().numpy()
            image = image.float().to(self.device)

            with torch.no_grad():
                pred = self.model(image)

            _, pred = torch.max(self.interp(F.softmax(pred, dim=1)).detach(), 1)
            pred = pred.squeeze().data.cpu().numpy()

            TP, FP, TN, FN, n_valid_sample = eval_image(
                pred.reshape(-1), label.reshape(-1), self.num_classes
            )
            TP_all += TP
            FP_all += FP
            TN_all += TN
            FN_all += FN
            n_valid_sample_all += n_valid_sample

        OA = np.sum(TP_all) * 1.0 / n_valid_sample_all
        for i in range(self.num_classes):
            P = TP_all[i] * 1.0 / (TP_all[i] + FP_all[i] + epsilon)
            R = TP_all[i] * 1.0 / (TP_all[i] + FN_all[i] + epsilon)
            F1[i] = 2.0 * P * R / (P + R + epsilon)
            if i == 1:
                batch_logs[f"{eval_name}_precision"] = P * 100
                batch_logs[f"{eval_name}_recall"] = P * 100
                batch_logs[f"{eval_name}_f1"] = F1[i] * 100

        mF1 = np.mean(F1)
        batch_logs[f"{eval_name}_mean_f1"] = mF1 * 100
        batch_logs[f"{eval_name}_overall_accuracy"] = OA * 100
