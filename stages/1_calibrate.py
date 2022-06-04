import os
import json
import logging
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn

from landslide4sense.data import LandslideDataSet
from landslide4sense.utils import set_deterministic, import_name, eval_image
from landslide4sense.config import Config
from landslide4sense import EPSILON

import hydra
from hydra.core.config_store import ConfigStore

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
logger = logging.getLogger(__name__)

name_classes = ["Non-Landslide", "Landslide"]


def evaluate(y_pred: np.ndarray, y_true: np.ndarray):
    TP, FP, TN, FN, n_valid_sample = eval_image(
        y_pred.astype(np.uint8).squeeze().reshape(-1),
        y_true,
        2,
    )

    OA = np.sum(TP[1]) * 1.0 / n_valid_sample
    P = np.squeeze((TP[1] * 1.0) / (TP[1] + FP[1] + EPSILON))
    R = np.squeeze((TP[1] * 1.0) / (TP[1] + FN[1] + EPSILON))
    F1 = (2.0 * P * R) / (P + R + EPSILON)
    return OA, P, R, F1


def initialize_arrays(num: int, size: int) -> ty.List[np.ndarray]:
    arrays = []
    for _ in range(num):
        arrays.append(np.zeros((size, 1)))
    return arrays


@hydra.main(config_path="../", config_name="params")
def main(cfg: Config):
    set_deterministic(cfg.train.seed)
    save_path = os.path.join(
        cfg.train.snapshot_dir,
        cfg.train.callbacks.wandb.name,
        cfg.train.results_filename,
    )

    with open(save_path, "r") as f:
        results = json.load(f)

    logger.info("Loading training results")
    logger.info(results)
    wandb_run = wandb.init(
        resume="must", id=results["wandb_id"], project=cfg.train.callbacks.wandb.project
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.train.gpu_id)
    snapshot_dir = cfg.train.snapshot_dir
    if os.path.exists(snapshot_dir) == False:
        os.makedirs(snapshot_dir)

    w, h = cfg.data.input_size
    input_size = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Create network
    model_cls = import_name(cfg.model.module, cfg.model.name)
    model = model_cls(**cfg.model.args)

    saved_state_dict = torch.load(
        os.path.join(
            cfg.train.snapshot_dir,
            cfg.train.callbacks.wandb.name,
            cfg.calibrate.model_filename,
        ),
        map_location=torch.device(device),
    )
    model.load_state_dict(saved_state_dict)

    if device == "cuda":
        model = model.cuda()
    else:
        logger.warning("Could not load CUDA, trying to do inference with cpu...")

    test_loader = data.DataLoader(
        LandslideDataSet(cfg.data.dir, cfg.calibrate.data_path, set="labeled", max_iters=1),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode="bilinear")

    model.eval()

    model.eval()
    all_preds = []
    all_labels = []
    pbar = tqdm(
        enumerate(test_loader),
        desc=f"Predicting...",
        total=len(test_loader),
    )

    for i, batch in pbar:
        image, label, _, name = batch
        label = label.squeeze().numpy().astype(np.uint8)
        image = image.float().to(device)

        with torch.no_grad():
            pred = model(image)

        pred = interp(F.softmax(pred, dim=1)).half().detach()
        all_preds.append(pred)
        all_labels.append(label)

    preds = torch.cat(all_preds).cpu().numpy()
    labels = np.concatenate(all_labels)
    y_true = labels.astype(np.uint8).squeeze().reshape(-1)
    thrs, f1s, ps, rs, oas = initialize_arrays(5, cfg.calibrate.num_thresholds)

    for i, thr in tqdm(
        enumerate(np.linspace(0.05, 0.95, cfg.calibrate.num_thresholds)),
        total=cfg.calibrate.num_thresholds,
    ):
        y_pred = preds[:, 1, :, :] > thr
        OA, P, R, F1 = evaluate(y_pred, y_true)
        thrs[i] = thr
        oas[i] = OA
        ps[i] = P
        rs[i] = R
        f1s[i] = F1
    best_idx = f1s.argmax()
    optimal_thr = np.squeeze(thrs[best_idx])

    best_f1 = f1s.max()

    plt.plot(thrs, f1s, label=f"F1: {np.round(best_f1, 3)}")
    plt.plot(thrs, ps, label=f"Precision: {np.round(ps[best_idx], 3)}")
    plt.plot(thrs, rs, label=f"Recall {np.round(rs[best_idx], 3)}")
    plt.axvline(optimal_thr, label=f"Thr: {np.round(optimal_thr, 2)}")
    plt.legend()
    plt.show()

    plt.hist(preds.squeeze().reshape(-1), bins=100)
    plt.show()

    results["optimal_thr"] = float(optimal_thr)
    results["best_precision"] = float(ps[best_idx])
    results["best_recall"] = float(rs[best_idx])
    results["best_f1"] = float(best_f1)
    wandb.log(results)

    save_path = os.path.join(
        cfg.train.snapshot_dir,
        cfg.train.callbacks.wandb.name,
        cfg.calibrate.results_filename,
    )

    with open(save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
