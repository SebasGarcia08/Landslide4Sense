import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn

from landslide4sense.config import Config
from landslide4sense.data import LandslideDataSet
from landslide4sense.training import ModelTrainer
from landslide4sense.utils import import_name, set_deterministic
from landslide4sense.training.callbacks import (
    EarlyStopping,
    ModelCheckpointer,
    ProgressPrinter,
    WandbCallback,
)

import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

name_classes = ["Non-Landslide", "Landslide"]


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: Config):
    print(cfg)
    return
    train_cfg = cfg.training
    model_cfg = cfg.model
    data_cfg = cfg.data

    set_deterministic(train_cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(train_cfg.gpu_id)
        cudnn.enabled = True
        cudnn.benchmark = True

    snapshot_dir = train_cfg.snapshot_dir
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    w, h = map(int, model_cfg.input_size.split(","))
    input_size = (w, h)

    # Create network
    model_cls = import_name(model_cfg.module, model_cfg.name)
    model = model_cls(n_classes=model_cfg.num_classes)

    train_loader = data.DataLoader(
        LandslideDataSet(
            data_cfg.dir,
            data_cfg.train_list,
            max_iters=train_cfg.num_steps_stop * train_cfg.batch_size,
            set="labeled",
        ),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    test_loader = data.DataLoader(
        LandslideDataSet(data_cfg.dir, data_cfg.train_list, set="labeled"),
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)

    wandb_callback = WandbCallback(
        {"config": cfg, "project": "landslide4sense", "name": "baseline"}
    )

    callbacks = [
        ModelCheckpointer(os.path.join(snapshot_dir, wandb_callback.run.name), "train"),
        ProgressPrinter(),
        wandb_callback,
        EarlyStopping(monitor="train_f1", mode="max", patience=3, best_result=0.5),
    ]

    trainer = ModelTrainer(
        model,
        optimizer,
        cross_entropy_loss,
        train_set=train_loader,
        eval_sets=[test_loader],
        eval_names=["train"],
        input_size=input_size,
        num_classes=model_cfg.num_classes,
        device=device,
    )

    trainer.train(
        max_epochs=train_cfg.num_steps_stop // 500,
        steps_per_epoch=500,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
