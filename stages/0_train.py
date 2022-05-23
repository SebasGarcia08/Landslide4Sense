import json
import os
import typing as ty
import logging

import torch
import torch.backends.cudnn as cudnn

from landslide4sense.config import Config
from landslide4sense.training import ModelTrainer
from landslide4sense.utils import set_deterministic
from landslide4sense.experiments import (
    setup_model,
    setup_loss_fn,
    setup_callbacks,
    setup_datasets,
    setup_optimizer,
)

import hydra
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

name_classes = ["Non-Landslide", "Landslide"]
device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="../", config_name="params")
def main(cfg: Config):
    set_deterministic(cfg.train.seed)

    print(f"Using device: {device}")
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.train.gpu_id)
        cudnn.enabled = True
        cudnn.benchmark = True

    model = setup_model(cfg)
    optimizer = setup_optimizer(cfg, model)
    loss_fn = setup_loss_fn(cfg)
    train_loader, eval_sets = setup_datasets(cfg)
    (
        early_stopper,
        model_checkpointer,
        progress_printer,
        wandb_callback,
    ) = setup_callbacks(cfg)

    trainer = ModelTrainer(
        model,
        optimizer,
        loss_fn,
        train_set=train_loader,
        eval_sets=eval_sets,
        eval_names=cfg.data.eval_names,
        device=device,
    )

    trainer.train(
        max_epochs=cfg.train.num_steps_stop // cfg.train.steps_per_epoch,
        steps_per_epoch=cfg.train.steps_per_epoch,
        callbacks=[
            early_stopper,
            model_checkpointer,
            progress_printer,
            wandb_callback,
        ],
        start_epoch=cfg.train.start_epoch,
    )

    results: ty.Dict[str, ty.Any] = dict()
    results["model_name"] = wandb_callback.run.name
    results["wandb_id"] = wandb_callback.run.id
    results["best_epoch"] = early_stopper.best_epoch
    results["best_train_result"] = early_stopper.best_result
    results["metric"] = early_stopper.monitor

    save_path = os.path.join(
        cfg.train.snapshot_dir,
        cfg.train.callbacks.wandb.name,
        cfg.train.results_filename,
    )

    with open(save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
