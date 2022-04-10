import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn

from landslide4sense.data import LandslideDataSet
from landslide4sense.training import ModelTrainer
from landslide4sense.training.callbacks import (
    EarlyStopping,
    ModelCheckpointer,
    ProgressPrinter,
    WandbCallback,
)

name_classes = ["Non-Landslide", "Landslide"]


def import_name(modulename: str, name: str):
    """Import a named object from a module in the context of this function."""
    module = __import__(modulename, globals(), locals(), [name])
    return vars(module)[name]


def get_arguments():

    parser = argparse.ArgumentParser(description="Baseline method for Land4Seen")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/Land4Sense_Competition_h5/",
        help="dataset path.",
    )
    parser.add_argument(
        "--model_module",
        type=str,
        default="landslide4sense.models",
        help="model module to import",
    )

    parser.add_argument(
        "--model_name", type=str, default="Unet", help="modle name in given module"
    )

    parser.add_argument(
        "--train_list",
        type=str,
        default="./data/train.txt",
        help="training list file.",
    )
    parser.add_argument(
        "--test_list", type=str, default="./data/train.txt", help="test list file."
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default="128,128",
        help="width and height of input images.",
    )
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes.")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="number of images in each batch."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for multithread dataloading.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate."
    )
    parser.add_argument(
        "--num_steps", type=int, default=5000, help="number of training steps."
    )
    parser.add_argument(
        "--num_steps_stop",
        type=int,
        default=5000,
        help="number of training steps for early stopping.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="regularisation parameter for L2-loss.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id in the training.")
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default="./exp/",
        help="where to save snapshots of the model.",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        cudnn.enabled = True
        cudnn.benchmark = True

    snapshot_dir = args.snapshot_dir
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size.split(","))
    input_size = (w, h)

    # Create network
    model_cls = import_name(args.model_module, args.model_name)
    model = model_cls(n_classes=args.num_classes)

    train_loader = data.DataLoader(
        LandslideDataSet(
            args.data_dir,
            args.train_list,
            max_iters=args.num_steps_stop * args.batch_size,
            set="labeled",
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = data.DataLoader(
        LandslideDataSet(args.data_dir, args.train_list, set="labeled"),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)

    wandb_callback = WandbCallback(
        {"config": args, "project": "landslide4sense", "name": "baseline"}
    )

    callbacks = [
        ModelCheckpointer(os.path.join(snapshot_dir, wandb_callback.run.name), "f1"),
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
        num_classes=args.num_classes,
        device=device,
    )

    trainer.train(
        max_epochs=args.num_steps_stop // 500, steps_per_epoch=500, callbacks=callbacks
    )


if __name__ == "__main__":
    main()
