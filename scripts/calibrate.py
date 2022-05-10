import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.backends.cudnn as cudnn
import h5py

from landslide4sense.data import LandslideDataSet
from landslide4sense.utils import set_deterministic, import_name
from landslide4sense.config import Config

import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
logger = logging.getLogger(__name__)

name_classes = ["Non-Landslide", "Landslide"]


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: Config):
    set_deterministic(cfg.train.seed)
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
        cfg.model.restore_from, map_location=torch.device(device)
    )
    model.load_state_dict(saved_state_dict)

    if device == "cuda":
        model = model.cuda()
    else:
        logger.warning("Could not load CUDA, trying to do inference with cpu...")

    test_loader = data.DataLoader(
        LandslideDataSet(cfg.data.dir, cfg.data.test_list, set="unlabeled"),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode="bilinear")

    print("Testing..........")
    model.eval()

    for index, batch in enumerate(test_loader):
        image, _, name = batch
        image = image.float()
        if device == "cuda":
            image = image.cuda()
        name = name[0].split(".")[0].split("/")[-1].replace("image", "mask")
        print(index + 1, "/", len(test_loader), ": Testing ", name)

        with torch.no_grad():
            pred = model(image)
        predicted_mask = interp(F.softmax(pred, dim=1)).detach()
        _, pred = torch.max(predicted_mask, 1)
        pred = pred.squeeze().data.cpu().numpy().astype("uint8")
        with h5py.File(snapshot_dir + name + ".h5", "w") as hf:
            hf.create_dataset("mask", data=pred)


if __name__ == "__main__":
    main()
