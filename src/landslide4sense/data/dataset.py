import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

import typing as ty

from .transformations import Transformation

LabeledDatasetIterable = ty.Tuple[np.ndarray, np.ndarray, np.ndarray, str]
UnlabeledDatasetIterable = ty.Tuple[np.ndarray, np.ndarray, str]


class LandslideDataSet(Dataset):
    def __init__(
        self,
        data_dir: str,
        list_path: str,
        max_iters: ty.Optional[int] = None,
        set: str = "label",
        transform: ty.Optional[Transformation] = None,
    ):
        self.transform = transform
        self.list_path = list_path
        self.mean = [
            -0.4914,
            -0.3074,
            -0.1277,
            -0.0625,
            0.0439,
            0.0803,
            0.0644,
            0.0802,
            0.3000,
            0.4082,
            0.0823,
            0.0516,
            0.3338,
            0.7819,
        ]
        self.std = [
            0.9325,
            0.8775,
            0.8860,
            0.8869,
            0.8857,
            0.8418,
            0.8354,
            0.8491,
            0.9061,
            1.6072,
            0.8848,
            0.9232,
            0.9018,
            1.2913,
        ]
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if max_iters is not None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = (
                self.img_ids * n_repeat
                + self.img_ids[: max_iters - n_repeat * len(self.img_ids)]
            )

        self.files: ty.List[ty.Dict[str, str]] = []

        if set == "labeled":
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = data_dir + name.replace("img", "mask").replace(
                    "image", "mask"
                )
                self.files.append({"img": img_file, "label": label_file, "name": name})
        elif set == "unlabeled":
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({"img": img_file, "name": name})

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, index: int
    ) -> ty.Union[LabeledDatasetIterable, UnlabeledDatasetIterable]:
        datafiles = self.files[index]

        if self.set == "labeled":
            with h5py.File(datafiles["img"], "r") as hf:
                image = hf["img"][:]
            with h5py.File(datafiles["label"], "r") as hf:
                label = hf["mask"][:]
            name = datafiles["name"]

            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i, :, :] -= self.mean[i]
                image[i, :, :] /= self.std[i]

            if self.transform:
                image, label = self.transform(image, label)

            return image.copy(), label.copy(), np.array(size), name

        else:
            with h5py.File(datafiles["img"], "r") as hf:
                image = hf["img"][:]
            name = datafiles["name"]

            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i, :, :] -= self.mean[i]
                image[i, :, :] /= self.std[i]

            return image.copy(), np.array(size), name


if __name__ == "__main__":

    train_dataset = LandslideDataSet(
        data_dir="/scratch/Land4Sense_Competition/", list_path="./train.txt"
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True
    )

    channels_sum, channel_squared_sum = 0, 0
    num_batches = len(train_loader)
    for data, _, _, _ in train_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(data**2, dim=[0, 2, 3])

    mean = channels_sum / num_batches
    std = (channel_squared_sum / num_batches - mean**2) ** 0.5
    print(mean, std)
    # [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
    # [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
