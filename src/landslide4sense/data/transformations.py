from abc import abstractmethod
from dataclasses import dataclass
import typing as ty
import numpy as np
from albumentations.core.composition import Compose


@dataclass
class BaseAugmentation:
    transform: Compose

    @abstractmethod
    def __call__(
        self, img: np.ndarray, mask: np.ndarray
    ) -> ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


@dataclass
class Transformation(BaseAugmentation):
    def __call__(
        self, img: np.ndarray, mask: np.ndarray
    ) -> ty.Tuple[np.ndarray, np.ndarray]:
        # C x H x W -> H x W x C
        transposed = img.transpose(1, 2, 0)
        augmented = self.transform(image=transposed, mask=mask)
        # Transpose back H x W x C -> C x H x W
        augmented_img = augmented["image"].transpose(2, 0, 1)
        augmented_mask = augmented["mask"]
        return augmented_img, augmented_mask
