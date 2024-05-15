from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import numpy as np
import PIL
import torch
from torchfusion.core.data.data_augmentations.base import DataAugmentation
from torchvision import transforms

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass
class DictTransform(DataAugmentation):
    """
    Applies the transformation on given keys for dictionary outputs

    Args:
        keys (str): data key to apply this augmentation to
        transform (callable): Transformation to be applied
    """

    key: Union[str, List[str]]
    transform: typing.Callable

    def __call__(self, sample):
        if isinstance(self.key, list):
            for key in self.key:
                if key in sample:
                    sample[key] = self.transform(sample[key])
        else:
            if self.key in sample:
                sample[self.key] = self.transform(sample[self.key])

        return sample

    def __repr__(self) -> str:
        return f"DictTransform(key={self.key}, transform={self.transform})"


@dataclass
class ToTensor(DataAugmentation):
    """
    Converts input to tensor
    """

    def _initialize_aug(self):
        # generate transformations list
        aug = []

        # convert images to tensor
        aug.append(transforms.ToTensor())

        # change dtype to float
        aug.append(transforms.ConvertImageDtype(torch.float))

        # generate torch transformation
        return transforms.Compose(aug)

    def __post_init__(self):
        self._aug = self._initialize_aug()

    def __call__(self, sample):
        if isinstance(sample, list):
            return [self._aug(s) for s in sample]
        else:
            return self._aug(sample)


@dataclass
class Brightness(DataAugmentation):
    """
    Increases/decreases brightness of a numpy image based on the beta parameter.
    """

    beta: float = 0.5

    def __call__(self, image: ArrayLike):
        return np.clip(image + self.beta, 0, 1)


@dataclass
class Contrast(DataAugmentation):
    """
    Increases/decreases contrast of a numpy image based on the alpha parameter.
    """

    alpha: float = 0.5

    def __call__(self, image: ArrayLike):
        channel_means = np.mean(image, axis=(0, 1))
        return np.clip((image - channel_means) * self.alpha + channel_means, 0, 1)


@dataclass
class GrayScaleToRGB(DataAugmentation):
    """
    Converts a gray-scale torch image to rgb image.
    """

    def __call__(self, image: torch.Tensor):
        if len(image.shape) == 2 or image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        else:
            return image


@dataclass
class RGBToGrayScale(DataAugmentation):
    """
    Converts a rgb image to grayscale.
    """

    def __call__(self, image: PIL.Image):
        return image.convert("L")


@dataclass
class RGBToBGR(DataAugmentation):
    """
    Converts a torch tensor from RGB to BGR
    """

    def __call__(self, image: torch.Tensor):
        return image.permute(2, 1, 0)
