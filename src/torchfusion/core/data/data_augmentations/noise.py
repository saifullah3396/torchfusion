from __future__ import annotations

import importlib

import cv2
import numpy as np
import ocrodeg
from numpy.typing import ArrayLike

from torchfusion.core.data.data_augmentations.base import DataAugmentation

ocrodeg = importlib.find_loader("ocrodeg")


class GaussianNoiseRGB(DataAugmentation):
    """
    Applies RGB Gaussian noise to a numpy image.
    """

    def __init__(self, magnitude: float):
        super().__init__()

        self.magnitude = magnitude

    def __call__(self, image: ArrayLike):
        # if len(image.shape) == 2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return np.clip(image + np.random.normal(size=image.shape, scale=self.magnitude), 0, 1)


class ShotNoiseRGB(DataAugmentation):
    """
    Applies shot noise to a numpy image.
    """

    def __init__(self, magnitude: float):
        super().__init__()

        self.magnitude = magnitude

    def __call__(self, image: ArrayLike):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return np.clip(np.random.poisson(image * self.magnitude) / float(self.magnitude), 0, 1)


class FibrousNoise(DataAugmentation):
    """
    Applies fibrous noise to a numpy image.
    """

    def __init__(self, blur: float = 1.0, blotches: float = 5e-5):
        super().__init__()

        self.blur = blur
        self.blotches = blotches

        if ocrodeg is None:
            raise ImportError(
                "ocrodeg is not installed. Please install it using pip install ocrodeg to use FibrousNoise augmentation."
            )

    def __call__(self, image: ArrayLike):
        return ocrodeg.printlike_fibrous(image, blur=self.blur, blotches=self.blotches)


class MultiscaleNoise(DataAugmentation):
    """
    Applies multiscale noise to a numpy image.
    """

    def __init__(self, blur: float = 1.0, blotches: float = 5e-5):
        super().__init__()

        self.blur = blur
        self.blotches = blotches

        if ocrodeg is None:
            raise ImportError(
                "ocrodeg is not installed. Please install it using pip install ocrodeg to use MultiscaleNoise augmentation."
            )

    def __call__(self, image: ArrayLike):
        return ocrodeg.printlike_multiscale(image, blur=self.blur, blotches=self.blotches)
