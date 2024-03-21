from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import ocrodeg
import scipy.ndimage as ndi
from PIL import ImageOps

from torchfusion.core.data.data_augmentations.base import DataAugmentation

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass
class RandomDistortion(DataAugmentation):
    """
    Applies random distortion to a numpy image.
    """

    sigma: float
    maxdelta: float

    def __call__(self, image: ArrayLike):
        noise = ocrodeg.bounded_gaussian_noise(image.shape, self.sigma, self.maxdelta)
        return ocrodeg.distort_with_noise(image, noise)


@dataclass
class RandomBlotches(DataAugmentation):
    """
    Applies random blobs to a numpy image.
    """

    fgblobs: float
    bgblobs = float
    fgscale: float = 10
    bgscale: float = 10

    def __call__(self, image: ArrayLike):
        return ocrodeg.random_blotches(
            image,
            fgblobs=self.fgblobs,
            bgblobs=self.bgblobs,
            fgscale=self.fgscale,
            bgscale=self.bgscale,
        )


@dataclass
class SurfaceDistortion(DataAugmentation):
    """
    Applies surface distortion to a numpy image.
    """

    magnitude: float

    def __call__(self, image: ArrayLike):
        noise = ocrodeg.noise_distort1d(image.shape, magnitude=self.magnitude)
        return ocrodeg.distort_with_noise(image, noise)


@dataclass
class Threshold(DataAugmentation):
    """
    Applies threshold distortion on a numpy image.
    """

    magnitude: float

    def __call__(self, image: ArrayLike):
        blurred = ndi.gaussian_filter(image, self.magnitude)
        return 1.0 * (blurred > 0.5)


@dataclass
class Pixelate(DataAugmentation):
    """
    Applies pixelation to a numpy image.
    """

    magnitude: float

    def __call__(self, image: ArrayLike):
        h, w = image.shape
        image = cv2.resize(
            image,
            (int(w * self.magnitude), int(h * self.magnitude)),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)


@dataclass
class JPEGCompression(DataAugmentation):
    """
    Applies jpeg compression to a numpy image.
    """

    quality: float

    def __call__(self, image: ArrayLike):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encimg = cv2.imencode(".jpg", image * 255, encode_param)
        decimg = cv2.imdecode(encimg, 0) / 255.0
        return decimg


@dataclass
class Solarization(DataAugmentation):
    """
    Applies solarization to a numpy image.
    """

    def __call__(self, image: ArrayLike):
        return ImageOps.solarize(image)
