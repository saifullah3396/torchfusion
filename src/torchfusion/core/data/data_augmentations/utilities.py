from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.ndimage import zoom as scizoom

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def disk(radius, alias_blur=0.1, dtype=np.float32):
    """
    Creats the aliased kernel disk over image.
    """

    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def clipped_zoom(self, image: ArrayLike, zoom_factor: float):
    """
    Applies clipped zoom over image.
    """

    h = image.shape[0]
    w = image.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(
        image[top : top + ch, left : left + cw],
        (self.zoom_factor, self.zoom_factor),
        order=1,
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top : trim_top + h, trim_left : trim_left + w]
