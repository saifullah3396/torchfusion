import os
import pickle
from pathlib import Path

import datasets
import numpy as np
import PIL

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_image_dataset import (
    FusionImageDataset,
    FusionImageDatasetConfig,
)

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
"""

_DESCRIPTION = """\
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images
per class. There are 50000 training images and 10000 test images.
"""

_HOMEPAGE = "https://www.cs.toronto.edu/~kriz/cifar.html"

_DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class Cifar10Config(FusionImageDatasetConfig):
    """BuilderConfig for Cifar10Config"""

    def __init__(self, *args, **kwargs):
        """BuilderConfig for Cifar10Config.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Cifar10Config, self).__init__(*args, **kwargs)


class Cifar10(FusionImageDataset):
    BUILDER_CONFIGS = [
        Cifar10Config(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default dataset config",
            citation=_CITATION,
            homepage=_HOMEPAGE,
            data_url=_DATA_URL,
            labels=_NAMES,
        ),
    ]

    def _dataset_features(self):
        return datasets.Features(
            {
                DataKeys.IMAGE: datasets.features.Image(decode=True),
                DataKeys.LABEL: datasets.features.ClassLabel(names=self.config.labels),
            }
        )

    def _generate_examples_impl(self, data_dir, split):
        """This function returns the examples in the raw (text) form."""

        if split == "train":
            batches = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]

        if split == "test":
            batches = ["test_batch"]
        for path in os.listdir(Path(data_dir) / "cifar-10-batches-py"):
            if path in batches:
                with open(Path(data_dir) / "cifar-10-batches-py" / path, "rb") as f:
                    dict = pickle.load(f, encoding="bytes")

                    labels = dict[b"labels"]
                    images = dict[b"data"]

                    for idx, _ in enumerate(images):
                        img_reshaped = np.transpose(
                            np.reshape(images[idx], (3, 32, 32)), (1, 2, 0)
                        )
                        yield f"{path}_{idx}", {
                            DataKeys.IMAGE: img_reshaped,
                            DataKeys.LABEL: labels[idx],
                        }
