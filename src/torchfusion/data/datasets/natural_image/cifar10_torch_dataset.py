from __future__ import annotations

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from torchfusion.core.constants import DataKeys
from torchfusion.core.utilities.logging import get_logger

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


class Cifar10TorchDataset(Dataset):
    def __init__(
        self,
        data_dir,
        cache_dir,
        config_name: str = "default",
        split: str = "train",
    ):
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._config_name = config_name
        self._split = split
        self._logger = get_logger()

        if split == "train":
            self._data = CIFAR10(self._data_dir, train=True, download=True)
        else:
            self._data = CIFAR10(self._data_dir, train=False, download=True)

    @property
    def info(self):
        return {"splits": ["train", "test"], "features": {DataKeys.LABEL: _NAMES}}

    def __getitem__(self, index):
        output = self._data[index]
        return {DataKeys.IMAGE: output[0], DataKeys.LABEL: output[1]}

    def __len__(self):
        return len(self._data)
