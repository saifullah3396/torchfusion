"""
Defines the base DataAugmentation class for defining any kind of data augmentation.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from torchfusion.core.utilities.dataclasses.abstract_dataclass import AbstractDataclass


@dataclass
class DataAugmentation(AbstractDataclass):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        return self._aug(*args, **kwargs)
