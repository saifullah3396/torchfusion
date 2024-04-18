"""
Defines the base TrainValSampler class for defining training/validation split samplers.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset

from torchfusion.core.utilities.dataclasses.abstract_dataclass import AbstractDataclass


@dataclass
class TrainValSampler(AbstractDataclass):
    @abstractmethod
    def __call__(self, dataset: Dataset):
        pass
