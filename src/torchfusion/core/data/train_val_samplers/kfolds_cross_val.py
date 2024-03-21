"""
Defines the KFold cross validation sampling strategy.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset

from torchfusion.core.data.train_val_samplers.base import TrainValSampler


@dataclass
class KFoldCrossValSampler(TrainValSampler):
    """
    KFold cross validation sampling strategy.
    """

    k_folds: int = field(
        default=5,
        metadata={"help": ("The number of K-folds to use if using kfold cross validation data sampling strategy")},
    )

    def __post_init__(self) -> None:
        from sklearn.model_selection import KFold

        self._k_fold_sampler = KFold(n_splits=self.k_folds, shuffle=True)

    def __call__(self, train_dataset: Dataset) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes the training dataset as input and returns the next k-fold split
        train / validation sets on every call
        """

        for _, (train_ids, val_ids) in enumerate(self._k_fold_sampler.split(train_dataset)):
            yield train_dataset[train_ids], train_dataset[val_ids]
