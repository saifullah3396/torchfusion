"""
Defines the random split sampling strategy.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader

from torchfusion.core.data.train_val_samplers.base import TrainValSampler


@dataclass
class RandomSplitSampler(TrainValSampler):
    """
    Random split sampling strategy.
    """

    # The default seed used for random sampling
    seed: int = 42

    # The train/validation dataset split ratio
    random_split_ratio: float = 0.8

    def __call__(self, train_dataset: typing.Union[HFDataset, TorchDataset]) -> typing.Tuple[DataLoader, DataLoader]:
        """
        Takes the training dataset as input and returns split train / validation
        sets based on the split ratio.
        """
        if isinstance(train_dataset, HFDataset):
            output = train_dataset.train_test_split(
                test_size=round(1.0 - self.random_split_ratio, 2),
                shuffle=True,
                seed=self.seed,
                load_from_cache_file=False,
            )
            return output["train"], output["test"]
        elif isinstance(train_dataset, TorchDataset):
            import torch
            from torch.utils.data.dataset import random_split

            train_dataset_size = len(train_dataset)
            val_dataset_size = int(train_dataset_size * round(1.0 - self.random_split_ratio, 2))
            train_set, val_set = random_split(
                train_dataset,
                [train_dataset_size - val_dataset_size, val_dataset_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
            train_set.info = train_dataset.info
            val_set.info = train_dataset.info
            return train_set, val_set
        else:
            raise ValueError(f"Dataset of type {type(train_dataset)} is not supported.")
