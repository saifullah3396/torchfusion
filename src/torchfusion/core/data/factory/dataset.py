from __future__ import annotations

import inspect
from multiprocessing import get_logger

import datasets
from datasets import config, load_dataset, load_dataset_builder

from torchfusion.core.utilities.module_import import ModuleLazyImporter


class DatasetFactory:
    @staticmethod
    def create(
        dataset_name: str,
        **dataset_kwargs,
    ) -> datasets.DatasetBuilder:
        config.MAX_SHARD_SIZE = "64GB"  # we don't want very small msgpack shards
        dataset_class = ModuleLazyImporter.get_datasets().get(dataset_name, None)
        logger = get_logger()
        if dataset_class is not None:
            dataset_class = dataset_class()

            # if we have a local file, we load dataset from that
            dataset_module = inspect.getfile(dataset_class)
            return load_dataset(
                dataset_module,
                **dataset_kwargs,
                ignore_verifications=True,
            )
        else:  # if locally the dataset is not available, we try to find on huggingface
            logger.warning(
                f"Could not match dataset to any of the available datasets: {ModuleLazyImporter.get_datasets().keys()}."
                f"Loading the dataset [{dataset_name}] directly from huggingface."
            )
            return load_dataset(
                dataset_name,
                **dataset_kwargs,
                ignore_verifications=True,
            )

    @staticmethod
    def create_builder(
        dataset_name: str,
        **dataset_kwargs,
    ) -> datasets.DatasetBuilder:
        dataset_class = ModuleLazyImporter.get_datasets().get(dataset_name, None)
        logger = get_logger()
        if dataset_class is not None:
            dataset_class = dataset_class()

            # if we have a local file, we load dataset from that
            dataset_module = inspect.getfile(dataset_class)
            return load_dataset_builder(
                dataset_module,
                **dataset_kwargs,
            )
        else:  # if locally the dataset is not available, we try to find on huggingface
            logger.warning(
                f"Could not match dataset to any of the available datasets: {ModuleLazyImporter.get_datasets().keys()}"
            )
            return load_dataset_builder(
                dataset_name,
                **dataset_kwargs,
            )
