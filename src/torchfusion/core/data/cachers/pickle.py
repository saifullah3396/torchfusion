"""
Defines the MNIST dataset.
"""

import pickle
from pathlib import Path

import pandas as pd
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


class PickleDataCacher:
    def __init__(
        self,
        dataset_name: str,
        dataset_cache_dir: str,
        cache_file_name: str,
        split: str,
        overwrite: bool = False,
    ):
        self._dataset_name = dataset_name
        self._dataset_cache_dir = dataset_cache_dir
        self._cache_file_name = cache_file_name
        self._split = split
        self._overwrite = overwrite

    @property
    def cache_file_path(self):
        return (
            Path(self._dataset_cache_dir)
            / self._dataset_name
            / self._split
            / self._cache_file_name
        )

    def save_to_cache(self, data):
        logger.info(f"Saving dataset to cache file {[str(self.cache_file_path)]}...")
        # make target directory if not available
        if not self.cache_file_path.parent.exists():
            self.cache_file_path.parent.mkdir(parents=True)

        with open(self.cache_file_path, "wb") as f:
            pickle.dump(data, f)

        return data

    def load_from_cache(self):
        logger.info(f"Loading dataset from cache file {[str(self.cache_file_path)]}...")
        if self.cache_file_path.exists():
            return pd.read_pickle(self.cache_file_path)

    def is_cached(self):
        return self.cache_file_path and self.cache_file_path.exists()
