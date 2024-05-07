"""
Defines the MNIST dataset.
"""

import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable

import pandas as pd
from datadings.reader import MsgpackReader
from datadings.writer import FileWriter
from torchfusion.core.constants import DataKeys
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


class DatadingsDataCacher:
    def __init__(
        self,
        dataset_name: str,
        dataset_cache_dir: str,
        cache_file_name: str,
        split: str,
        workers=8,
        overwrite: bool = False,
    ):
        self._dataset_name = dataset_name
        self._dataset_cache_dir = dataset_cache_dir
        self._cache_file_name = cache_file_name
        self._split = split
        self._overwrite = overwrite
        self._workers = workers

    @property
    def cache_file_path(self):
        return (
            Path(self._dataset_cache_dir)
            / self._dataset_name
            / self._split
            / f"{self._cache_file_name}.msgpack"
        )

    @property
    def cache_file_metadata(self):
        return (
            Path(self._dataset_cache_dir)
            / self._dataset_name
            / self._split
            / f"{self._cache_file_name}.metadata"
        )

    def _save_dataset_metadata(self, data: pd.DataFrame):
        import pickle

        if isinstance(data, pd.DataFrame):
            sample = data.iloc[0].to_dict()
        else:
            sample = data[0]
        dataset_meta = {
            "size": len(data),
            "keys": [DataKeys.INDEX, *list(sample.keys())],
        }
        with open(self.cache_file_metadata, "wb") as f:
            pickle.dump(dataset_meta, f)

    def sample_preprocessor(self, preprocess_fn: Callable = None):
        def wrap_fn(sample):
            idx, sample = sample
            if preprocess_fn is not None:
                sample = preprocess_fn(sample)
            return {"key": str(idx), "data": pickle.dumps(sample)}

        return wrap_fn

    def sample_generator(self, data: pd.DataFrame):
        for idx, _ in data.iterrows():
            yield idx, data.iloc[idx].to_dict()

    def _write_msgpack(self, data: pd.DataFrame, preprocess_fn: Callable = None):
        import tqdm

        try:
            # create a thread pool for parallel writing
            # Pillow has a problem loading images in threaded envrionment :/
            pool = ThreadPool(self._workers)
            with FileWriter(self.cache_file_path, overwrite=True) as writer:
                logger.info(
                    f"Writing dataset [{self._dataset_name}] to a datadings "
                    f"file {self.cache_file_path}. This might take a while..."
                )
                progress_bar = tqdm.tqdm(
                    pool.imap_unordered(
                        self.sample_preprocessor(preprocess_fn=preprocess_fn),
                        self.sample_generator(data),
                    ),
                    total=len(data),
                )
                for sample in progress_bar:
                    writer.write({**sample})

                    # note: progress bar might be slightly off with
                    # multiple processes
                    progress_bar.update(1)

            # # create writier instnace
            # with FileWriter(self.cache_file_path, overwrite=True) as writer:
            #     logger.info(
            #         f"Writing  dataset [{self._dataset_name}] to a datadings "
            #         f"file {self.cache_file_path}. This might take a while..."
            #     )

            #     fn = self.sample_preprocessor(preprocess_fn)
            #     for sample in tqdm.tqdm(self.sample_generator(data)):
            #         writer.write(fn(sample))

        except KeyboardInterrupt as exc:
            logger.exception(f"Data caching interrupted. Exiting...")
            exit(1)
        except Exception as exc:
            logger.exception(f"Exception raise while writing data: {exc}")
            exit(1)

    def save_to_cache(self, data: pd.DataFrame, preprocess_fn: Callable = None):
        logger.info(f"Saving dataset to cache file {[str(self.cache_file_path)]}...")
        # make target directory if not available
        if not self.cache_file_path.parent.exists():
            self.cache_file_path.parent.mkdir(parents=True)

        # save dataset meta info
        self._save_dataset_metadata(data)

        # save to msgpack
        self._write_msgpack(data, preprocess_fn=preprocess_fn)

        # load the cached file
        return MsgpackReader(self.cache_file_path)

    def load_from_cache(self):
        logger.info(f"Loading dataset from cache file {[str(self.cache_file_path)]}...")
        if self.cache_file_path.exists():
            return MsgpackReader(self.cache_file_path)

    def is_cached(self):
        import pickle

        # make sure cache files exist
        if not (self.cache_file_metadata.exists() and self.cache_file_path.exists()):
            return False

        # make sure the cached data and its metadata align
        data_reader = MsgpackReader(self.cache_file_path)
        with open(self.cache_file_metadata, "rb") as f:
            metadata = pickle.load(f)

        if len(data_reader) != metadata["size"]:
            return False
        return True
