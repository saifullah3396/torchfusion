import copy
import pickle
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import numpy as np
import tqdm
from datadings.reader import MsgpackReader as MsgpackFileReader
from datasets.arrow_dataset import DatasetInfoMixin
from datasets.info import DatasetInfo
from datasets.splits import NamedSplit
from datasets.utils import logging
from torch.utils.data import Dataset

from torchfusion.core.constants import DataKeys

if TYPE_CHECKING:
    from datasets.info import DatasetInfo  # noqa: F401

logger = logging.get_logger(__name__)


class MsgpackBasedDataset(Dataset, DatasetInfoMixin):
    """A mix of torch dataset and huggingface dataset info backed by a msgpack file or a list of msgpack files.
    This dataset is loaded by the MsgpackBuilder
    """

    def __init__(
        self,
        msgpack_readers: Union[List[MsgpackFileReader], MsgpackFileReader],
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        fingerprint: Optional[str] = None,
        transforms: Optional[Callable] = None,
        load_data_into_ram: bool = False,
    ):
        info = info.copy() if info is not None else DatasetInfo()
        DatasetInfoMixin.__init__(self, info=info, split=split)

        self._data = msgpack_readers
        self._format_type: Optional[str] = None
        self._format_kwargs: dict = {}
        self._format_columns: Optional[list] = None
        self._fingerprint: str = fingerprint
        self._transforms = transforms
        self._data_is_loaded = False

        # map indices from multiple readers
        self.cumulative_sizes = []
        self.total_size = 0

        for data in self._data:
            self.total_size += len(data)
            self.cumulative_sizes.append(self.total_size)
            data._close()
        self.cumulative_sizes = np.array(self.cumulative_sizes)

        loaded_data = []
        if load_data_into_ram:
            for index in tqdm.tqdm(range(self.total_size)):
                loaded_data.append(self.get_sample(index))
            self._data = loaded_data
            self._data_is_loaded = True

    def get_sample(self, index):
        if len(self._data) == 1:
            return self._data[-1][index]

        # find the shard where the index falls
        shard_index = np.where((index / self.cumulative_sizes) < 1)[0][0]
        shard = self._data[shard_index]

        if shard_index > 0:
            return shard[index - self.cumulative_sizes[shard_index - 1]]
        else:
            return shard[index]

    def __getitem__(self, index):
        if self._data_is_loaded:
            sample = self._data[index]
        else:
            sample = self.get_sample(index)

        if "data" in sample:
            sample = pickle.loads(sample["data"])

        for key, value in sample.items():
            if key in self.features and hasattr(self.features[key], "decode_example"):
                # we manually stop decoding if decode is set to False cuz huggingface instead throws a runtime error
                if (
                    hasattr(self.features[key], "decode")
                    and not self.features[key].decode
                ):
                    continue
                sample[key] = self.features[key].decode_example(value)

        # apply transforms
        if self._transforms is not None:
            sample = self._transforms(sample)

        # assign sample index
        sample[DataKeys.INDEX] = index

        return sample

    def __len__(self):
        return self.total_size

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self._info.features.keys())},\n    num_rows: {self.total_size}\n}})"

    def _close(self):
        if self._data_is_loaded:
            return
        return [d._close() for d in self._data]

    def close(self):
        self._close()


class MsgpackBasedTorchDataset(Dataset):
    """A mix of torch dataset and huggingface dataset info backed by a msgpack file or a list of msgpack files.
    This dataset is loaded by the MsgpackBuilder
    """

    def __init__(
        self,
        msgpack_readers: Union[List[MsgpackFileReader], MsgpackFileReader],
        split: str,
        info: Optional[DatasetInfo] = None,
        transforms: Optional[Callable] = None,
        load_data_into_ram: bool = False,
    ):
        self._data = msgpack_readers
        self.split = split
        self.info = copy.deepcopy(info)
        self._transforms = transforms
        self._data_is_loaded = False

        # map indices from multiple readers
        self.cumulative_sizes = []
        self.total_size = 0
        for data in self._data:
            self.total_size += len(data)
            self.cumulative_sizes.append(self.total_size)
            data._close()
        self.cumulative_sizes = np.array(self.cumulative_sizes)

        loaded_data = []
        if load_data_into_ram:
            for index in tqdm.tqdm(range(self.total_size)):
                loaded_data.append(self.get_sample(index))
            self._data = loaded_data
            self._data_is_loaded = True

    def get_sample(self, index):
        if len(self._data) == 1:
            return self._data[-1][index]

        # find the shard where the index falls
        shard_index = np.where((index / self.cumulative_sizes) < 1)[0][0]
        shard = self._data[shard_index]
        if shard_index > 0:
            return shard[index - self.cumulative_sizes[shard_index - 1]]
        else:
            return shard[index]

    def __getitem__(self, index):
        if self._data_is_loaded:
            sample = self._data[index]
        else:
            sample = self.get_sample(index)

        if "data" in sample:
            sample = pickle.loads(sample["data"])

        # apply transforms
        if self._transforms is not None:
            sample = self._transforms(sample)

        # assign sample index
        sample[DataKeys.INDEX] = index

        return sample

    def __len__(self):
        return self.total_size

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self.info.features.keys())},\n    num_rows: {self.total_size}\n}})"

    def _close(self):
        if self._data_is_loaded:
            return
        return [d._close() for d in self._data]

    def close(self):
        self._close()
