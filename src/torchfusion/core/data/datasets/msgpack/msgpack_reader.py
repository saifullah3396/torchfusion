import copy
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from datadings.reader import MsgpackReader as MsgpackFileReader
from datasets import DownloadConfig
from datasets.arrow_reader import (
    HF_GCP_BASE_URL,
    DatasetNotOnHfGcsError,
    MissingFilesOnHfGcsError,
    ReadInstruction,
    make_file_instructions,
)
from datasets.info import DatasetInfo
from datasets.splits import Split
from datasets.table import Table
from datasets.utils import logging
from datasets.utils.file_utils import cached_path

if TYPE_CHECKING:
    from datasets.info import DatasetInfo  # noqa: F401
    from datasets.splits import Split  # noqa: F401

logger = logging.get_logger(__name__)


class MsgpackReader:
    """
    A msgpack reader class compatible with huggingface datasets.
    """

    def __init__(self, path: str, info: Optional["DatasetInfo"]):
        """Initializes MsgpackReader.

        Args:
            path (str): path where tfrecords are stored.
            info (DatasetInfo): info about the dataset.
        """
        self._path: str = path
        self._info: Optional["DatasetInfo"] = info
        self._filetype_suffix: Optional[str] = "msgpack"

    def _read_files(self, files, in_memory=False) -> Table:
        """Returns Dataset for given file instructions.

        Args:
            files: List[dict(filename, skip, take)], the files information.
                The filenames contain the absolute path, not relative.
                skip/take indicates which example read in the file: `ds.slice(skip, take)`
            in_memory (bool, default False): Whether to copy the data in-memory.
        """
        if len(files) == 0 or not all(isinstance(f, dict) for f in files):
            raise ValueError("please provide valid file informations")
        readers = []
        files = copy.deepcopy(files)
        for f in files:
            f["filename"] = os.path.join(self._path, f["filename"])
        for f in files:
            readers.append(MsgpackFileReader(f["filename"]))
        return readers

    def get_file_instructions(self, name, instruction, split_infos):
        """Return list of dict {'filename': str, 'skip': int, 'take': int}"""
        file_instructions = make_file_instructions(
            name,
            split_infos,
            instruction,
            filetype_suffix=self._filetype_suffix,
            prefix_path=self._path,
        )
        files = file_instructions.file_instructions
        return files

    def read(
        self,
        name,
        instructions,
        split_infos,
        in_memory=False,
    ):
        """Returns Dataset instance(s).

        Args:
            name (str): name of the dataset.
            instructions (ReadInstruction): instructions to read.
                Instruction can be string and will then be passed to the Instruction
                constructor as it.
            split_infos (list of SplitInfo proto): the available splits for dataset.
            in_memory (bool, default False): Whether to copy the data in-memory.

        Returns:
             kwargs to build a single Dataset instance.
        """

        files = self.get_file_instructions(name, instructions, split_infos)
        if not files:
            msg = f'Instruction "{instructions}" corresponds to no data!'
            raise ValueError(msg)
        return self.read_files(
            files=files, original_instructions=instructions, in_memory=in_memory
        )

    def read_files(
        self,
        files: List[dict],
        original_instructions: Union[None, "ReadInstruction", "Split"] = None,
        in_memory=False,
    ):
        """Returns single Dataset instance for the set of file instructions.

        Args:
            files: List[dict(filename, skip, take)], the files information.
                The filenames contains the relative path, not absolute.
                skip/take indicates which example read in the file: `ds.skip().take()`
            original_instructions: store the original instructions used to build the dataset split in the dataset.
            in_memory (bool, default False): Whether to copy the data in-memory.

        Returns:
            kwargs to build a Dataset instance.
        """
        if in_memory:
            raise NotImplementedError("in_memory=True is not implemented yet.")

        # Prepend path to filename
        msgpack_readers = self._read_files(files, in_memory=in_memory)
        # If original_instructions is not None, convert it to a human-readable NamedSplit
        if original_instructions is not None:
            from datasets.splits import Split  # noqa

            split = Split(str(original_instructions))
        else:
            split = None
        dataset_kwargs = {
            "msgpack_readers": msgpack_readers,
            "info": self._info,
            "split": split,
        }
        return dataset_kwargs

    def download_from_hf_gcs(self, download_config: DownloadConfig, relative_data_dir):
        """
        Download the dataset files from the Hf GCS

        Args:
            dl_cache_dir: `str`, the local cache directory used to download files
            relative_data_dir: `str`, the relative directory of the remote files from
                the `datasets` directory on GCS.

        """
        remote_cache_dir = (
            HF_GCP_BASE_URL + "/" + relative_data_dir.replace(os.sep, "/")
        )
        try:
            remote_dataset_info = os.path.join(remote_cache_dir, "dataset_info.json")
            downloaded_dataset_info = cached_path(
                remote_dataset_info.replace(os.sep, "/")
            )
            shutil.move(
                downloaded_dataset_info, os.path.join(self._path, "dataset_info.json")
            )
            if self._info is not None:
                self._info.update(self._info.from_directory(self._path))
        except FileNotFoundError as err:
            raise DatasetNotOnHfGcsError(err) from None
        try:
            for split in self._info.splits:
                file_instructions = self.get_file_instructions(
                    name=self._info.builder_name,
                    instruction=split,
                    split_infos=self._info.splits.values(),
                )
                for file_instruction in file_instructions:
                    file_to_download = str(
                        Path(file_instruction["filename"]).relative_to(self._path)
                    )
                    remote_prepared_filename = os.path.join(
                        remote_cache_dir, file_to_download
                    )
                    downloaded_prepared_filename = cached_path(
                        remote_prepared_filename.replace(os.sep, "/"),
                        download_config=download_config,
                    )
                    shutil.move(
                        downloaded_prepared_filename, file_instruction["filename"]
                    )
        except FileNotFoundError as err:
            raise MissingFilesOnHfGcsError(err) from None
