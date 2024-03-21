import contextlib
import os
import posixpath
import shutil
import time
import warnings
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Union

import datasets
import fsspec
from datadings.index import (
    SUFFIX_FILTER,
    SUFFIX_KEY_HASHES,
    SUFFIX_KEYS,
    SUFFIX_OFFSETS,
)
from datasets import DownloadConfig, DownloadManager, DownloadMode, config
from datasets.arrow_reader import (
    DatasetNotOnHfGcsError,
    MissingFilesOnHfGcsError,
    ReadInstruction,
)
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from datasets.download.mock_download_manager import MockDownloadManager
from datasets.filesystems import is_remote_filesystem
from datasets.splits import Split, SplitGenerator, SplitInfo
from datasets.utils import logging
from datasets.utils.filelock import FileLock
from datasets.utils.info_utils import VerificationMode
from datasets.utils.py_utils import (
    convert_file_size_to_int,
    has_sufficient_disk_space,
    iflatmap_unordered,
    size_str,
    temporary_assignment,
)
from datasets.utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from multiprocess import Pool
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import thread_map

from torchfusion.core.data.datasets.download_manager import FusionDownloadManager
from torchfusion.core.data.datasets.msgpack.dataset import MsgpackBasedDataset
from torchfusion.core.data.datasets.msgpack.msgpack_reader import MsgpackReader
from torchfusion.core.data.datasets.msgpack.msgpack_writer import MsgpackWriter

if TYPE_CHECKING:
    from datasets.splits import Split, SplitInfo  # noqa: F401

logger = logging.get_logger(__name__)


class MsgpackBasedBuilder(datasets.GeneratorBasedBuilder):
    """Base class for datasets with data generation based on Msgpack loading functions."""

    def download_and_prepare(
        self,
        output_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        ignore_verifications="deprecated",
        try_from_hf_gcs: bool = True,
        dl_manager: Optional[DownloadManager] = None,
        base_path: Optional[str] = None,
        use_auth_token="deprecated",
        file_format: str = "msgpack",
        max_shard_size: Optional[Union[int, str]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **download_and_prepare_kwargs,
    ):
        """Downloads and prepares dataset for reading.

        Args:
            output_dir (`str`, *optional*):
                Output directory for the dataset.
                Default to this builder's `cache_dir`, which is inside `~/.cache/huggingface/datasets` by default.

                <Added version="2.5.0"/>
            download_config (`DownloadConfig`, *optional*):
                Specific download configuration parameters.
            download_mode ([`DownloadMode`] or `str`, *optional*):
                Select the download/generate mode, default to `REUSE_DATASET_IF_EXISTS`.
            verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
                Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).

                <Added version="2.9.1"/>
            ignore_verifications (`bool`, defaults to `False`):
                Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/...).

                <Deprecated version="2.9.1">

                `ignore_verifications` was deprecated in version 2.9.1 and will be removed in 3.0.0.
                Please use `verification_mode` instead.

                </Deprecated>
            try_from_hf_gcs (`bool`):
                If `True`, it will try to download the already prepared dataset from the HF Google cloud storage.
            dl_manager (`DownloadManager`, *optional*):
                Specific `DownloadManger` to use.
            base_path (`str`, *optional*):
                Base path for relative paths that are used to download files. This can be a remote url.
                If not specified, the value of the `base_path` attribute (`self.base_path`) will be used instead.
            use_auth_token (`Union[str, bool]`, *optional*):
                Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
                If True, or not specified, will get token from ~/.huggingface.

                <Deprecated version="2.7.1">

                Pass `use_auth_token` to the initializer/`load_dataset_builder` instead.

                </Deprecated>
            file_format (`str`, *optional*):
                Format of the data files in which the dataset will be written.
                Supported formats: "msgpack", "parquet". Default to "msgpack" format.
                If the format is "parquet", then image and audio data are embedded into the Parquet files instead of pointing to local files.

                <Added version="2.5.0"/>
            max_shard_size (`Union[str, int]`, *optional*):
                Maximum number of bytes written per shard, default is "500MB".
                The size is based on uncompressed data size, so in practice your shard files may be smaller than
                `max_shard_size` thanks to Parquet compression for example.

                <Added version="2.5.0"/>
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.7.0"/>
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the caching file-system backend, if any.

                <Added version="2.5.0"/>
            **download_and_prepare_kwargs (additional keyword arguments): Keyword arguments.

        Example:

        Download and prepare the dataset as Arrow files that can be loaded as a Dataset using `builder.as_dataset()`:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> ds = builder.download_and_prepare()
        ```

        Download and prepare the dataset as sharded Parquet files locally:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> ds = builder.download_and_prepare("./output_dir", file_format="parquet")
        ```

        Download and prepare the dataset as sharded Parquet files in a cloud storage:

        ```py
        >>> from datasets import load_dataset_builder
        >>> storage_options = {"key": aws_access_key_id, "secret": aws_secret_access_key}
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> ds = builder.download_and_prepare("s3://my-bucket/my_rotten_tomatoes", storage_options=storage_options, file_format="parquet")
        ```
        """
        if ignore_verifications != "deprecated":
            verification_mode = (
                VerificationMode.NO_CHECKS
                if ignore_verifications
                else VerificationMode.ALL_CHECKS
            )
            warnings.warn(
                "'ignore_verifications' was deprecated in favor of 'verification_mode' in version 2.9.1 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'verification_mode={verification_mode.value}' instead.",
                FutureWarning,
            )
        if use_auth_token != "deprecated":
            warnings.warn(
                "'use_auth_token' was deprecated in version 2.7.1 and will be removed in 3.0.0. Pass `use_auth_token` to the initializer/`load_dataset_builder` instead.",
                FutureWarning,
            )
        else:
            use_auth_token = self.use_auth_token

        output_dir = output_dir if output_dir is not None else self._cache_dir
        # output_dir can be a remote bucket on GCS or S3 (when using BeamBasedBuilder for distributed data processing)
        fs_token_paths = fsspec.get_fs_token_paths(
            output_dir, storage_options=storage_options
        )
        self._fs: fsspec.AbstractFileSystem = fs_token_paths[0]
        is_local = not is_remote_filesystem(self._fs)
        self._output_dir = (
            fs_token_paths[2][0]
            if is_local
            else self._fs.unstrip_protocol(fs_token_paths[2][0])
        )

        download_mode = DownloadMode(
            download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS
        )
        verification_mode = VerificationMode(
            verification_mode or VerificationMode.BASIC_CHECKS
        )
        base_path = base_path if base_path is not None else self.base_path

        if file_format is not None and file_format not in ["msgpack"]:
            raise ValueError(
                f"Unsupported file_format: {file_format}. Expected 'msgpack'"
            )

        if self._fs._strip_protocol(self._output_dir) == "":
            # We don't support the root directory, because it has no dirname,
            # and we need a dirname to use a <dirname>.incomplete directory
            # when the dataset is being written
            raise RuntimeError(
                f"Unable to download and prepare the dataset at the root {self._output_dir}. "
                f"Please specify a subdirectory, e.g. '{self._output_dir + self.name}'"
            )

        if dl_manager is None:
            if download_config is None:
                download_config = DownloadConfig(
                    cache_dir=self._cache_downloaded_dir,
                    force_download=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    force_extract=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    use_etag=False,
                    num_proc=num_proc,
                    use_auth_token=use_auth_token,
                    storage_options=self.storage_options,
                )  # We don't use etag for data files to speed up the process
            dl_manager = FusionDownloadManager(
                dataset_name=self.name,
                download_config=download_config,
                data_dir=self.config.data_dir,
                base_path=base_path,
                record_checksums=(
                    self._record_infos
                    or verification_mode == VerificationMode.ALL_CHECKS
                ),
            )

        if (
            isinstance(dl_manager, MockDownloadManager)
            or not is_local
            or file_format != "msgpack"
            or max_shard_size is not None
        ):
            try_from_hf_gcs = False
        self.dl_manager = dl_manager

        # Prevent parallel local disk operations
        if is_local:
            # Create parent directory of the output_dir to put the lock file in there
            Path(self._output_dir).parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._output_dir + "_builder.lock"

        # File locking only with local paths; no file locking on GCS or S3
        with FileLock(lock_path) if is_local else contextlib.nullcontext():
            # Check if the data already exists
            path_join = os.path.join if is_local else posixpath.join
            data_exists = self._fs.exists(
                path_join(self._output_dir, config.DATASET_INFO_FILENAME)
            )
            if data_exists and download_mode == DownloadMode.REUSE_DATASET_IF_EXISTS:
                logger.warning(f"Found cached dataset {self.name} ({self._output_dir})")
                # We need to update the info in case some splits were added in the meantime
                # for example when calling load_dataset from multiple workers.
                self.info = self._load_info()
                self.download_post_processing_resources(dl_manager)
                return

            logger.info(f"Generating dataset {self.name} ({self._output_dir})")
            if is_local:  # if cache dir is local, check for available space
                if not has_sufficient_disk_space(
                    self.info.size_in_bytes or 0,
                    directory=Path(self._output_dir).parent,
                ):
                    raise OSError(
                        f"Not enough disk space. Needed: {size_str(self.info.size_in_bytes or 0)} (download: {size_str(self.info.download_size or 0)}, generated: {size_str(self.info.dataset_size or 0)}, post-processed: {size_str(self.info.post_processing_size or 0)})"
                    )

            @contextlib.contextmanager
            def incomplete_dir(dirname):
                """Create temporary dir for dirname and rename on exit."""
                if not is_local:
                    self._fs.makedirs(dirname, exist_ok=True)
                    yield dirname
                else:
                    tmp_dir = dirname + ".incomplete"
                    os.makedirs(tmp_dir, exist_ok=True)
                    try:
                        yield tmp_dir
                        if os.path.isdir(dirname):
                            shutil.rmtree(dirname)
                        # LocalFileSystem.mv does copy + rm, it is more efficient to simply rename a local directory
                        shutil.move(tmp_dir, dirname)
                    finally:
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)

            # Print is intentional: we want this to always go to stdout so user has
            # information needed to cancel download/preparation if needed.
            # This comes right before the progress bar.
            if self.info.size_in_bytes:
                print(
                    f"Downloading and preparing dataset {self.info.builder_name}/{self.info.config_name} "
                    f"(download: {size_str(self.info.download_size)}, generated: {size_str(self.info.dataset_size)}, "
                    f"post-processed: {size_str(self.info.post_processing_size)}, "
                    f"total: {size_str(self.info.size_in_bytes)}) to {self._output_dir}..."
                )
            else:
                _dest = (
                    self._fs._strip_protocol(self._output_dir)
                    if is_local
                    else self._output_dir
                )
                print(
                    f"Downloading and preparing dataset {self.info.builder_name}/{self.info.config_name} to {_dest}..."
                )

            self._check_manual_download(dl_manager)

            # Create a tmp dir and rename to self._output_dir on successful exit.
            with incomplete_dir(self._output_dir) as tmp_output_dir:
                # Temporarily assign _output_dir to tmp_data_dir to avoid having to forward
                # it to every sub function.
                with temporary_assignment(self, "_output_dir", tmp_output_dir):
                    # Try to download the already prepared dataset files
                    downloaded_from_gcs = False
                    if try_from_hf_gcs:
                        try:
                            self._download_prepared_from_hf_gcs(
                                dl_manager.download_config
                            )
                            downloaded_from_gcs = True
                        except (DatasetNotOnHfGcsError, MissingFilesOnHfGcsError):
                            logger.info(
                                "Dataset not on Hf google storage. Downloading and preparing it from source"
                            )
                        except ConnectionError:
                            logger.warning(
                                "HF google storage unreachable. Downloading and preparing it from source"
                            )
                    if not downloaded_from_gcs:
                        prepare_split_kwargs = {"file_format": file_format}
                        if max_shard_size is not None:
                            prepare_split_kwargs["max_shard_size"] = max_shard_size
                        if num_proc is not None:
                            prepare_split_kwargs["num_proc"] = num_proc
                        self._download_and_prepare(
                            dl_manager=dl_manager,
                            verification_mode=verification_mode,
                            **prepare_split_kwargs,
                            **download_and_prepare_kwargs,
                        )
                    # Sync info
                    self.info.dataset_size = sum(
                        split.num_bytes for split in self.info.splits.values()
                    )
                    self.info.download_checksums = (
                        dl_manager.get_recorded_sizes_checksums()
                    )
                    self.info.size_in_bytes = (
                        self.info.dataset_size + self.info.download_size
                    )
                    # Save info
                    self._save_info()

            # Download post processing resources
            self.download_post_processing_resources(dl_manager)

            print(
                f"Dataset {self.name} downloaded and prepared to {self._output_dir}. "
                f"Subsequent calls will reuse this data."
            )

    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        check_duplicate_keys: bool,
        file_format="msgpack",
        num_proc: Optional[int] = None,
        max_shard_size: Optional[Union[int, str]] = None,
    ):
        max_shard_size = convert_file_size_to_int(
            max_shard_size or config.MAX_SHARD_SIZE
        )
        is_local = not is_remote_filesystem(self._fs)
        path_join = os.path.join if is_local else posixpath.join

        if self.info.splits is not None:
            split_info = self.info.splits[split_generator.name]
        else:
            split_info = split_generator.split_info

        SUFFIX = "-JJJJJ-SSSSS-of-NNNNN"
        fname = f"{self.name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = path_join(self._output_dir, fname)

        num_proc_per_shard = 1
        if num_proc and num_proc > 1:
            num_input_shards = _number_of_shards_in_gen_kwargs(
                split_generator.gen_kwargs
            )
            if num_input_shards <= 1 and num_proc is not None:
                logger.warning(
                    f"Using num_proc={num_proc} for a single shard instead of multiprocessing with multiple hards for {split_info.name} split."
                )
                num_proc_per_shard = num_proc
                num_proc = 1
            elif num_proc is not None and num_input_shards < num_proc:
                logger.info(
                    f"Setting num_proc from {num_proc} to {num_input_shards} for the {split_info.name} split as it only contains {num_input_shards} shards."
                )
                num_proc = num_input_shards

        pbar = logging.tqdm(
            disable=not logging.is_progress_bar_enabled(),
            unit=" examples",
            total=split_info.num_examples,
            leave=False,
            desc=f"Generating {split_info.name} split",
        )

        _prepare_split_args = {
            "fpath": fpath,
            "file_format": file_format,
            "max_shard_size": max_shard_size,
            "split_info": split_info,
            "check_duplicate_keys": check_duplicate_keys,
            "num_proc_per_shard": num_proc_per_shard,
        }

        if num_proc is None or num_proc == 1:
            result = None
            gen_kwargs = split_generator.gen_kwargs
            job_id = 0
            with pbar:
                for job_id, done, content in self._prepare_split_single(
                    gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args
                ):
                    if done:
                        result = content
                    else:
                        pbar.update(content)
            # wrapping everything into lists for consistency with the multiprocessed code path
            assert result is not None, "Failed to retrieve results from prepare_split"
            (
                examples_per_job,
                bytes_per_job,
                features_per_job,
                shards_per_job,
                shard_lengths_per_job,
            ) = [[item] for item in result]
        else:
            kwargs_per_job = [
                {"gen_kwargs": gen_kwargs, "job_id": job_id, **_prepare_split_args}
                for job_id, gen_kwargs in enumerate(
                    _split_gen_kwargs(split_generator.gen_kwargs, max_num_jobs=num_proc)
                )
            ]
            num_jobs = len(kwargs_per_job)

            examples_per_job = [None] * num_jobs
            bytes_per_job = [None] * num_jobs
            features_per_job = [None] * num_jobs
            shards_per_job = [None] * num_jobs
            shard_lengths_per_job = [None] * num_jobs

            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(
                        pool, self._prepare_split_single, kwargs_iterable=kwargs_per_job
                    ):
                        if done:
                            # the content is the result of the job
                            (
                                examples_per_job[job_id],
                                bytes_per_job[job_id],
                                features_per_job[job_id],
                                shards_per_job[job_id],
                                shard_lengths_per_job[job_id],
                            ) = content
                        else:
                            # the content is the number of examples progress update
                            pbar.update(content)

            assert (
                None not in examples_per_job
            ), f"Failed to retrieve results from prepare_split: result list {examples_per_job} still contains None - at least one worker failed to return its results"

        total_shards = sum(shards_per_job)
        total_num_examples = sum(examples_per_job)
        total_num_bytes = sum(bytes_per_job)
        features = features_per_job[0]

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        # should rename everything at the end
        logger.debug(f"Renaming {total_shards} shards.")
        if total_shards > 1:
            # use the -SSSSS-of-NNNNN pattern

            def _rename_shard(shard_and_job: Tuple[int]):
                shard_id, job_id = shard_and_job
                global_shard_id = sum(shards_per_job[:job_id]) + shard_id
                self._rename(
                    fpath.replace("SSSSS", f"{shard_id:05d}").replace(
                        "JJJJJ", f"{job_id:05d}"
                    ),
                    fpath.replace("JJJJJ-SSSSS", f"{global_shard_id:05d}").replace(
                        "NNNNN", f"{total_shards:05d}"
                    ),
                )

                for suffix in [
                    SUFFIX_FILTER,
                    SUFFIX_KEY_HASHES,
                    SUFFIX_KEYS,
                    SUFFIX_OFFSETS,
                    ".md5",
                ]:
                    path = str(Path(fpath).with_suffix(f".msgpack{suffix}"))
                    self._rename(
                        path.replace("SSSSS", f"{shard_id:05d}").replace(
                            "JJJJJ", f"{job_id:05d}"
                        ),
                        path.replace("JJJJJ-SSSSS", f"{global_shard_id:05d}").replace(
                            "NNNNN", f"{total_shards:05d}"
                        ),
                    )

            shards_and_jobs = [
                (shard_id, job_id)
                for job_id, num_shards in enumerate(shards_per_job)
                for shard_id in range(num_shards)
            ]
            thread_map(_rename_shard, shards_and_jobs, disable=True, max_workers=64)

            split_generator.split_info.shard_lengths = [
                shard_length
                for shard_lengths in shard_lengths_per_job
                for shard_length in shard_lengths
            ]
        else:
            # don't use any pattern
            shard_id, job_id = 0, 0
            self._rename(
                fpath.replace("SSSSS", f"{shard_id:05d}").replace(
                    "JJJJJ", f"{job_id:05d}"
                ),
                fpath.replace(SUFFIX, ""),
            )
            for suffix in [
                SUFFIX_FILTER,
                SUFFIX_KEY_HASHES,
                SUFFIX_KEYS,
                SUFFIX_OFFSETS,
                ".md5",
            ]:
                path = str(Path(fpath).with_suffix(f".msgpack{suffix}"))
                self._rename(
                    path.replace("SSSSS", f"{shard_id:05d}").replace(
                        "JJJJJ", f"{job_id:05d}"
                    ),
                    path.replace(SUFFIX, ""),
                )

        if self.info.features is None:
            self.info.features = features

    def _prepare_split_single(
        self,
        gen_kwargs: dict,
        fpath: str,
        file_format: str,
        max_shard_size: int,
        split_info: SplitInfo,
        check_duplicate_keys: bool,
        num_proc_per_shard: int,
        job_id: int,
    ) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
        generator = self._generate_examples(**gen_kwargs)
        writer_class = MsgpackWriter
        shard_lengths = []
        total_num_examples, total_num_bytes = 0, 0

        shard_id = 0
        num_examples_progress_update = 0
        try:
            writer = writer_class(
                self.info.features,
                fpath.replace("SSSSS", f"{shard_id:05d}").replace(
                    "JJJJJ", f"{job_id:05d}"
                ),
                overwrite=True,
            )
            try:
                _time = time.time()

                # process samples in a multithreading environemnt instead
                pool = ThreadPool(num_proc_per_shard)

                def record_preprocessor(input):
                    key, record = input
                    if self.config.preprocess_transforms is not None:
                        record = self.config.preprocess_transforms(record)
                    record = (
                        self.info.features.encode_example(record)
                        if self.info.features is not None
                        else record
                    )
                    return key, record

                for key, record in pool.imap(record_preprocessor, generator):
                    # we do not perform sharding on the go as done in hf arrow builder this is a multiprocessing loop and it might
                    # mess up the file writing
                    if not isinstance(key, str):  # msgpack only supports string keys
                        key = str(key)
                    writer.write(record, key)
                    num_examples_progress_update += 1
                    if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield job_id, False, num_examples_progress_update
                        num_examples_progress_update = 0
            finally:
                yield job_id, False, num_examples_progress_update
                num_shards = shard_id + 1
                num_examples, num_bytes = writer.finalize()
                writer.close()
                shard_lengths.append(num_examples)
                total_num_examples += num_examples
                total_num_bytes += num_bytes
        except Exception as e:
            # Ignore the writer's error for no examples written to the file if this error was caused by the error in _generate_examples before the first example was yielded
            if isinstance(e, SchemaInferenceError) and e.__context__ is not None:
                e = e.__context__
            raise DatasetGenerationError(
                "An error occurred while generating the dataset"
            ) from e

        yield job_id, True, (
            total_num_examples,
            total_num_bytes,
            writer._features,
            num_shards,
            shard_lengths,
        )

    def _as_dataset(
        self,
        split: Union[ReadInstruction, Split] = Split.TRAIN,
        in_memory: bool = False,
    ) -> Dataset:
        """Constructs a `Dataset`.

        This is the internal implementation to overwrite called when user calls
        `as_dataset`. It should read the pre-processed datasets files and generate
        the `Dataset` object.

        Args:
            split (`datasets.Split`):
                which subset of the data to read.
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            `Dataset`
        """
        cache_dir = self._fs._strip_protocol(self._output_dir)
        dataset_kwargs = MsgpackReader(cache_dir, self.info).read(
            name=self.name,
            instructions=split,
            split_infos=self.info.splits.values(),
            in_memory=in_memory,
        )
        fingerprint = self._get_dataset_fingerprint(split)
        return MsgpackBasedDataset(fingerprint=fingerprint, **dataset_kwargs)
