from __future__ import annotations

import copy
import io
import json
import os
import pickle
import sys
from abc import ABC
from pathlib import Path
from typing import Callable, Optional, Type

import datasets
import ignite.distributed as idist
import PIL
from datadings.reader import MsgpackReader as MsgpackFileReader
from datasets import DownloadConfig
from torch.utils.data import BatchSampler, DataLoader, Dataset

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.msgpack.dataset import (
    MsgpackBasedDataset,
    MsgpackBasedTorchDataset,
)
from torchfusion.core.data.factory.dataset import DatasetFactory
from torchfusion.core.data.text_utils.tokenizers.factory import TokenizerFactory
from torchfusion.core.data.text_utils.tokenizers.hf_tokenizer import (
    HuggingfaceTokenizer,
)
from torchfusion.core.data.train_val_samplers.base import TrainValSampler
from torchfusion.core.data.utilities.containers import CollateFnDict, TransformsDict
from torchfusion.core.data.utilities.data_visualization import (
    print_batch_info,
    show_batch,
    show_images,
)
from torchfusion.core.data.utilities.dataset_stats import load_or_precalc_dataset_stats
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.utilities.logging import get_logger
from torchfusion.utilities.module_import import ModuleLazyImporter


class TorchDataset(Dataset):
    def __init__(self, dataset, transforms, info):
        self.dataset = dataset
        self.transforms = transforms
        self.info = info

    def __getitem__(self, index):
        # load sample
        sample = self.dataset.iloc[index].to_dict()

        # decode image if required
        image_load_map = {
            DataKeys.IMAGE: DataKeys.IMAGE_FILE_PATH,
            DataKeys.GT_IMAGE: DataKeys.GT_IMAGE_FILE_PATH,
            DataKeys.COND_IMAGE: DataKeys.COND_IMAGE_FILE_PATH,
        }

        for image_key, path_key in image_load_map.items():
            if image_key in sample and isinstance(sample[image_key], (bytes, str)):
                sample[image_key] = PIL.Image.open(io.BytesIO(sample[image_key]))

        # apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)

        # assign sample index
        sample[DataKeys.INDEX] = index

        return sample

    def __len__(self):
        return len(self.dataset)


class TorchMsgpackDataset(Dataset):
    def __init__(self, dataset, transforms, info):
        self.dataset = dataset
        self.transforms = transforms
        self.info = info

    def __getitem__(self, index):
        # load sample
        sample = pickle.loads(self.dataset[index]["data"])

        # decode image if required
        image_load_map = {
            DataKeys.IMAGE: DataKeys.IMAGE_FILE_PATH,
            DataKeys.GT_IMAGE: DataKeys.GT_IMAGE_FILE_PATH,
            DataKeys.COND_IMAGE: DataKeys.COND_IMAGE_FILE_PATH,
        }

        for image_key, path_key in image_load_map.items():
            if image_key in sample and isinstance(sample[image_key], (bytes, str)):
                sample[image_key] = PIL.Image.open(io.BytesIO(sample[image_key]))

        # apply transforms
        if self.transforms is not None:
            sample = self.transforms(sample)

        # assign sample index
        sample[DataKeys.INDEX] = index

        return sample

    def __len__(self):
        return len(self.dataset)


class FusionDataModule(ABC):
    def __init__(
        self,
        dataset_name: str,
        dataset_cache_dir: str,
        cache_file_name: str,
        dataset_dir: Optional[str] = None,
        use_auth_token: bool = False,
        dataset_config_name: str = "default",
        collate_fns: CollateFnDict = CollateFnDict(),
        preprocess_transforms: TransformsDict = TransformsDict(),
        realtime_transforms: TransformsDict = TransformsDict(),
        train_val_sampler: Optional[TrainValSampler] = None,
        preprocess_batch_size: int = 100,
        dataset_kwargs: dict = {},
        enable_caching: bool = True,
        num_proc: int = 8,
        compute_dataset_statistics: bool = False,
        dataset_statistics_n_samples: int = 50000,
        stats_filename: str = "stats",
        features_path: Optional[str] = None,
    ):
        # initialize base classes
        super().__init__()

        # initialize the arguments
        self._dataset_name = dataset_name
        self._dataset_cache_dir = dataset_cache_dir
        self._dataset_dir = dataset_dir
        self._cache_file_name = cache_file_name
        self._use_auth_token = use_auth_token
        self._dataset_config_name = dataset_config_name
        self._collate_fns = collate_fns
        self._preprocess_transforms = preprocess_transforms
        self._realtime_transforms = realtime_transforms
        self._train_val_sampler = train_val_sampler
        self._preprocess_batch_size = preprocess_batch_size
        self._dataset_kwargs = dataset_kwargs
        self._num_proc = num_proc
        self._enable_caching = enable_caching
        self._compute_dataset_statistics = compute_dataset_statistics
        self._dataset_statistics_n_samples = dataset_statistics_n_samples
        self._stats_filename = stats_filename
        self._features_path = features_path

        # initialize splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # setup logger
        self._logger = get_logger()

    def load_features_dataset(self, dataset_class, split: str = "train"):
        _, realtime_transforms = self._get_transforms(split)

        # if features are given we load dataset directly from features but still use info from original dataset
        if self._compute_dataset_statistics:
            self._logger.warning(
                "You wish to compute dataset_statistics with features_path given. "
                "This is not possible. Skipping dataset_statistics computation. "
                "To compute dataset_statistics run the same command without features_path."
            )

        msgpack_readers = []
        for file in os.listdir(self._features_path):
            if file.endswith(".msgpack") and split in file:
                msgpack_readers.append(
                    MsgpackFileReader(Path(self._features_path) / file)
                )

        if issubclass(dataset_class, Dataset):  # torch dataset
            dataset = MsgpackBasedTorchDataset(
                msgpack_readers=msgpack_readers,
                split=split,
                info=dataset_class._info(self._dataset_config_name),
                transforms=realtime_transforms,
            )
        else:
            # remove 'image' info from builder.info as it has been updated by features
            builder = self._get_builder()
            # builder.info.features.pop("image")
            builder.info.features[DataKeys.IMAGE].decode = False

            dataset = MsgpackBasedDataset(
                msgpack_readers=msgpack_readers,
                info=builder.info,
                split=split,
                transforms=realtime_transforms,
            )
        return dataset

    def _load_torch_dataset(self, dataset_class: Type[Dataset], split: str = "train"):
        _, realtime_transforms = self._get_transforms(split)
        dataset = dataset_class(
            config_name=self._dataset_config_name,
            data_dir=self._dataset_dir,
            cache_dir=self._dataset_cache_dir,
            realtime_transforms=realtime_transforms,
            split=split,
            **self._dataset_kwargs,
        )

        if self._compute_dataset_statistics:
            # for computing statistics, we always use evaluation transforms instead of train ones
            dataset._transforms = self._realtime_transforms["test"]
            load_or_precalc_dataset_stats(
                dataset,
                cache_dir=Path(self._dataset_dir) / "fid_stats",
                split=split,
                batch_size=200,
                dataset_statistics_n_samples=self._dataset_statistics_n_samples,
                stats_filename=self._stats_filename,
                logger=self._logger,
            )

        return dataset

    def _load_fusion_dataset(self, split: str = "train"):
        builder = self._get_builder()
        if builder.info.splits is not None and split not in builder.info.splits.keys():
            self._logger.warning(f"The split {split} is not available in this dataset.")
            return

        # then add all additional kwargs
        dataset_build_kwargs = {
            **self._get_builder_kwargs(),
            **dict(
                split=split,
                num_proc=self._num_proc,
            ),
        }

        preprocess_transforms, realtime_transforms = self._get_transforms(split)
        if preprocess_transforms and preprocess_transforms.transforms is not None:
            dataset_build_kwargs["preprocess_transforms"] = preprocess_transforms

        # create the dataset
        self._logger.info("Loading dataset with the following kwargs:")
        self._logger.info(dataset_build_kwargs)
        msgpack_dataset = DatasetFactory.create(
            self._dataset_name,
            **dataset_build_kwargs,
        )

        # assign realtime transforms to msgpack dataset to be applied on runtime
        msgpack_dataset._transforms = realtime_transforms

        # set dataset
        dataset = msgpack_dataset

        if self._compute_dataset_statistics:
            # for computing statistics, we always use evaluation transforms instead of train ones
            dataset._transforms = self._realtime_transforms["test"]
            load_or_precalc_dataset_stats(
                dataset,
                cache_dir=Path(self._dataset_dir) / "fid_stats",
                split=split,
                batch_size=200,
                dataset_statistics_n_samples=self._dataset_statistics_n_samples,
                stats_filename=self._stats_filename,
                logger=self._logger,
            )

        return dataset

    def _get_download_config(self):
        return DownloadConfig(
            cache_dir=Path(self._dataset_cache_dir) / "downloads",
            force_download=False,
            force_extract=False,
            use_etag=False,
            use_auth_token=self._use_auth_token,
            delete_extracted=True,
        )

    def _get_builder_kwargs(self):
        # prepare download config
        download_config = self._get_download_config()

        # get dataset info from builder class of original dataset
        builder_kwargs = dict(
            name=self._dataset_config_name,
            data_dir=self._dataset_dir,
            cache_dir=self._dataset_cache_dir,
            cache_file_name=self._cache_file_name,
            download_config=download_config,
            **self._dataset_kwargs,
        )

        return builder_kwargs

    def _get_builder(self):
        # create the dataset
        return DatasetFactory.create_builder(
            self._dataset_name,
            **self._get_builder_kwargs(),
        )

    def _get_transforms(self, split: str = "train"):
        # get the right transforms
        preprocess_transforms = self._preprocess_transforms[split]
        realtime_transforms = self._realtime_transforms[split]
        if realtime_transforms.transforms is None:
            realtime_transforms = None
        if preprocess_transforms.transforms is None:
            preprocess_transforms = None
        return preprocess_transforms, realtime_transforms

    def _load_dataset(
        self,
        split: str = "train",
    ) -> Dataset:
        try:
            dataset_class = None
            if not self._dataset_name in ["imagefolder", "fusion_image_folder"]:
                dataset_class = ModuleLazyImporter.get_datasets().get(
                    self._dataset_name, None
                )

                if dataset_class is None:
                    raise ValueError(
                        f"Dataset class for {self._dataset_name} not found. "
                        f"Please make sure the dataset is added to registry. Available datasets: {ModuleLazyImporter.get_datasets()}"
                    )

                # initialize the class
                dataset_class = dataset_class()

            if self._features_path is not None:
                return self.load_features_dataset(
                    dataset_class=dataset_class, split=split
                )

            if dataset_class is not None and issubclass(
                dataset_class, Dataset
            ):  # if this is a torch dataset we use it directly
                return self._load_torch_dataset(dataset_class, split=split)
            else:
                return self._load_fusion_dataset(split=split)
        except Exception as exc:
            self._logger.exception(
                f"Exception raised while loading the dataset "
                f"[{self._dataset_name}]: {exc}"
            )
            sys.exit(1)

    def setup(
        self,
        stage: TrainingStage = TrainingStage.train,
        do_train: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        use_test_set_for_val: bool = False,
    ) -> None:
        from torch.utils.data import Subset

        if stage is not None:
            self._logger.info(f"Loading data for stage == {stage}")
        else:
            self._logger.info(f"Loading data for stage == train|test|val")

        # Assign train/val datasets for use in dataloaders using the train/val sampler
        # lightning calls training stage 'fit'
        if do_train and (stage == TrainingStage.train or stage is None):
            # here we load the dataset itself.
            self.train_dataset = self._load_dataset(
                split="train",
            )

            if str(datasets.Split.VALIDATION) in self.train_dataset.info.splits.keys():
                self.val_dataset = self._load_dataset(
                    split=str(datasets.Split.VALIDATION),
                )
            elif self._train_val_sampler is not None:
                self.train_dataset, self.val_dataset = self._train_val_sampler(
                    self.train_dataset
                )
            elif not use_test_set_for_val:
                logger = get_logger()
                logger.warning(
                    "Using train set as validation set as no validation dataset exists."
                    " If this behavior is not required set, do_val=False in config."
                )
                self.val_dataset = copy.deepcopy(self.train_dataset)

            if len(self.val_dataset) == 0:
                logger = get_logger()
                logger.warning(
                    "Using train set as validation set as no validation dataset exists."
                    " If this behavior is not required set, do_val=False in config."
                )
                self.val_dataset = copy.deepcopy(self.train_dataset)

            # if max_train_samples is set get the given number of examples
            # from the dataset
            if max_train_samples is not None:
                self.train_dataset = Subset(
                    self.train_dataset,
                    range(0, max_train_samples),
                )

            # if max_val_samples is set get the given number of examples
            # from the dataset
            if max_val_samples is not None and self.val_dataset is not None:
                self.val_dataset = Subset(
                    self.val_dataset,
                    range(0, max_val_samples),
                )

            if use_test_set_for_val:
                logger = get_logger()
                logger.warning(
                    "Using test set as validation set."
                    " If this behavior is not required set, use_test_set_for_val=False in config."
                )

                self.val_dataset = self._load_dataset(
                    split="test",
                )

                if self.val_dataset is not None:
                    if max_val_samples is not None:
                        self.val_dataset = Subset(
                            self.val_dataset,
                            range(0, max_val_samples),
                        )

            self._logger.info(f"Training set size = {len(self.train_dataset)}")
            if self.val_dataset is not None:
                self._logger.info(f"Validation set size = {len(self.val_dataset)}")

        # Assign test dataset for use in dataloader(s)
        if stage == TrainingStage.test or stage is None:
            try:
                self.test_dataset = self._load_dataset(
                    split="test",
                )

                if self.test_dataset is not None:
                    # if max_test_samples is set get the given number of examples
                    # from the dataset
                    if max_test_samples is not None:
                        self.test_dataset = Subset(
                            self.test_dataset,
                            range(0, max_test_samples),
                        )

                if self.test_dataset is not None:
                    self._logger.info(f"Test set size = {len(self.test_dataset)}")
            except Exception as e:
                self._logger.info(f"Error while loading test dataset: {e}")

    def setup_train_dataloader(
        self,
        dataset: Dataset,
        per_device_train_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_data: bool = True,
        dataloader_drop_last: bool = True,
        batch_sampler_wrapper: Optional[BatchSampler] = None,
        dataloader_init_fn: Callable = idist.auto_dataloader,
    ):
        """
        Defines the torch dataloader for train dataset.
        """

        from torch.utils.data import RandomSampler, SequentialSampler

        # setup sampler
        if shuffle_data:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        # setup custom batch sampler
        batch_sampler = (
            batch_sampler_wrapper(sampler)
            if batch_sampler_wrapper is not None
            else None
        )
        if batch_sampler is None:
            return dataloader_init_fn(
                dataset,
                sampler=sampler,
                batch_size=per_device_train_batch_size * idist.get_world_size(),
                collate_fn=self._collate_fns.train,
                num_workers=dataloader_num_workers,
                pin_memory=pin_memory,
                drop_last=True if idist.get_world_size() > 1 else dataloader_drop_last,
            )
        else:
            return dataloader_init_fn(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self._collate_fns.train,
                num_workers=dataloader_num_workers,
                pin_memory=pin_memory,
                drop_last=dataloader_drop_last,
            )

    def train_dataloader(
        self,
        per_device_train_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_data: bool = True,
        dataloader_drop_last: bool = True,
        batch_sampler_wrapper: Optional[BatchSampler] = None,
        dataloader_init_fn: Callable = idist.auto_dataloader,
    ) -> DataLoader:
        return self.setup_train_dataloader(
            self.train_dataset,
            per_device_train_batch_size=per_device_train_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            shuffle_data=shuffle_data,
            dataloader_drop_last=dataloader_drop_last,
            batch_sampler_wrapper=batch_sampler_wrapper,
            dataloader_init_fn=dataloader_init_fn,
        )

    def val_dataloader(
        self,
        per_device_eval_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
        dataloader_init_fn: Callable = idist.auto_dataloader,
    ) -> DataLoader:
        """
        Defines the torch dataloader for validation dataset.
        """

        import ignite.distributed as idist
        from torch.utils.data import SequentialSampler

        if self.val_dataset is None:
            return

        if idist.get_world_size() > 1:
            if len(self.val_dataset) % idist.get_world_size() != 0:
                self._logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )

            # let ignite handle distributed sampler
            sampler = None
        else:
            sampler = SequentialSampler(self.val_dataset)

        return dataloader_init_fn(
            self.val_dataset,
            sampler=sampler,
            batch_size=per_device_eval_batch_size * idist.get_world_size(),
            collate_fn=self._collate_fns.validation,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            drop_last=False,  # drop last is always false for validation
        )

    def setup_test_dataloader(
        self,
        dataset,
        per_device_eval_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
        dataloader_init_fn: Callable = idist.auto_dataloader,
    ) -> DataLoader:
        """
        Defines the torch dataloader for test dataset.
        """
        import ignite.distributed as idist
        from torch.utils.data import SequentialSampler

        if idist.get_world_size() > 1:
            if len(dataset) % idist.get_world_size() != 0:
                self._logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )

            # let ignite handle distributed sampler
            sampler = None
        else:
            sampler = SequentialSampler(dataset)

        return dataloader_init_fn(
            dataset,
            sampler=sampler,
            batch_size=per_device_eval_batch_size * idist.get_world_size(),
            collate_fn=self._collate_fns.test,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            shuffle=False,  # data is always sequential for test
            drop_last=False,  # drop last is always false for test
        )

    def test_dataloader(
        self,
        per_device_eval_batch_size: int,
        dataloader_num_workers: int = 4,
        pin_memory: bool = True,
        dataloader_init_fn: Callable = idist.auto_dataloader,
    ) -> DataLoader:
        return self.setup_test_dataloader(
            self.test_dataset,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            dataloader_init_fn=dataloader_init_fn,
        )

    def get_dataloader(self, stage: TrainingStage):
        if stage == TrainingStage.train:
            return self.train_dataloader()
        elif stage == TrainingStage.test:
            return self.test_dataloader()
        elif stage == TrainingStage.validation:
            return self.val_dataloader()

    def show_batch(self, batch):
        builder = self._get_builder()
        tokenizer = None
        if hasattr(builder.config, "tokenizer_config"):
            # create tokenizer
            tokenizer_name = builder.config.tokenizer_config["name"]
            tokenizer_kwargs = builder.config.tokenizer_config["kwargs"]
            tokenizer = TokenizerFactory.create(tokenizer_name, tokenizer_kwargs)
            tokenizer = (
                tokenizer.tokenizer
                if isinstance(tokenizer, HuggingfaceTokenizer)
                else tokenizer
            )
        print_batch_info(batch, tokenizer=tokenizer)
        show_batch(batch)
