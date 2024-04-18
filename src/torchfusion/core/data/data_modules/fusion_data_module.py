from __future__ import annotations

import copy
import dataclasses
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
from torch.utils.data import BatchSampler, DataLoader, Dataset, Subset

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.data_augmentations.general import DictTransform
from torchfusion.core.data.datasets.msgpack.dataset import (
    MsgpackBasedDataset,
    MsgpackBasedTorchDataset,
    TransformedDataset,
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
)
from torchfusion.core.data.utilities.dataset_stats import load_or_precalc_dataset_stats
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.general import (
    pretty_print_dict,
    print_transforms,
)
from torchfusion.core.utilities.logging import get_logger
from torchfusion.core.utilities.module_import import ModuleLazyImporter


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
        # if features are given we load dataset directly from features but still use info from original dataset
        if self._compute_dataset_statistics:
            self._logger.warning(
                "You wish to compute dataset_statistics with features_path given. "
                "This is not possible. Skipping dataset_statistics computation. "
                "To compute dataset_statistics with args.data_args.compute_dataset_statistics=True"
                "run the same command without features_path provided."
            )
            exit()

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
            )
        return dataset

    def _load_torch_dataset(self, dataset_class: Type[Dataset], split: str = "train"):
        dataset = dataset_class(
            config_name=self._dataset_config_name,
            data_dir=self._dataset_dir,
            cache_dir=self._dataset_cache_dir,
            split=split,
            **self._dataset_kwargs,
        )

        if self._compute_dataset_statistics:
            self._compute_dataset_statistics_fn(dataset=dataset, split=split)

        return dataset

    def _compute_dataset_statistics_fn(self, dataset, split):
        self._logger.info(
            "You have set args.data_args.compute_dataset_statistics=True. "
            "This will compute the FID stats for this dataset. "
        )
        # for computing statistics, we always use evaluation transforms instead of train ones
        self._logger.info("Using following transform for computing dataset statistics:")
        transforms = self._realtime_transforms["test"]
        print_transforms(transforms)
        load_or_precalc_dataset_stats(
            TransformedDataset(dataset=dataset, transforms=transforms),
            cache_dir=Path(self._dataset_dir) / "fid_stats",
            split=split,
            batch_size=200,
            dataset_statistics_n_samples=self._dataset_statistics_n_samples,
            stats_filename=self._stats_filename,
            logger=self._logger,
        )
        self._logger.info("Dataset statistics computed successfully.")

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

        preprocess_transforms, _ = self._get_transforms(split)
        if preprocess_transforms and preprocess_transforms.transforms is not None:
            dataset_build_kwargs["preprocess_transforms"] = preprocess_transforms

        # create the dataset
        self._logger.info(
            f"Loading dataset with the following kwargs: {pretty_print_dict(dataset_build_kwargs)}"
        )
        dataset = DatasetFactory.create(
            self._dataset_name,
            **dataset_build_kwargs,
        )

        if self._compute_dataset_statistics:
            self._compute_dataset_statistics_fn(dataset=dataset, split=split)

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

    def _get_builder_or_class(self):
        dataset_class = self._get_dataset_class()

        if dataset_class is not None and issubclass(
            dataset_class, Dataset
        ):  # if this is a torch dataset we use it directly
            return dataset_class
        else:
            return self._get_builder()

    def _get_tokenizer_config_if_available(self):
        dataset_class = self._get_dataset_class()

        # is this torch dataset? check if we have an argument of tokenizer_config passed to the dataset
        tokenizer_config = None
        if dataset_class is not None and issubclass(dataset_class, Dataset):
            if "tokenizer_config" in self._dataset_kwargs:
                return tokenizer_config
        else:  # else check for it in th config
            builder = self._get_builder()
            if hasattr(builder.config, "tokenizer_config"):
                return builder.config.tokenizer_config

    def _get_tokenizer_if_available(self):
        tokenizer_config = self._get_tokenizer_config_if_available()
        if tokenizer_config is None:
            return

        # create tokenizer
        tokenizer_name = tokenizer_config["name"]
        tokenizer_kwargs = tokenizer_config["kwargs"]
        tokenizer = TokenizerFactory.create(tokenizer_name, tokenizer_kwargs)
        tokenizer = (
            tokenizer.tokenizer
            if isinstance(tokenizer, HuggingfaceTokenizer)
            else tokenizer
        )
        return tokenizer

    def _get_dataset_class(self):
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

        return dataset_class

    def _get_transforms(self, split: str = "train"):
        # get the right transforms
        preprocess_transforms = self._preprocess_transforms[split]
        realtime_transforms = self._realtime_transforms[split]
        if realtime_transforms is None or realtime_transforms.transforms is None:
            realtime_transforms = None
        if preprocess_transforms is None or preprocess_transforms.transforms is None:
            preprocess_transforms = None
        return preprocess_transforms, realtime_transforms

    def _load_dataset(
        self,
        split: str = "train",
    ) -> Dataset:
        try:
            dataset_class = self._get_dataset_class()
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

            available_splits = None
            if isinstance(self.train_dataset.info, dict):
                available_splits = self.train_dataset.info["splits"]
            elif isinstance(self.train_dataset.info, datasets.DatasetInfo):
                available_splits = self.train_dataset.info.splits.keys()
            else:
                raise ValueError(
                    "Dataset info should be of type dict or datasets.DatasetInfo"
                )

            if str(datasets.Split.VALIDATION) in available_splits:
                self.val_dataset = self._load_dataset(
                    split=str(datasets.Split.VALIDATION),
                )
            elif self._train_val_sampler is not None:
                self._logger.info(
                    f"Using train/validation sampler [{self._train_val_sampler}] for splitting the "
                    f"dataset with following arguments: {pretty_print_dict(self._train_val_sampler)}"
                )
                self.train_dataset, self.val_dataset = self._train_val_sampler(
                    self.train_dataset
                )
            elif not use_test_set_for_val and (
                self.val_dataset is None or len(self.val_dataset) == 0
            ):
                logger = get_logger()
                logger.warning(
                    "Using train set as validation set as no validation dataset exists."
                    " If this behavior is not required set, do_val=False in config or set a "
                    "args.train_val_sampler=random_split/other."
                )
                self.val_dataset = copy.deepcopy(self.train_dataset)

            # if max_train_samples is set get the given number of examples
            # from the dataset
            if max_train_samples is not None:
                max_train_samples = min(max_train_samples, len(self.train_dataset))
                self.train_dataset = Subset(
                    self.train_dataset,
                    range(0, max_train_samples),
                )

            # if max_val_samples is set get the given number of examples
            # from the dataset
            if max_val_samples is not None and self.val_dataset is not None:
                max_val_samples = min(max_val_samples, len(self.val_dataset))
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
                        max_val_samples = min(max_val_samples, len(self.val_dataset))
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
                        max_test_samples = min(max_test_samples, len(self.test_dataset))
                        self.test_dataset = Subset(
                            self.test_dataset,
                            range(0, max_test_samples),
                        )

                if self.test_dataset is not None:
                    self._logger.info(f"Test set size = {len(self.test_dataset)}")
            except Exception as e:
                self._logger.exception(f"Error while loading test dataset: {e}")

        # assign transforms
        if self.train_dataset is not None:
            self.train_dataset = TransformedDataset(
                dataset=self.train_dataset,
                transforms=self._realtime_transforms[str(datasets.Split.TRAIN)],
            )
        if self.val_dataset is not None:
            self.val_dataset = TransformedDataset(
                dataset=self.val_dataset,
                transforms=self._realtime_transforms[str(datasets.Split.VALIDATION)],
            )
        if self.test_dataset is not None:
            self.test_dataset = TransformedDataset(
                dataset=self.test_dataset,
                transforms=self._realtime_transforms[str(datasets.Split.TEST)],
            )

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
        print_batch_info(batch, tokenizer=self._get_tokenizer_if_available())
        show_batch(batch)

    def get_dataset_info(self):
        if self.train_dataset is not None:
            dataset = self.train_dataset._dataset
        elif self.test_dataset is not None:
            dataset = self.test_dataset._dataset
        else:
            raise ValueError("No dataset found in datamodule.")

        dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

        if not hasattr(dataset, "info"):
            raise ValueError(
                "You must define a property info in the torch-like datasets, "
                "which returns relevant class/labels information does not have an info attribute."
            )

        return dataset.info
