"""
Defines the base DataAugmentation class for defining any kind of data augmentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ignite.distributed as idist
from torch.utils.data import Subset

from torchfusion.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from torchfusion.core.args.args import FusionArguments
from torchfusion.core.data.data_augmentations.general import DictTransform
from torchfusion.core.data.data_modules.fusion_data_module import FusionDataModule
from torchfusion.core.data.factory.data_augmentation import DataAugmentationFactory
from torchfusion.core.data.factory.train_val_sampler import TrainValSamplerFactory
from torchfusion.core.data.utilities.containers import CollateFnDict, TransformsDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.tasks import ModelTasks
from torchfusion.core.models.utilities.data_collators import PassThroughCollator
from torchfusion.core.training.functionality.default import DefaultTrainingFunctionality
from torchfusion.core.training.functionality.diffusion import (
    DiffusionTrainingFunctionality,
)
from torchfusion.core.training.functionality.gan import GANTrainingFunctionality
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.general import (
    TransformsWrapper,
    initialize_torch,
    setup_logging,
)
from torchfusion.utilities.dataclasses.dacite_wrapper import from_dict
from torchfusion.utilities.logging import get_logger


class AnalyzerTask(ABC):
    @dataclass
    class Config(AnalyzerTaskConfig):
        pass

    def __init__(
        self, args: FusionArguments, hydra_config, config: AnalyzerTaskConfig
    ) -> None:
        self._args = args
        self._hydra_config = hydra_config
        self._output_dir = None
        self._tb_logger = None
        self._trainer_functionality = None
        self._datamodule = None
        self._data_loader = None
        self._config = self._setup_config(config)
        self._logger = get_logger(hydra_config=hydra_config)

    @property
    def config(self):
        return self._config

    def _setup_config(self, config: AnalyzerTaskConfig):
        try:
            # initialize or validate the config
            if config is None:
                return self.Config()
            elif isinstance(config, dict):
                return from_dict(
                    data_class=self.Config,
                    data=config,
                )
        except Exception as e:
            self._logger.exception(
                f"Exception raised while initializing config for task: {self.__class__}: {e}"
            )
            exit()

    def get_collate_fns(self):
        return CollateFnDict(
            train=PassThroughCollator(),
            validation=PassThroughCollator(),
            test=PassThroughCollator(),
        )

    def _setup_analysis(self, task_name: str):
        # initialize training
        initialize_torch(
            self._args,
            seed=self._args.general_args.seed,
            deterministic=self._args.general_args.deterministic,
        )

        # initialize torch device (cpu or gpu)
        self._device = idist.device()

        # get device rank
        self._rank = idist.get_rank()

        # initialize logging directory and tensorboard logger
        self._output_dir, self._tb_logger = setup_logging(
            output_dir=Path(self._hydra_config.runtime.output_dir) / task_name
        )

    def _setup_trainer_functionality(self):
        if self._args.model_args.model_task == ModelTasks.gan:
            return GANTrainingFunctionality
        elif self._args.model_args.model_task == ModelTasks.diffusion:
            return DiffusionTrainingFunctionality
        else:
            return DefaultTrainingFunctionality

    def _setup_transforms(self):
        def load_from_args(train_augs, eval_augs):
            # define data transforms according to the configuration
            tf = TransformsDict()
            if train_augs is not None:
                tf.train = []
                for aug_args in train_augs:
                    aug = DataAugmentationFactory.create(
                        aug_args.name,
                        aug_args.kwargs,
                    )
                    tf.train.append(aug)

            if eval_augs is not None:
                tf.validation = []
                tf.test = []
                for aug_args in eval_augs:
                    aug = DataAugmentationFactory.create(
                        aug_args.name,
                        aug_args.kwargs,
                    )
                    tf.validation.append(aug)
                    tf.test.append(aug)

            # wrap the transforms in a callable class
            tf.train = TransformsWrapper(tf.train)
            tf.validation = TransformsWrapper(tf.validation)
            tf.test = TransformsWrapper(tf.test)

            return tf

        def print_transforms(tf, title):
            for split in ["train", "validation", "test"]:
                if tf[split].transforms is None:
                    continue
                self._logger.info(f"Defining [{split}] {title}:")
                if idist.get_rank() == 0:
                    for idx, transform in enumerate(tf[split].transforms):
                        if isinstance(transform, DictTransform):
                            self._logger.info(
                                f"{idx}, {transform.key}: {transform.transform}"
                            )
                        else:
                            self._logger.info(f"{idx}, {transform}")

        preprocess_transforms = load_from_args(
            self._args.data_args.train_preprocess_augs,
            self._args.data_args.eval_preprocess_augs,
        )
        realtime_transforms = load_from_args(
            self._args.data_args.train_realtime_augs,
            self._args.data_args.eval_realtime_augs,
        )

        print_transforms(preprocess_transforms, title="preprocess transforms")
        print_transforms(realtime_transforms, title="realtime transforms")
        return preprocess_transforms, realtime_transforms

    def _setup_datamodule(
        self, stage: TrainingStage = TrainingStage.train, override_collate_fns=None
    ) -> FusionDataModule:
        """
        Initializes the datamodule for training.
        """

        import ignite.distributed as idist

        from torchfusion.core.data.data_modules.fusion_data_module import (
            FusionDataModule,
        )

        self._logger.info("Setting up datamodule...")

        # setup transforms
        preprocess_transforms, realtime_transforms = self._setup_transforms()

        # set default collate_fns
        collate_fns = self.get_collate_fns()

        # setup train_val_sampler
        train_val_sampler = None
        if (
            self._args.general_args.do_val
            and not self._args.data_args.data_loader_args.use_test_set_for_val
            and self._args.data_args.train_val_sampler is not None
        ):
            # setup train/val sampler
            train_val_sampler = TrainValSamplerFactory.create(
                self._args.data_args.train_val_sampler.name,
                self._args.data_args.train_val_sampler.kwargs,
            )

        # initialize data module generator function
        datamodule = FusionDataModule(
            dataset_name=self._args.data_args.dataset_name,
            dataset_cache_dir=self._args.data_args.dataset_cache_dir,
            dataset_dir=self._args.data_args.dataset_dir,
            cache_file_name=self._args.data_args.cache_file_name,
            use_auth_token=self._args.data_args.use_auth_token,
            dataset_config_name=self._args.data_args.dataset_config_name,
            collate_fns=collate_fns,
            preprocess_transforms=preprocess_transforms,
            realtime_transforms=realtime_transforms,
            train_val_sampler=train_val_sampler,
            preprocess_batch_size=self._args.data_args.preprocess_batch_size,
            dataset_kwargs=self._args.data_args.dataset_kwargs,
            num_proc=self._args.data_args.num_proc,
            compute_dataset_statistics=self._args.data_args.compute_dataset_statistics,
            dataset_statistics_n_samples=self._args.data_args.dataset_statistics_n_samples,
            stats_filename=self._args.data_args.stats_filename,
            features_path=self._args.data_args.features_path,
        )

        # only download dataset on rank 0, all other ranks wait here for rank 0 to load the datasets
        if self._rank > 0:
            idist.barrier()

        # we manually prepare data and call setup here so dataset related properties can be initalized.
        datamodule.setup(
            stage=stage,
            do_train=self._args.general_args.do_train,
            max_train_samples=self._args.data_args.data_loader_args.max_train_samples,
            max_val_samples=self._args.data_args.data_loader_args.max_val_samples,
            max_test_samples=self._args.data_args.data_loader_args.max_test_samples,
            use_test_set_for_val=self._args.data_args.data_loader_args.use_test_set_for_val,
        )

        if self._rank == 0:
            idist.barrier()

        return datamodule

    def _setup_test_engine(self, model, checkpoint_type: str = "last"):
        # setup training engine
        test_engine = self._trainer_functionality.setup_test_engine(
            args=self._args,
            model=model,
            test_dataloader=self._data_loader,
            output_dir=self._output_dir,
            tb_logger=self._tb_logger,
            device=self._device,
            checkpoint_type=checkpoint_type,
        )
        test_engine.logger = get_logger()

        return test_engine

    def _setup_prediction_engine(self, model, convert_to_tensor: list = []):
        # setup training engine
        test_engine = self._trainer_functionality.setup_prediction_engine(
            args=self._args,
            model=model,
            test_dataloader=self._data_loader,
            output_dir=self._output_dir,
            tb_logger=self._tb_logger,
            device=self._device,
            convert_to_tensor=convert_to_tensor,
        )
        test_engine.logger = get_logger()

        return test_engine

    def _setup_model(
        self,
        summarize: bool = False,
        setup_for_train: bool = True,
        dataset_features: Optional[dict] = None,
        checkpoint: Optional[str] = None,
        strict: bool = False,
    ) -> FusionModel:
        """
        Initializes the model for training.
        """
        from torchfusion.core.models.factory import ModelFactory

        self._logger.info("Setting up model...")

        # setup model
        model = ModelFactory.create_fusion_model(
            self._args,
            checkpoint=checkpoint,
            tb_logger=self._tb_logger,
            dataset_features=dataset_features,
            strict=strict,
        )

        model.setup_model(setup_for_train=setup_for_train)

        # generate model summary
        if summarize:
            model.summarize_model()

        return model

    def _get_dataset_info(self):
        if self._datamodule.train_dataset is not None:
            dataset = self._datamodule.train_dataset._dataset
        elif self._datamodule.test_dataset is not None:
            dataset = self._datamodule.test_dataset._dataset
        else:
            raise ValueError("No dataset found in datamodule.")

        return dataset.dataset.info if isinstance(dataset, Subset) else dataset.info

    def setup(self, task_name: str):
        # setup training
        self._setup_analysis(task_name)

        # setup base training functionality
        self._trainer_functionality = self._setup_trainer_functionality()

        # setup datamodule
        self._datamodule = self._setup_datamodule(stage=None)

    def setup_dataloader(self, collate_fns: CollateFnDict):
        self._datamodule._collate_fns = collate_fns
        stage = TrainingStage.get(self._config.data_split)

        # setup dataloaders
        if stage == TrainingStage.train:
            return self._datamodule.train_dataloader(
                self._args.data_args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_args.data_loader_args.pin_memory,
                shuffle_data=False,
                dataloader_drop_last=False,
            )
        elif stage == TrainingStage.validation:
            return self._datamodule.val_dataloader(
                self._args.data_args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_args.data_loader_args.pin_memory,
            )
        else:
            return self._datamodule.test_dataloader(
                self._args.data_args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_args.data_loader_args.pin_memory,
            )

    def cleanup(self):
        # close tb logger
        self._tb_logger.close()

    @abstractmethod
    def run(self):
        pass
