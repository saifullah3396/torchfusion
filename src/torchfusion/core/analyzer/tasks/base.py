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
from torchfusion.core.data.utilities.loaders import load_datamodule_from_args
from torchfusion.core.data.utilities.transforms import load_transforms_from_config
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
    initialize_torch,
    print_transforms,
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
        self._data_labels = None
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

    def _setup_prediction_engine(self, model, keys_to_device: list = []):
        # setup training engine
        test_engine = self._trainer_functionality.setup_prediction_engine(
            args=self._args,
            model=model,
            test_dataloader=self._data_loader,
            output_dir=self._output_dir,
            tb_logger=self._tb_logger,
            device=self._device,
            keys_to_device=keys_to_device,
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
        self._datamodule, self._data_labels = load_datamodule_from_args(
            self._args, stage=None, rank=self._rank
        )

    def setup_dataloader(self, collate_fns: CollateFnDict):
        self._datamodule._collate_fns = collate_fns
        stage = TrainingStage.get(self._config.data_split)

        # setup dataloaders
        if stage == TrainingStage.train:
            return self._datamodule.train_dataloader(
                self._args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_loader_args.pin_memory,
                shuffle_data=False,
                dataloader_drop_last=False,
            )
        elif stage == TrainingStage.validation:
            return self._datamodule.val_dataloader(
                self._args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_loader_args.pin_memory,
            )
        else:
            return self._datamodule.test_dataloader(
                self._args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_loader_args.pin_memory,
            )

    def cleanup(self):
        # close tb logger
        self._tb_logger.close()

    @abstractmethod
    def run(self):
        pass
