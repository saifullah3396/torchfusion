"""
Defines the feature attribution generation task.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

from torch.utils.data import Subset
from torchfusion.core.analyzer.tasks.base import AnalyzerTask
from torchfusion.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.tb_logger import FusionTensorboardLogger
from torchfusion.core.utilities.logging import get_logger


class TestModels(AnalyzerTask):
    @dataclass
    class Config(AnalyzerTaskConfig):
        base_dir: str = ""
        model_dirs: List[str] = field(default_factory=lambda: [""])
        checkpoint_prefix: str = "saved_checkpoint"
        checkpoint_keys: List[str] = field(
            default_factory=lambda: ["model", "ema_model"]
        )

    def _setup_model(
        self,
        summarize: bool = False,
        stage: TrainingStage = TrainingStage.train,
        dataset_features: Optional[dict] = None,
        checkpoint: Optional[str] = None,
        strict: Optional[bool] = None,
        wrapper_class: Type[FusionModel] = FusionModel,
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
            wrapper_class=wrapper_class,
        )
        model.setup_model(stage=stage)

        # generate model summary
        if summarize:
            model.summarize_model()

        return model

    def setup(self, task_name: str):
        # setup training
        self._setup_analysis(task_name)

        # setup base training functionality
        self._trainer_functionality = self._setup_trainer_functionality()

        # setup datamodule
        self._datamodule = self._setup_datamodule(stage=None)

        # setup dataloaders
        if self._args.data_loader_args.use_val_set_for_test:
            # setup datamodule (since we need validation dataset, we load the complete datamodule here)
            self._datamodule = self._setup_datamodule()

            # get features info
            self._features_info = (
                self._datamodule.val_dataset.dataset.info
                if isinstance(self._datamodule.val_dataset, Subset)
                else self._datamodule.val_dataset.info
            )
        else:
            # setup datamodule (we only load the test dataset here)
            self._datamodule = self._setup_datamodule(stage=TrainingStage.test)

            # get features info
            self._features_info = (
                self._datamodule.test_dataset.dataset.info
                if isinstance(self._datamodule.test_dataset, Subset)
                else self._datamodule.test_dataset.info
            )

    def run(self):
        logger = get_logger()
        self._test_dataloader = None
        for model_dir in self.config.model_dirs:
            base_checkpoint_dir = (
                Path(self.config.base_dir) / Path(model_dir) / "checkpoints"
            )
            files = os.listdir(base_checkpoint_dir)
            files = sorted([f for f in files if self.config.checkpoint_prefix in f])
            self._logger.info(f"Files = {files}")

            if Path(self._output_dir / model_dir / "results.txt").exists():
                self._logger.info("Testing already finished.")
                exit()
            Path(self._output_dir / model_dir).mkdir(parents=True, exist_ok=True)
            with open(self._output_dir / model_dir / "results.txt", "w") as f:
                f.write("model_name fid\n")
            for checkpoint in files:
                for key in self.config.checkpoint_keys:
                    self._args.model_args.checkpoint_state_dict_key = key

                    logger.info(
                        f"Testing model [{model_dir} with checkpoint [{checkpoint}]], checkpoint_key={key}"
                    )

                    # running checkpoint
                    self._tb_logger = FusionTensorboardLogger(
                        self._output_dir / model_dir / checkpoint / key
                    )

                    # setup model
                    model = self._setup_model(
                        summarize=False,
                        stage=TrainingStage.test,
                        dataset_features=self._features_info.features,
                        checkpoint=base_checkpoint_dir / checkpoint,
                        strict=True,
                    )

                    # now assign collate fns
                    collate_fns = model._nn_model.get_data_collators()
                    self._datamodule._collate_fns = collate_fns

                    # create dataloader
                    if self._test_dataloader is None:
                        if self._args.data_loader_args.use_val_set_for_test:
                            self._test_dataloader = self._datamodule.val_dataloader(
                                self._args.data_loader_args.per_device_eval_batch_size,
                                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                                pin_memory=self._args.data_loader_args.pin_memory,
                            )
                        else:
                            self._test_dataloader = self._datamodule.test_dataloader(
                                self._args.data_loader_args.per_device_eval_batch_size,
                                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                                pin_memory=self._args.data_loader_args.pin_memory,
                            )

                    # print metrics
                    self._logger.info(f"Metrics = {model._metrics.test.keys()}")

                    # setup the test engine
                    # test_engine = self._setup_test_engine(model)

                    # # visualization_engine is just another evaluation engine
                    self._visualization_engine = (
                        self._trainer_functionality.initialize_visualization_engine(
                            args=self._args,
                            model=model,
                            training_engine=None,
                            output_dir=self._output_dir,
                            device=self._device,
                            tb_logger=self._tb_logger,
                        )
                    )

                    # # run visualizer
                    self._visualization_engine.run(
                        self._test_dataloader, max_epochs=1, epoch_length=1
                    )

                    # run test
                    # test_engine.run(self._test_dataloader)
                    # with open (self._output_dir / model_dir / 'results.txt', 'a') as f:
                    #     f.write(f'{model_dir}-{checkpoint}-{key} {test_engine.state.metrics["last/test/fid"]}\n')

                    self._tb_logger.close()
