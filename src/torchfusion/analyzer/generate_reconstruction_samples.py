"""
Defines the feature attribution generation task.
"""

from dataclasses import dataclass
from typing import Optional, Type

import torch
from ignite.engine import Events

from torchfusion.analyzer.evaluators import (
    DataSaverHandler,
    EvaluatorEvents,
    EvaluatorKeys,
    ImageReconstructionEvaluator,
)
from torchfusion.core.analyzer.tasks.base import AnalyzerTask
from torchfusion.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from torchfusion.core.constants import DataKeys
from torchfusion.core.models.factory import ModelFactory
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.tb_logger import FusionTensorboardLogger
from torchfusion.utilities.logging import get_logger


class GenerateReconstructionSamples(AnalyzerTask):
    @dataclass
    class Config(AnalyzerTaskConfig):
        pass

    def setup(self, task_name: str):
        super().setup(task_name=task_name)

    def _setup_model(
        self,
        summarize: bool = False,
        stage: TrainingStage = TrainingStage.train,
        dataset_features: Optional[dict] = None,
        checkpoint: Optional[str] = None,
        strict: bool = False,
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

    def run(self):
        logger = get_logger()

        data_key_type_map = {
            DataKeys.INDEX: torch.long,
            DataKeys.IMAGE_FILE_PATH: str,
            DataKeys.IMAGE: torch.float,
        }

        # get data collator required for the model
        stage = TrainingStage.get(self._config.data_split)
        model_class = ModelFactory.get_fusion_nn_model_class(self._args.model_args)
        collate_fns = model_class.get_data_collators(self._args, data_key_type_map=data_key_type_map)
        self._data_loader.collate_fn = collate_fns[stage]

        if self._args.analyzer_args.model_checkpoints is None:
            self._args.analyzer_args.model_checkpoints = [(self._args.model_args.name, None)]

        for model_name, checkpoint in self._args.analyzer_args.model_checkpoints:
            # set output file
            output_file = self._output_dir / f"{model_name}_reconstructions.h5"

            logger.info(f"Running analysis on model [{model_name}]")
            self._tb_logger = FusionTensorboardLogger(self._output_dir / model_name)

            # setup model
            model = self._setup_model(
                summarize=True,
                stage=stage,
                dataset_features=self._data_loader.dataset.info.features,
                checkpoint=checkpoint,
                strict=True,
            )

            prediction_engine = self._setup_prediction_engine(model)
            prediction_engine.register_events(*EvaluatorEvents)

            image_reconstruction_evaluator = ImageReconstructionEvaluator(
                output_file, self._tb_logger.writer, overwrite=True
            )
            image_reconstruction_evaluator.attach(
                prediction_engine, name=EvaluatorKeys.IMAGE_RECONSTRUCTION, event=Events.ITERATION_COMPLETED
            )

            # add a handler for computing attribution maps
            prediction_engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                DataSaverHandler(
                    output_file,
                    keys_to_save=[],
                    attached_evaluators={EvaluatorKeys.IMAGE_RECONSTRUCTION: image_reconstruction_evaluator},
                ),
            )

            prediction_engine.run(self._data_loader)

            self._tb_logger.close()
