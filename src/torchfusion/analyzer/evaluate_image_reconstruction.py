"""
Defines the feature attribution generation task.
"""

from dataclasses import dataclass

import torch
from ignite.engine import Events
from torch.utils.data import Subset

from torchfusion.analyzer.evaluators.data_saver import DataSaverHandler
from torchfusion.analyzer.evaluators.image_reconstruction_evaluator import (
    ImageReconstructionEvaluator,
)
from torchfusion.core.analyzer.tasks.base import AnalyzerTask
from torchfusion.core.constants import DataKeys
from torchfusion.core.training.utilities.tb_logger import FusionTensorboardLogger
from torchfusion.core.utilities.logging import get_logger


class EvaluateImageReconstruction(AnalyzerTask):
    @dataclass
    class Config(AnalyzerTask.Config):
        pass

    def run(self):
        logger = get_logger()

        # if there is no checkpoint provided just use the preloaded model
        if self._args.analyzer_args.model_checkpoints is None:
            self._args.analyzer_args.model_checkpoints = [
                (self._args.model_args.name, None)
            ]

        for model_name, checkpoint in self._args.analyzer_args.model_checkpoints:
            # set output file
            output_file = self._output_dir / f"{model_name}_reconstructions.h5"

            logger.info(f"Running analysis on model [{model_name}]")
            self._tb_logger = FusionTensorboardLogger(self._output_dir / model_name)

            # setup model
            model = self._setup_model(
                summarize=True,
                setup_for_train=False,
                dataset_features=self._get_dataset_info().features,
                checkpoint=checkpoint,
                strict=False,
            )

            # for analysis we want additional keys passed for data collation
            data_key_type_map = {
                DataKeys.INDEX: torch.long,
                DataKeys.IMAGE_FILE_PATH: str,
                DataKeys.IMAGE: torch.float,
            }

            # now assign collate fns
            collate_fns = model.get_data_collators(data_key_type_map=data_key_type_map)
            data_loader = self.setup_dataloader(collate_fns)

            # setup prediction engine
            prediction_engine = self._setup_prediction_engine(model)

            # setup evaluator
            image_reconstruction_evaluator = ImageReconstructionEvaluator(
                output_file, self._tb_logger.writer, overwrite=True
            )
            image_reconstruction_evaluator.attach(
                prediction_engine,
                event=Events.ITERATION_COMPLETED,
            )

            # add a handler for saving data to file
            prediction_engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                DataSaverHandler(
                    output_file,
                    keys_to_save=[],
                    attached_evaluators=[image_reconstruction_evaluator],
                ),
            )

            prediction_engine.run(data_loader)

            self._tb_logger.close()
