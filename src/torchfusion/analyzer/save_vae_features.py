"""
Defines the feature attribution generation task.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import ignite.distributed as idist
import torch
from datadings.writer import FileWriter
from ignite.engine import Engine, Events

from torchfusion.core.analyzer.tasks.base import AnalyzerTask
from torchfusion.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.utilities.data_collators import BatchToTensorDataCollator
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.general import initialize_torch, setup_logging
from torchfusion.core.training.utilities.tb_logger import FusionTensorboardLogger
from torchfusion.utilities.logging import get_logger

EVENT_KEY = "vae_features_computed"
EVALUATOR_KEY = "vae_features"


class ImageVAEFeaturesSaver:
    def __init__(
        self,
        output_file: FileWriter,
        ignore_keys: Optional[Union[str, list[str]]] = None,
    ):
        self._writer = FileWriter(output_file, overwrite=True)
        self._output_transform = lambda x: (x[DataKeys.RECONS], x[DataKeys.POSTERIOR])
        self._ignore_keys = ignore_keys
        self._logger = get_logger()
        self._warned = False
        self._show_sample_info = True

    def __call__(self, engine: Engine) -> None:
        batch = engine.state.batch
        if not self._warned:
            self._logger.warning(
                f"Ignoring keys: {self._ignore_keys} in bathc with keys = {engine.state.batch.keys()}"
            )
            self._warned = True
        recons, posterior = self._output_transform(engine.state.output)

        # debugging for image reconstruction output
        # for image, recon in zip(batch["image"], recons):
        #     print(f"Image shape: {image.shape}, Reconstructed shape: {recon.shape}")
        #     import matplotlib.pyplot as plt

        #     fig, ax = plt.subplots(1, 2)
        #     ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
        #     ax[1].imshow(recon.permute(1, 2, 0).cpu().numpy())
        #     plt.show()
        #     exit()

        latents = posterior.sample()
        samples = [dict(zip(batch, t)) for t in zip(*batch.values())]
        for sample, latent in zip(samples, latents):
            sample[DataKeys.IMAGE] = latent.detach().cpu().numpy()
            for key in self._ignore_keys:
                sample.pop(key, None)

            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.detach().cpu().numpy()
                    if sample[key].shape == ():
                        sample[key] = value.item()
                else:
                    sample[key] = value

            if self._show_sample_info:
                self._logger.info("Writing features dataset with following samples:")
                for key, value in sample.items():
                    self._logger.info(f"{key}: {value}")
                self._show_sample_info = False

            if DataKeys.IMAGE_FILE_PATH in sample:
                sample["key"] = sample[DataKeys.IMAGE_FILE_PATH]
            else:
                sample["key"] = sample[DataKeys.INDEX]
            self._writer.write(sample)

    def close(self):
        self._writer.close()


class ImageVAEReconstructionSaver:
    def __init__(
        self,
        output_file: FileWriter,
        ignore_keys: Optional[Union[str, list[str]]] = None,
    ):
        self._writer = FileWriter(output_file, overwrite=False)
        self._output_transform = lambda x: x[DataKeys.RECONS]
        self._ignore_keys = ignore_keys
        self._logger = get_logger()
        self._warned = False

    def __call__(self, engine: Engine, name: str) -> None:
        batch = engine.state.batch
        if not self._warned:
            self._logger.warning(
                f"Ignoring keys: {self._ignore_keys} in batch with keys = {engine.state.batch.keys()}"
            )
            self._warned = True
        reconstructed = self._output_transform(engine.state.output)
        samples = [dict(zip(batch, t)) for t in zip(*batch.values())]
        for sample, latent in zip(samples, reconstructed):
            sample[DataKeys.IMAGE] = latent.detach().cpu().numpy()
            for key in self._ignore_keys:
                sample.pop(key, None)

            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.detach().cpu().numpy()
                    if sample[key].shape == ():
                        sample[key] = value.item()
                else:
                    sample[key] = value
            self._writer.write(sample)

    def close(self):
        self._writer.close()


class SaveVAEFeatures(AnalyzerTask):
    @dataclass
    class Config(AnalyzerTaskConfig):
        save_type: str = "features"

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
            output_dir=Path(self._hydra_config.runtime.output_dir)
        )

    def run(self):
        logger = get_logger()

        # get data collator required for the model
        stage = TrainingStage.get(self._config.data_split)

        if self._args.analyzer_args.model_checkpoints is None:
            self._args.analyzer_args.model_checkpoints = [
                (self._args.model_args.name, None)
            ]

        data_saver_class = (
            ImageVAEFeaturesSaver
            if self._config.save_type == "features"
            else ImageVAEReconstructionSaver
        )

        # remove metrics
        self._args.training_args.metric_args = []

        for model_name, checkpoint in self._args.analyzer_args.model_checkpoints:
            # set output file
            if idist.get_world_size() > 1:
                output_file = (
                    self._output_dir / f"{stage}_features_{idist.get_rank()}.msgpack"
                )
            else:
                output_file = self._output_dir / f"{stage}_features.msgpack"

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

            # model._nn_model = model._nn_model.module.module
            logger.info(f"Writing output to file: {output_file}")
            collate_fns = model.get_data_collators()
            collate_fns = CollateFnDict(
                train=BatchToTensorDataCollator(
                    data_key_type_map={DataKeys.IMAGE: torch.float},
                    allow_unmapped_data=True,
                ),
                validation=BatchToTensorDataCollator(
                    data_key_type_map={DataKeys.IMAGE: torch.float},
                    allow_unmapped_data=True,
                ),
                test=BatchToTensorDataCollator(
                    data_key_type_map={DataKeys.IMAGE: torch.float},
                    allow_unmapped_data=True,
                ),
            )
            dataloader = self.setup_dataloader(collate_fns)

            prediction_engine: Engine = self._setup_prediction_engine(
                # since we allow unmapped keys, we need to only allow image to be converted to tensor here
                model,
                keys_to_device=["image"],
            )
            image_vae_features_evaluator = data_saver_class(
                output_file, ignore_keys=[DataKeys.HOCR]
            )
            prediction_engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                image_vae_features_evaluator,
            )

            # run the prediction engine on the data
            prediction_engine.run(dataloader)

            image_vae_features_evaluator.close()
            self._tb_logger.close()
