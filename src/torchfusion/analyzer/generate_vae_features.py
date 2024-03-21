"""
Defines the feature attribution generation task.
"""

from dataclasses import dataclass
import io
from typing import Optional, Type, Union
import numpy as np
import ignite.distributed as idist
import torch
from datadings.writer import FileWriter
from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList
import pickle
from torchfusion.core.analyzer.tasks.base import AnalyzerTask
from torchfusion.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.utilities.data_collators import BatchToTensorDataCollator
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.tb_logger import FusionTensorboardLogger
from torchfusion.utilities.logging import get_logger

EVENT_KEY = "vae_features_computed"
EVALUATOR_KEY = "vae_features"


class VAEFeaturesEvaluator:
    def __init__(
        self,
        dataset_path: str,
        writer: FileWriter,
    ):
        self._dataset_path = dataset_path
        self._writer = writer
        self._output_transform = lambda x: x[DataKeys.POSTERIOR]

    def compute(self, engine: Engine) -> None:
        posterior = self._output_transform(engine.state.output)
        latents = posterior.sample()
        return dict(
            latents=latents,
        )

    def __call__(self, engine: Engine, name: str) -> None:
        data = self.compute(engine)
        # import cv2
        # cv2.imwrite("test1.png", 255*(engine.state.batch['image'][0].permute(1,2,0).detach().cpu().numpy()*2+1))
        # cv2.imwrite("test2.png", 255*(engine.state.batch['image'][1].permute(1,2,0).detach().cpu().numpy()*2+1))
        # cv2.imwrite("test3.png", 255*(engine.state.batch['image'][2].permute(1,2,0).detach().cpu().numpy()*2+1))
        # cv2.imwrite("test4.png", 255*(engine.state.batch['image'][3].permute(1,2,0).detach().cpu().numpy()*2+1))
        # exit()
        for i in range(len(data["latents"])):
            output = {}
            # compressed_image = io.BytesIO()
            # np.savez_compressed(compressed_image, data["latents"][i].detach().cpu().numpy())
            # output[DataKeys.IMAGE] = compressed_image.getvalue()
            output[DataKeys.IMAGE] = data["latents"][i].detach().cpu().numpy()
            for key, value in engine.state.batch.items():
                if key in [DataKeys.IMAGE, DataKeys.HOCR]:
                    continue
                if key == DataKeys.IMAGE_FILE_PATH:
                    output["key"] = value[i].replace(self._dataset_path, "")
                    continue
                if isinstance(value[i], torch.Tensor):
                    output[key] = value[i].detach().cpu().numpy()
                    if output[key].shape == ():
                        output[key] = value[i].item()
                else:
                    output[key] = value[i]

            # result = {
            #     'key': output['key'],
            #     'data': pickle.dumps(output)
            # }
            # self._writer.write(result)
            self._writer.write(output)

    def attach(
        self,
        engine: Engine,
        name: str = "attr_map",
        event: Union[str, Events, CallableEventWithFilter, EventsList] = Events.ITERATION_COMPLETED,
    ) -> None:
        if not hasattr(engine.state, name):
            setattr(engine.state, name, None)
        engine.add_event_handler(event, self, name)


class GenerateVAEFeatures(AnalyzerTask):
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

        # get data collator required for the model
        stage = TrainingStage.get(self._config.data_split)
        collate_fns = CollateFnDict(
            train=BatchToTensorDataCollator(), validation=BatchToTensorDataCollator(), test=BatchToTensorDataCollator()
        )
        self._data_loader.collate_fn = collate_fns[stage]

        if self._args.analyzer_args.model_checkpoints is None:
            self._args.analyzer_args.model_checkpoints = [(self._args.model_args.name, None)]

        for model_name, checkpoint in self._args.analyzer_args.model_checkpoints:
            # set output file
            if idist.get_world_size() > 1:
                output_file = self._output_dir / f"{stage}_features_{idist.get_rank()}.msgpack"
            else:
                output_file = self._output_dir / f"{stage}_features.msgpack"

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

            # model._nn_model = model._nn_model.module.module
            logger.info(f"Writing output to file: {output_file}")
            with FileWriter(output_file, overwrite=True) as writer:
                prediction_engine = self._setup_prediction_engine(model, convert_to_tensor=["image"])
                prediction_engine.register_events(EVENT_KEY)

                evaluator = VAEFeaturesEvaluator(self._args.data_args.dataset_dir, writer)
                evaluator.attach(prediction_engine, name=EVALUATOR_KEY, event=Events.ITERATION_COMPLETED)
                prediction_engine.run(self._data_loader)

            self._tb_logger.close()
