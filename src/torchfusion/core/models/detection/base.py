import dataclasses
import json
from dataclasses import dataclass
from typing import Optional

from detectron2.modeling.meta_arch import GeneralizedRCNN  # noqa
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.constructors.detectron2 import Detectron2ModelConstructor
from torchfusion.core.models.constructors.factory import ModelConstructorFactory
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.utilities.data_collators import PassThroughCollator
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


class FusionModelForObjectDetection(FusionModel):
    @dataclass
    class Config(FusionModel.Config):
        vis_period: int = 1

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        logger.info(
            "Initializing the model with the following config: {}".format(
                json.dumps(dataclasses.asdict(self.config), indent=4)
            )
        )

    def _build_detection_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        model_constructor: Optional[dict] = None,
        model_constructor_args: Optional[dict] = None,
        **kwargs,
    ):
        model_constructor = ModelConstructorFactory.create(
            name=model_constructor,
            kwargs=model_constructor_args,
        )
        assert isinstance(
            model_constructor,
            (Detectron2ModelConstructor,),
        ), (
            f"""Model constructor must be of type: {(
                Detectron2ModelConstructor
            )}. """
            f"Got {type(model_constructor)}"
        )

        return model_constructor(checkpoint=checkpoint, strict=strict, **kwargs)

    def _build_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
    ):
        # we build this mainly around detectron2 as it provides all the necessary resources
        model = self._build_detection_model(
            checkpoint=checkpoint,
            strict=strict,
            model_constructor=self.config.model_constructor,
            model_constructor_args=self.config.model_constructor_args,
            vis_period=self.config.vis_period,
        )

        return model

    def _training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        assert self._torch_model.training

        # compute logits
        losses = self._model_forward(batch)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: sum(losses.values()),
                **losses,
            }
        else:
            return list(losses.values())

    def _evaluation_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> None:
        assert not self.torch_model.training

        outputs = self._model_forward(batch)

        # convert list of dict to dict of list
        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}
        return outputs

    def _predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        assert not self.torch_model.training

        outputs = self._model_forward(batch)

        # convert list of dict to dict of list
        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}
        return outputs

    def _visualization_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        **kwargs,
    ) -> None:
        import math

        import ignite.distributed as idist
        import torchvision

        assert not self.torch_model.training

        # detectr2on ensures the bbox always remap to original image size,
        # but we don't want that during visualization so we manually update it to current image size
        for sample in batch:
            sample[DataKeys.IMAGE_HEIGHT] = sample[DataKeys.IMAGE].shape[2]
            sample[DataKeys.IMAGE_WIDTH] = sample[DataKeys.IMAGE].shape[3]

        outputs = self._model_forward(batch)

        if idist.get_rank() == 0:
            import torch
            from detectron2.utils.visualizer import ColorMode, Visualizer
            from torchvision.transforms.functional import resize

            image_batch = []
            for sample, output in zip(batch, outputs):
                image = sample[DataKeys.IMAGE].detach().cpu().permute(1, 2, 0).numpy()
                v = Visualizer(image, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
                image_output = v.draw_instance_predictions(
                    output["instances"].to("cpu")
                ).get_image()
                image_batch.append(
                    resize(
                        torch.from_numpy(image_output.transpose(2, 0, 1)), (512, 512)
                    )
                )

            logger.info(f"Saving image batch {engine.state.iteration} to tensorboard.")
            # step
            global_step = (
                training_engine.state.iteration if training_engine is not None else 1
            )
            batch_step = engine.state.iteration
            # save a single image to tensorboard
            num_samples = len(image_batch)
            tb_logger.writer.add_image(
                f"visualization/pred_instances_{batch_step}",  # engine .iteration refers to the batch id
                torchvision.utils.make_grid(
                    image_batch, nrow=int(math.sqrt(num_samples))
                ),
                global_step,
            )

    def _model_forward(self, batch):
        return self.torch_model(batch)

    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        from torchfusion.core.data.utilities.containers import CollateFnDict

        collate_fn = PassThroughCollator(return_batch_dict=False)

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)
