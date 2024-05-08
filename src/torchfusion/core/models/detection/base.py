import dataclasses
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


class FusionModelForObjectDetection(FusionModel):
    @dataclass
    class Config(FusionModel.Config):
        pass

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

    @abstractmethod
    def _build_detection_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        model_constructor: Optional[dict] = None,
        model_constructor_args: Optional[dict] = None,
    ):
        pass

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

    def _model_forward(self, batch):
        return self.torch_model(batch)

    @abstractmethod
    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        pass
