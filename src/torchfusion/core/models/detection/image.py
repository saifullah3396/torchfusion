from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.constructors.detectron2 import Detectron2ModelConstructor
from torchfusion.core.models.constructors.factory import ModelConstructorFactory
from torchfusion.core.models.detection.base import FusionModelForObjectDetection
from torchfusion.core.models.utilities.data_collators import PassThroughCollator
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


class FusionModelForImageObjectDetection(FusionModelForObjectDetection):
    @dataclass
    class Config(FusionModelForObjectDetection.Config):
        pass

    def _build_detection_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        model_constructor: Optional[dict] = None,
        model_constructor_args: Optional[dict] = None,
        num_labels: Optional[int] = None,
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

        return model_constructor(
            checkpoint=checkpoint, strict=strict, num_labels=num_labels
        )

    @abstractmethod
    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        return batch[DataKeys.IMAGE]

    @abstractmethod
    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        return batch[DataKeys.GT_INSTANCES]

    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        from torchfusion.core.data.utilities.containers import CollateFnDict

        collate_fn = PassThroughCollator(return_batch_dict=False)

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)
