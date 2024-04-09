from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.classification.base import FusionModelForClassification
from torchfusion.core.models.constructors.base import ModelConstructor
from torchfusion.core.models.constructors.factory import ModelConstructorFactory
from torchfusion.core.models.constructors.fusion import FusionModelConstructor
from torchfusion.core.models.constructors.timm import TimmModelConstructor
from torchfusion.core.models.constructors.torchvision import TorchvisionModelConstructor
from torchfusion.core.models.constructors.transformers import (
    TransformersModelConstructor,
)


class FusionModelForImageClassification(FusionModelForClassification):
    _SUPPORTS_CUTMIX = True
    _SUPPORTS_KD = True
    _LABEL_KEY = DataKeys.LABEL

    @dataclass
    class Config(FusionModelForClassification.Config):
        pass

    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        return batch[DataKeys.IMAGE]

    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        return batch[self._LABEL_KEY]

    def _build_classification_model(
        self, checkpoint: Optional[str] = None, strict: bool = False
    ):
        model_constructor = ModelConstructorFactory.create(
            name=self.config.model_constructor,
            kwargs=self.config.model_constructor_args,
        )
        assert isinstance(
            model_constructor,
            (
                FusionModelConstructor,
                TorchvisionModelConstructor,
                TransformersModelConstructor,
                TimmModelConstructor,
            ),
        ), (
            f"Model constructor must be of type TransformersModelConstructor. "
            f"Got {type(model_constructor)}"
        )
        return model_constructor(self.num_labels)

    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        import torch

        from torchfusion.core.data.utilities.containers import CollateFnDict
        from torchfusion.core.models.utilities.data_collators import (
            BatchToTensorDataCollator,
        )

        if data_key_type_map is None:
            data_key_type_map = {
                DataKeys.IMAGE: torch.float,
                self._LABEL_KEY: torch.long,
            }
        else:
            data_key_type_map[DataKeys.IMAGE] = torch.float
            data_key_type_map[self._LABEL_KEY] = torch.long

        collate_fn = BatchToTensorDataCollator(
            data_key_type_map=data_key_type_map,
        )

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)
