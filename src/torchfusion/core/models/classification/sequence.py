from __future__ import annotations

from dataclasses import dataclass

import torch
from dacite import Optional
from matplotlib.transforms import Transform
from transformers import AutoConfig, AutoModelForSequenceClassification

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.text_utils.data_collators import SequenceDataCollator
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.classification.base import FusionModelForClassification
from torchfusion.core.models.constructors.factory import ModelConstructorFactory
from torchfusion.core.models.constructors.transformers import (
    TransformersModelConstructor,
)
from torchfusion.models.utilities import (
    find_layer_in_model,
    freeze_layers,
    freeze_layers_by_name,
)


class FusionModelForSequenceClassification(FusionModelForClassification):
    _SUPPORTS_CUTMIX = True
    _SUPPORTS_KD = True
    _LABEL_KEY = DataKeys.LABEL

    @dataclass
    class Config(FusionModelForClassification.Config):
        use_bbox: bool = True
        use_image: bool = True

    def _build_classification_model(
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
            (TransformersModelConstructor),
        ), (
            f"Model constructor must be of type TransformersModelConstructor. "
            f"Got {type(model_constructor)}"
        )
        return model_constructor(
            checkpoint=checkpoint, strict=strict, num_labels=num_labels
        )

    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        inputs = dict(
            input_ids=batch[DataKeys.TOKEN_IDS],
            attention_mask=batch[DataKeys.ATTENTION_MASKS],
            labels=batch[self._LABEL_KEY],
        )

        if self.config.use_image:
            inputs["pixel_values"] = batch[DataKeys.IMAGE]
        if self.config.use_bbox:
            inputs["bbox"] = batch[DataKeys.TOKEN_BBOXES]

        return inputs

    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        return batch[self._LABEL_KEY]

    def get_data_collators(self, data_key_type_map=None) -> CollateFnDict:
        collate_fn_class = SequenceDataCollator
        if data_key_type_map is None:
            data_key_type_map = {
                DataKeys.TOKEN_IDS: torch.long,
                DataKeys.TOKEN_TYPE_IDS: torch.long,
                DataKeys.ATTENTION_MASKS: torch.long,
                self._LABEL_KEY: torch.long,
            }
        else:
            data_key_type_map[DataKeys.TOKEN_IDS] = torch.long
            data_key_type_map[DataKeys.TOKEN_TYPE_IDS] = torch.long
            data_key_type_map[DataKeys.ATTENTION_MASKS] = torch.long
            data_key_type_map[self._LABEL_KEY] = torch.long

        if self.model_args.config.use_bbox:
            data_key_type_map[DataKeys.TOKEN_BBOXES] = torch.long

        if self.model_args.config.use_image:
            data_key_type_map[DataKeys.IMAGE] = torch.float

        collate_fn = collate_fn_class(
            data_key_type_map=data_key_type_map,
        )

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)
