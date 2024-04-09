from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from torchfusion.core.data.text_utils.data_collators import SequenceDataCollator
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.core.models.classification.base import (
    BaseFusionNNModelForClassification,
)

if TYPE_CHECKING:
    from torchfusion.core.data.args.data_args import DataArguments
    from torchfusion.core.training.args.training import TrainingArguments
    from torchfusion.core.data.utilities.containers import CollateFnDict

from torchfusion.core.constants import DataKeys


class FusionNNModelForSequenceClassification(BaseFusionNNModelForClassification):
    @dataclass
    class Config(BaseFusionNNModelForClassification.Config):
        use_bbox: bool = True
        use_image: bool = True

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        dataset_features: Any,
        **kwargs,
    ):
        super().__init__(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            dataset_features=dataset_features,
            supports_cutmix=False,
            label_key=DataKeys.LABEL,
            **kwargs,
        )

    @abstractmethod
    def _build_classification_model(self):
        pass

    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        inputs = dict(
            input_ids=batch[DataKeys.TOKEN_IDS],
            attention_mask=batch[DataKeys.ATTENTION_MASKS],
            labels=batch[self._label_key],
        )

        if self.config.use_image:
            inputs["pixel_values"] = batch[DataKeys.IMAGE]
        if self.config.use_bbox:
            inputs["bbox"] = batch[DataKeys.TOKEN_BBOXES]

        return inputs

    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        return batch[self._label_key]

    def get_data_collators(self, data_key_type_map=None) -> CollateFnDict:
        collate_fn_class = SequenceDataCollator
        if data_key_type_map is None:
            data_key_type_map = {
                DataKeys.TOKEN_IDS: torch.long,
                DataKeys.TOKEN_TYPE_IDS: torch.long,
                DataKeys.ATTENTION_MASKS: torch.long,
                self._label_key: torch.long,
            }
        else:
            data_key_type_map[DataKeys.TOKEN_IDS] = torch.long
            data_key_type_map[DataKeys.TOKEN_TYPE_IDS] = torch.long
            data_key_type_map[DataKeys.ATTENTION_MASKS] = torch.long
            data_key_type_map[self._label_key] = torch.long

        if self.model_args.config.use_bbox:
            data_key_type_map[DataKeys.TOKEN_BBOXES] = torch.long

        if self.model_args.config.use_image:
            data_key_type_map[DataKeys.IMAGE] = torch.float

        collate_fn = collate_fn_class(
            data_key_type_map=data_key_type_map,
        )

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)
