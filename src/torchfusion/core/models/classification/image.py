from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.core.models.classification.base import (
    BaseFusionNNModelForClassification,
)
from torchfusion.core.models.fusion_nn_model import FusionNNModel

if TYPE_CHECKING:
    from torchfusion.core.data.args.data_args import DataArguments
    from torchfusion.core.training.args.training import TrainingArguments
    from torchfusion.core.data.utilities.containers import CollateFnDict

from torchfusion.core.constants import DataKeys, MetricKeys
from torchfusion.core.training.utilities.constants import TrainingStage


class FusionNNModelForImageClassification(BaseFusionNNModelForClassification):
    @dataclass
    class Config(BaseFusionNNModelForClassification.Config):
        pass

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
            supports_cutmix=True,
            **kwargs,
        )

    @abstractmethod
    def _build_classification_model(self):
        pass

    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        return batch[DataKeys.IMAGE]

    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        return batch[self._label_key]

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
                DataKeys.LABEL: torch.long,
            }
        else:
            data_key_type_map[DataKeys.IMAGE] = torch.float
            data_key_type_map[DataKeys.LABEL] = torch.long

        collate_fn = BatchToTensorDataCollator(
            data_key_type_map=data_key_type_map,
        )

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)
