from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.core.models.fusion_nn_model import FusionNNModel

if TYPE_CHECKING:
    from torchfusion.core.data.args.data_args import DataArguments
    from torchfusion.core.training.args.training import TrainingArguments
    from torchfusion.core.data.utilities.containers import CollateFnDict

from torchfusion.core.constants import DataKeys, MetricKeys
from torchfusion.core.training.utilities.constants import TrainingStage


class FusionNNModelForImageClassification(FusionNNModel):
    @dataclass
    class Config(FusionModelConfig):
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
            **kwargs,
        )

        self._logger.info(
            "Initialized the model with data labels: {}".format(
                self._dataset_features[DataKeys.LABEL].names
            )
        )

    @property
    def labels(self):
        return self._dataset_features[DataKeys.LABEL].names

    @property
    def num_labels(self):
        return len(self.labels)

    def init_metrics(self):
        from ignite.metrics import Accuracy

        def acc_output_transform(output):
            return output[DataKeys.LOGITS], output[DataKeys.LABEL]

        return {
            MetricKeys.ACCURACY: lambda: Accuracy(
                output_transform=acc_output_transform,
            )
        }

    def _build_model(self):
        import torch
        from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

        self.model = self._build_classification_model()
        self.loss_fn_train = torch.nn.CrossEntropyLoss()
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

        # setup mixup function
        if self.training_args.cutmixup_args is not None:
            self.mixup_fn = self.training_args.cutmixup_args.get_fn(
                num_classes=self.num_labels, smoothing=self.training_args.smoothing
            )

        # setup loss accordingly if mixup, or label smoothing is required
        if self.mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.loss_fn_train = SoftTargetCrossEntropy()
        elif self.training_args.smoothing > 0.0:
            self.loss_fn_train = LabelSmoothingCrossEntropy(
                smoothing=self.training_args.smoothing
            )
        else:
            self.loss_fn_train = torch.nn.CrossEntropyLoss()
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

    @abstractmethod
    def _build_classification_model(self):
        pass

    def training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        assert DataKeys.LABEL in batch, "Label must be passed for training"

        # get data
        image, label = batch[DataKeys.IMAGE], batch[DataKeys.LABEL]

        # apply mixup if required
        if self.mixup_fn is not None:
            image, label = self.mixup_fn(image, label)

        # compute logits
        logits = self(image)

        # compute loss
        loss = self.loss_fn_train(logits, label)

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                DataKeys.LABEL: label,
            }
        else:
            return (loss, logits, label)

    def evaluation_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> None:
        assert DataKeys.LABEL in batch, "Label must be passed for evaluation"

        # get data
        image, label = batch[DataKeys.IMAGE], batch[DataKeys.LABEL]

        # compute logits
        logits = self(image)

        # compute loss
        loss = self.loss_fn_eval(logits, label)

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                DataKeys.LABEL: label,
            }
        else:
            return (loss, logits, label)

    def predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        image = batch[DataKeys.IMAGE]

        # compute logits
        logits = self(image)

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOGITS: logits,
            }
        else:
            return (logits,)

    def forward(self, image):
        # compute logits
        output = self.model(image)
        if isinstance(output, torch.Tensor):  # usually timm returns a tensor directly
            return output
        elif is_dataclass(output):  # usually huggingface returns a dataclass
            return getattr(output, "logits")

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
