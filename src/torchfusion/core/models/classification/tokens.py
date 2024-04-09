from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from torchfusion.core.constants import DataKeys, MetricKeys
from torchfusion.core.data.args.data_args import DataArguments
from torchfusion.core.data.text_utils.data_collators import SequenceDataCollator
from torchfusion.core.data.utilities.containers import CollateFnDict, MetricsDict
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.core.models.classification.base import (
    BaseFusionNNModelForClassification,
)
from torchfusion.core.training.args.training import TrainingArguments
from torchfusion.core.training.metrics.seqeval import create_seqeval_metric
from torchfusion.core.training.utilities.constants import TrainingStage


class FusionNNModelForTokenClassification(BaseFusionNNModelForClassification):
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

    def init_metrics(self):
        def output_transform(output):
            return output[DataKeys.LOGITS].argmax(dim=2), output[self._label_key]

        metrics = {
            MetricKeys.ACCURACY: lambda: create_seqeval_metric(
                self.labels,
                fn="accuracy",
                output_transform=output_transform,
            ),
            MetricKeys.PRECISION: lambda: create_seqeval_metric(
                self.labels,
                fn="precision",
                output_transform=output_transform,
            ),
            MetricKeys.RECALL: lambda: create_seqeval_metric(
                self.labels,
                fn="recall",
                output_transform=output_transform,
            ),
            MetricKeys.F1: lambda: create_seqeval_metric(
                self.labels,
                fn="f1",
                output_transform=output_transform,
            ),
        }

        return MetricsDict(
            train=metrics, validation=metrics, test=metrics, predict=metrics
        )

    @abstractmethod
    def _build_classification_model(self):
        pass

    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        inputs = dict(
            input_ids=batch[DataKeys.TOKEN_IDS],
            attention_mask=batch[DataKeys.ATTENTION_MASKS],
        )
        if self.config.use_image:
            inputs["pixel_values"] = batch[DataKeys.IMAGE]
        if self.config.use_bbox:
            inputs["bbox"] = batch[DataKeys.TOKEN_BBOXES]

        return inputs

    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        return batch[self._label_key]

    def _build_model(self):
        self.model = self._build_classification_model()

    def training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        assert self._label_key in batch, "Label must be passed for training"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        hf_output = self({**input, "labels": label})

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOSS: hf_output.loss,
                DataKeys.LOGITS: hf_output.logits,
                self._label_key: label,
            }
        else:
            return (hf_output.loss, hf_output.logits, label)

    def evaluation_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> None:
        assert self._label_key in batch, "Label must be passed for evaluation"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        hf_output = self({**input, "labels": label})

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOSS: hf_output.loss,
                DataKeys.LOGITS: hf_output.logits,
                self._label_key: label,
            }
        else:
            return (hf_output.loss, hf_output.logits, label)

    def predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        hf_output = self(input)

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOGITS: hf_output.logits,
            }
        else:
            return (hf_output.logits,)

    def forward(self, input):
        if isinstance(input, dict):
            return self.model(**input)
        else:
            return self.model(input)

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
