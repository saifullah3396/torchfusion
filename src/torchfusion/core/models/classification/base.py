from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Any, Optional

import torch
from datasets.features import Sequence

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.args.data_args import DataArguments
from torchfusion.core.data.utilities.containers import CollateFnDict, MetricsDict
from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.core.models.fusion_nn_model import FusionNNModel
from torchfusion.core.training.args.training import TrainingArguments
from torchfusion.core.training.utilities.constants import TrainingStage


class BaseFusionNNModelForClassification(FusionNNModel):
    @dataclass
    class Config(FusionModelConfig):
        pass

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        dataset_features: Any,
        label_key: str = DataKeys.LABEL,
        supports_cutmix: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            dataset_features=dataset_features,
            **kwargs,
        )

        self._label_key = label_key
        self._supports_cutmix = supports_cutmix

        self._logger.info(
            "Initialized the model with data labels: {}".format(self.labels)
        )

    @property
    def labels(self):
        if isinstance(self._dataset_features[self._label_key], Sequence):
            return self._dataset_features[self._label_key].feature.names
        else:
            return self._dataset_features[self._label_key].names

    @property
    def num_labels(self):
        return len(self.labels)

    def _build_model(self):
        import torch
        from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

        self.model = self._build_classification_model()
        self.loss_fn_train = torch.nn.CrossEntropyLoss()
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

        # setup mixup function
        self.mixup_fn = None
        if self._supports_cutmix and self.training_args.cutmixup_args is not None:
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

    def training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        assert self._label_key in batch, "Label must be passed for training"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits = self(input)

        # compute loss
        loss = self.loss_fn_train(logits, label)

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                self._label_key: label,
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
        assert self._label_key in batch, "Label must be passed for evaluation"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits = self(input)

        # compute loss
        loss = self.loss_fn_eval(logits, label)

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                self._label_key: label,
            }
        else:
            return (loss, logits, label)

    def predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits = self(input)

        # return outputs
        if self.config.return_dict:
            return {
                DataKeys.LOGITS: logits,
            }
        else:
            return (logits,)

    def forward(self, input):
        if isinstance(input, dict):
            # compute logits
            output = self.model(**input)
        else:
            output = self.model(input)
        if isinstance(output, torch.Tensor):  # usually timm returns a tensor directly
            return output
        elif is_dataclass(output):  # usually huggingface returns a dataclass
            return getattr(output, "logits")

    @abstractmethod
    def _build_classification_model(self):
        pass

    @abstractmethod
    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        pass

    @abstractmethod
    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        pass

    @abstractmethod
    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        pass
