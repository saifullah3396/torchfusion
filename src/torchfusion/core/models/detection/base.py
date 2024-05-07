import dataclasses
import json
from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from textwrap import indent
from typing import Optional

import torch
from datasets.features import Sequence
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.utilities.knowledge_distillation import (
    EnsembleKnowledgeTransferLoss,
    GaussianLoss,
    TemperatureScaledKLDivLoss,
)
from torchfusion.core.training.utilities.constants import TrainingStage


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
        self._logger.info(
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
        assert self._LABEL_KEY in batch, "Label must be passed for training"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        losses = self._model_forward(input)

        print("losses", losses)

        # return outputs
        if self.model_args.return_dict:
            return losses
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
        assert self._LABEL_KEY in batch, "Label must be passed for evaluation"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits = self._model_forward(input)

        # compute loss
        loss = self.loss_fn_eval(logits, label)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                self._LABEL_KEY: label,
            }
        else:
            return (loss, logits, label)

    def _predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits = self._model_forward(input)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOGITS: logits,
            }
        else:
            return (logits,)

    def _model_forward(self, input, return_logits=True):
        if isinstance(input, dict):
            # compute logits
            output = self.torch_model(**input)
        else:
            output = self.torch_model(input)

        # we assume first element is logits
        if isinstance(output, torch.Tensor):  # usually timm returns a tensor directly
            return output
        elif isinstance(output, tuple):  # usually timm returns a tensor directly
            if return_logits:
                return output[0]
            else:
                return output
        elif is_dataclass(output):  # usually huggingface returns a dataclass
            if return_logits:
                return getattr(output, "logits")
            else:
                return output

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
