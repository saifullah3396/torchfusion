""" PyTorch module that defines the base model for training/testing etc. """

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from torch import nn

from torchfusion.core.data.utilities.containers import CollateFnDict, MetricsDict
from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.utilities.logging import get_logger

if TYPE_CHECKING:
    from torchfusion.core.data.args.data_args import DataArguments
    from torchfusion.core.training.args.training import TrainingArguments

from torchfusion.utilities.logging import get_logger


class FusionNNModel(nn.Module):
    @dataclass
    class Config(FusionModelConfig):
        pass

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        dataset_features: Any,
    ):
        super().__init__()

        # initialize arguments
        self._model_args = model_args
        self._data_args = data_args
        self._training_args = training_args
        self._dataset_features = dataset_features

        # initialize logger
        self._logger = get_logger()

    @property
    def model_args(self):
        return self._model_args

    @property
    def data_args(self):
        return self._data_args

    @property
    def training_args(self):
        return self._training_args

    @property
    def config(self) -> FusionModelConfig:
        return self._model_args.config

    def build_model(self):
        self._build_model()

    @abstractmethod
    def _build_model(self):
        pass

    def _reinit_weights(self):
        pass

    def get_param_groups(self):
        return {
            "default": list(self.parameters()),
        }

    def training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        return self(
            engine=engine,
            batch=batch,
            tb_logger=tb_logger,
            stage=TrainingStage.train,
            **kwargs,
        )

    def evaluation_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> None:
        return self(
            engine=engine,
            training_engine=training_engine,
            batch=batch,
            tb_logger=tb_logger,
            stage=stage,
            **kwargs,
        )

    def predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        return self(
            engine=engine,
            batch=batch,
            tb_logger=tb_logger,
            stage=TrainingStage.predict,
            **kwargs,
        )

    def get_checkpoint_state_dict(self):
        """Return information to be saved in the checkpoint."""
        return {}

    def on_load_checkpoint(self, checkpoint, strict: bool = True):
        current_state_dict = self.state_dict()
        new_state_dict = {}
        unmatched_keys = []
        for state in checkpoint:
            current_state_name = state
            if current_state_name in current_state_dict:
                if (
                    current_state_dict[current_state_name].size()
                    == checkpoint[state].size()
                ):
                    new_state_dict[current_state_name] = checkpoint[state]
                else:
                    unmatched_keys.append(state)
        if len(unmatched_keys) > 0:
            if strict:
                raise RuntimeError(
                    f"Found keys that are in the model state dict but their sizes don't match: {unmatched_keys}"
                )
            else:
                self._logger.warning(
                    f"Found keys that are in the model state dict but their sizes don't match: {unmatched_keys}"
                )
        return self.load_state_dict(new_state_dict, strict=strict)

    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        return CollateFnDict()
