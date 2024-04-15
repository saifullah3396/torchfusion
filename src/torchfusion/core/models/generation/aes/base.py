from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Optional

import torch

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.utilities.constants import TrainingStage


class FusionModelForAutoEncoding(FusionModel):
    @dataclass
    class Config(FusionModel.Config):
        loss: str = "l2"

    @abstractmethod
    def _build_autoencoder(
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
        model = self._build_autoencoder(
            checkpoint=checkpoint,
            strict=strict,
            model_constructor=self.config.model_constructor,
            model_constructor_args=self.config.model_constructor_args,
        )

        # initialize classification related stuff
        if self.config.loss == "l2":
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError()

        return model

    def _training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        reconstruction = self._model_forward(input)

        # compute loss
        loss = self.loss(input=reconstruction, target=input)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.RECONS: reconstruction,
            }
        else:
            return (
                loss,
                reconstruction,
            )

    def _evaluation_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        reconstruction = self._model_forward(input)

        # compute loss
        loss = self.loss(input=reconstruction, target=input)

        # visualize if required
        self._visualize(
            input=input,
            reconstruction=reconstruction,
            engine=engine,
            training_engine=training_engine,
            batch=batch,
            tb_logger=tb_logger,
            stage=stage,
            **kwargs,
        )

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.RECONS: reconstruction,
            }
        else:
            return (
                loss,
                reconstruction,
            )

    def _predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        reconstruction = self._model_forward(input)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.RECONS: reconstruction,
            }
        else:
            return (reconstruction,)

    def _visualization_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        **kwargs,
    ) -> None:
        pass

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
    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        pass
