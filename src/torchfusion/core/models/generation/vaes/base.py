import math
from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Optional

import ignite.distributed as idist
import torch
import torchvision

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.utilities.constants import TrainingStage


class FusionModelForVariationalAutoEncoding(FusionModel):
    @dataclass
    class Config(FusionModel.Config):
        loss: str = "l2"
        rec_loss_weight: float = 1.0
        kl_loss_weight: float = 1.0

    @abstractmethod
    def _build_autoencoder(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        model_constructor: Optional[dict] = None,
        model_constructor_args: Optional[dict] = None,
    ):
        pass

    def _build_model(self, checkpoint: Optional[str] = None, strict: bool = False):
        model = self._build_autoencoder(
            checkpoint=checkpoint,
            strict=strict,
            model_constructor=self.config.model_constructor,
            model_constructor_args=self.config.model_constructor_args,
        )

        # make sure the underlying model got an encode and decode method
        assert hasattr(model, "encode")
        assert hasattr(model, "decode")

        # initialize classification related stuff
        if self.config.loss == "l2":
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError()

        return model

    def _compute_loss(self, reconstruction, posterior):
        # compute rec loss
        rec_loss = self.loss(input=reconstruction, target=input)

        # compute kl loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # compute total loss
        loss = (
            self.config.rec_loss_weight * rec_loss
            + self.config.kl_loss_weight * kl_loss
        )

        return loss, rec_loss, kl_loss

    def _training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        reconstruction, posterior = self._model_forward(input)

        # compute loss
        loss, rec_loss, kl_loss = self._compute_loss(reconstruction, posterior)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.RECONS: reconstruction,
                DataKeys.POSTERIOR: posterior,
                "rec_loss": rec_loss,
                "kl_loss": kl_loss,
            }
        else:
            return (
                loss,
                reconstruction,
                posterior,
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
        reconstruction, posterior = self._model_forward(input)

        # compute loss
        loss, rec_loss, kl_loss = self._compute_loss(reconstruction, posterior)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.RECONS: reconstruction,
                DataKeys.POSTERIOR: posterior,
                "rec_loss": rec_loss,
                "kl_loss": kl_loss,
            }
        else:
            return (
                loss,
                reconstruction,
                posterior,
            )

    def _predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        reconstruction, posterior = self._model_forward(input)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.RECONS: reconstruction,
                DataKeys.POSTERIOR: posterior,
            }
        else:
            return (
                reconstruction,
                posterior,
            )

    def _visualization_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        **kwargs,
    ) -> None:
        pass

    def _model_forward(
        self,
        input: torch.Tensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        if isinstance(input, dict):
            # compute logits
            posterior = self.torch_model.encode(**input)
        else:
            posterior = self.torch_model.encode(input)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        reconstruction = self.torch_model.decode(z).sample
        return reconstruction, posterior

    @abstractmethod
    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        pass

    @abstractmethod
    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        pass
