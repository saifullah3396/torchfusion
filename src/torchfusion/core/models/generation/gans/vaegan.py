import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import ignite.distributed as idist
import torch
import torchvision
from torch import nn

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.constructors.diffusers import DiffusersModelConstructor
from torchfusion.core.models.constructors.factory import ModelConstructorFactory
from torchfusion.core.models.constructors.fusion import FusionModelConstructor
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.generation.gans.lpips import LPIPSWithDiscriminator
from torchfusion.core.training.utilities.constants import GANStage, TrainingStage


class FusionModelForVAEGAN(FusionModel):
    @dataclass
    class Config(FusionModel.Config):
        loss: str = "l2"
        disc_start: float = 50001
        kl_weight: float = 0.000001
        disc_weight: float = 0.5
        perceptual_weight: float = 1.0

    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        image = batch[DataKeys.IMAGE]
        assert (
            image.min() >= -1 and image.max() <= 1
        ), "Image must be normalized between -1 and 1 for Autoencoder"
        return image

    def _build_autoencoder(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        model_constructor: Optional[dict] = None,
        model_constructor_args: Optional[dict] = None,
    ):
        model_constructor = ModelConstructorFactory.create(
            name=model_constructor,
            kwargs=model_constructor_args,
        )
        assert isinstance(
            model_constructor,
            (
                FusionModelConstructor,
                DiffusersModelConstructor,
            ),
        ), (
            f"Model constructor must be of type {(FusionModelConstructor,DiffusersModelConstructor,)}. "
            f"Got {type(model_constructor)}"
        )
        return model_constructor(checkpoint=checkpoint, strict=strict)

    @abstractmethod
    def get_last_layer(self):
        return self.torch_model.vae.decoder.conv_out.weight

    def get_param_groups(self):
        return {
            "gen": list(self.torch_model.vae.parameters()),
            "disc": list(self.torch_model.loss.discriminator.parameters()),
        }

    def _build_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
    ):
        vae = self._build_autoencoder(
            checkpoint=checkpoint,
            strict=strict,
            model_constructor=self.config.model_constructor,
            model_constructor_args=self.config.model_constructor_args,
        )

        # make sure the underlying model got an encode and decode method
        assert hasattr(vae, "encode")
        assert hasattr(vae, "decode")
        assert hasattr(vae, "encoder")
        assert hasattr(vae, "decoder")

        loss = LPIPSWithDiscriminator(
            disc_start=self.config.disc_start,
            kl_weight=self.config.kl_weight,
            disc_weight=self.config.disc_weight,
            disc_in_channels=vae.decoder.conv_out.out_channels,
        )

        class VAEGAN(nn.Module):
            def __init__(self, vae, loss):
                super().__init__()
                self.vae = vae
                self.loss = loss

            def forward(self, input):
                pass

        # this return model is torch_model
        return VAEGAN(vae=vae, loss=loss)

    def _compute_loss(
        self,
        input,
        reconstruction,
        posterior,
        global_step,
        gan_stage=GANStage.train_gen,
    ):
        # train teh generator or discriminator depending upon the stage
        if gan_stage == GANStage.train_gen:
            loss, logged_outputs = self.torch_model.loss(
                inputs=input,
                reconstructions=reconstruction,
                posteriors=posterior,
                optimizer_idx=0,
                global_step=global_step,
                last_layer=self.get_last_layer(),
            )

        elif gan_stage == GANStage.train_disc:
            loss, logged_outputs = self.torch_model.loss(
                inputs=input,
                reconstructions=reconstruction,
                posteriors=posterior,
                optimizer_idx=1,
                global_step=global_step,
                last_layer=self.get_last_layer(),
            )

        return loss, logged_outputs

    def _training_step(
        self, engine, batch, tb_logger, gan_stage=GANStage.train_gen, **kwargs
    ) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        reconstruction, posterior = self._model_forward(input)

        # compute loss
        loss, logged_outputs = self._compute_loss(
            input,
            reconstruction,
            posterior,
            global_step=engine.state.iteration,
            gan_stage=gan_stage,
        )

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                **logged_outputs,
                DataKeys.RECONS: reconstruction,
                DataKeys.IMAGE: input,
            }
        else:
            return (
                loss,
                *list(logged_outputs.values()),
                reconstruction,
                input,
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
        _, log_dict_gen = self._compute_loss(
            input,
            reconstruction,
            posterior,
            global_step=(
                training_engine.state.iteration if training_engine is not None else 1
            ),
            gan_stage=GANStage.train_gen,
        )
        _, log_dict_disc = self._compute_loss(
            input,
            reconstruction,
            posterior,
            global_step=(
                training_engine.state.iteration if training_engine is not None else 1
            ),
            gan_stage=GANStage.train_disc,
        )

        # create output
        output = dict(**log_dict_gen, **log_dict_disc)

        # return outputs
        if self.model_args.return_dict:
            return {
                **output,
                DataKeys.RECONS: reconstruction,
                DataKeys.POSTERIOR: posterior,
                DataKeys.IMAGE: input,
            }
        else:
            return (
                output["ae_loss"],
                reconstruction,
                posterior,
                input,
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
                DataKeys.IMAGE: input,
            }
        else:
            return (
                reconstruction,
                posterior,
                input,
            )

    def _visualization_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        **kwargs,
    ) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        reconstruction, posterior = self._model_forward(input)

        # visualize
        global_step = (
            training_engine.state.iteration if training_engine is not None else 1
        )

        # step
        rank = idist.get_rank()
        if rank == 0 and engine.state.iteration <= self.config.visualized_batches:
            # save images
            generated_samples = self.vae.decode(
                torch.randn_like(posterior.sample())
            ).sample

            # this only saves first batch always if you want you can shuffle validation set and save random batches
            self._logger.info(
                f"Saving image batch {engine.state.iteration} to tensorboard"
            )

            # save images to tensorboard
            num_samples = batch[self.config.image_key].shape[0]
            tb_logger.writer.add_image(
                f"visualization/{self.config.image_key}_{rank}",
                torchvision.utils.make_grid(
                    (input / 2 + 0.5).clamp(0, 1), nrow=int(math.sqrt(num_samples))
                ),
                global_step,
            )
            tb_logger.writer.add_image(
                f"visualization/{DataKeys.GEN_SAMPLES}_{rank}",
                torchvision.utils.make_grid(
                    (generated_samples / 2 + 0.5).clamp(0, 1),
                    nrow=int(math.sqrt(num_samples)),
                ),
                global_step,
            )
            tb_logger.writer.add_image(
                f"visualization/{DataKeys.IMAGE_RECONS}_{rank}",
                torchvision.utils.make_grid(
                    (reconstruction / 2 + 0.5).clamp(0, 1),
                    nrow=int(math.sqrt(num_samples)),
                ),
                global_step,  # this is iteration of the training engine1
            )

        return {}

    def _model_forward(
        self,
        input: torch.Tensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        if isinstance(input, dict):
            # compute logits
            posterior = self.torch_model.vae.encode(**input).latent_dist
        else:
            posterior = self.torch_model.vae.encode(input).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        reconstruction = self.torch_model.vae.decode(z).sample
        return reconstruction, posterior

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
            }
        else:
            data_key_type_map[DataKeys.IMAGE] = torch.float

        collate_fn = BatchToTensorDataCollator(
            data_key_type_map=data_key_type_map,
        )

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)
