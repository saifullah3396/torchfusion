import math
from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Optional

import ignite.distributed as idist
import torch
import torchvision

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.constructors.diffusers import DiffusersModelConstructor
from torchfusion.core.models.constructors.factory import ModelConstructorFactory
from torchfusion.core.models.constructors.fusion import FusionModelConstructor
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.generation.aes.base import FusionModelForAutoEncoding
from torchfusion.core.models.generation.vaes.base import FusionModelForVariationalAutoEncoding
from torchfusion.core.models.utilities.data_collators import BatchToTensorDataCollator
from torchfusion.core.training.utilities.constants import TrainingStage


class FusionModelForVariationalImageAutoEncoding(FusionModelForVariationalAutoEncoding):
    @dataclass
    class Config(FusionModelForVariationalAutoEncoding.Config):
        visualized_batches: int = 1
        unnormalize: bool = False
        kl_weight: float = 0.000001

    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        return batch[DataKeys.IMAGE]

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
            f"Model constructor must be of type {(
                FusionModelConstructor,
                DiffusersModelConstructor,
            )}. "
            f"Got {type(model_constructor)}"
        )
        return model_constructor(checkpoint=checkpoint, strict=strict)

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

        # save images
        generated_samples = self.torch_model.decode(torch.randn_like(posterior.sample())).sample

        # step
        global_step = (
            training_engine.state.iteration if training_engine is not None else 1
        )
        if (
            idist.get_rank() == 0
            and engine.state.iteration <= self.config.visualized_batches
        ):
            # this only saves first batch always if you want you can shuffle validation set and save random batches
            self._logger.info(
                f"Saving image batch {engine.state.iteration} to tensorboard"
            )
            if self.config.unnormalize:
                image = image / 2 + 0.5
                generated_samples = generated_samples / 2 + 0.5
                reconstruction = reconstruction / 2 + 0.5

            # save images to tensorboard
            num_samples = batch[self.config.image_key].shape[0]
            tb_logger.writer.add_image(
                f"visualization/image",
                torchvision.utils.make_grid(
                    image.clamp(0, 1), nrow=int(math.sqrt(num_samples))
                ),
                global_step,
            )
            tb_logger.writer.add_image(
                f"visualization/generation",
                torchvision.utils.make_grid(
                    generated_samples.clamp(0, 1),
                    nrow=int(math.sqrt(num_samples)),
                ),
                global_step,
            )
            tb_logger.writer.add_image(
                f"visualization/reconstruction",
                torchvision.utils.make_grid(
                    reconstruction.clamp(0, 1),
                    nrow=int(math.sqrt(num_samples)),
                ),
                global_step,  # this is iteration of the training engine1
            )

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
