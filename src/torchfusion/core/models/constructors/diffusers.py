from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL

from torchfusion.core.models.constructors.base import ModelConstructor
from torchfusion.core.models.tasks import ModelTasks


@dataclass
class DiffusersModelConstructor(ModelConstructor):
    def __post_init__(self):
        assert self.model_task in [
            ModelTasks.autoencoding,
        ], f"Task {self.model_task} not supported for DiffusersModelConstructor."

    def _init_model(self) -> torch.Any:
        if self.model_task == ModelTasks.autoencoding:
            initializer_class = AutoencoderKL
        else:
            raise ValueError(f"Task {self.model_task} not supported.")

        if self.pretrained:
            model = initializer_class.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **self.init_args,
            )
        else:
            model = initializer_class(
                self.model_name,
                cache_dir=self.cache_dir,
                **self.init_args,
            )

        return model
