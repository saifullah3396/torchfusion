from __future__ import annotations

from dataclasses import dataclass

import timm
import torch

from torchfusion.core.models.constructors.base import ModelConstructor


@dataclass
class TimmModelConstructor(ModelConstructor):
    def __post_init__(self):
        assert self.model_task in [
            "image_classification",
        ], f"Task {self.model_task} not supported for TimmModelConstructor."

    def _init_model(self, **kwargs) -> torch.Any:
        return timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            # cache_dir=self.cache_dir, # does not take cache_dir
            **kwargs,
            **self.init_args,
        )
