from __future__ import annotations

from dataclasses import dataclass, field

import torch
from dacite import Optional

from torchfusion.core.models.constructors.base import ModelConstructor
from torchfusion.core.models.factory import ModelFactory
from torchfusion.core.models.tasks import ModelTasks


@dataclass
class FusionModelConstructor(ModelConstructor):
    def __post_init__(self):
        super().__post_init__()
        assert self.model_task in [
            ModelTasks.image_classification,
            ModelTasks.token_classification,
            ModelTasks.gan,
            ModelTasks.autoencoding,
        ], f"Task {self.model_task} not supported for FusionModelConstructor."

    def _init_model(self, **kwargs) -> torch.Any:
        model_class = ModelFactory.get_torch_model_class(
            self.model_name, self.model_task
        )
        return model_class(**self.init_args, **kwargs)
