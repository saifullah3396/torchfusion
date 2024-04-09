from __future__ import annotations

from dataclasses import dataclass, field

import torch
from dacite import Optional

from torchfusion.core.models.constructors.base import ModelConstructor
from torchfusion.core.models.factory import ModelFactory


@dataclass
class FusionModelConstructor(ModelConstructor):
    def __post_init__(self):
        super().__post_init__()
        assert self.model_task in [
            "image_classification",
        ], f"Task {self.model_task} not supported for FusionModelConstructor."

    def _init_model(self, num_labels: int) -> torch.Any:
        model_class = ModelFactory.get_torch_model_class(
            self.model_name, self.model_task
        )
        return model_class(**self.init_args, num_labels=num_labels)
