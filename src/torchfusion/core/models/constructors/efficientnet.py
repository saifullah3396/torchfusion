from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from efficientnet_pytorch import EfficientNet
from torchfusion.core.models.constructors.base import ModelConstructor
from torchfusion.core.models.tasks import ModelTasks


@dataclass
class EfficientNetModelConstructor(ModelConstructor):
    def __post_init__(self):
        assert self.model_task in [
            ModelTasks.image_classification,
        ], f"Task {self.model_task} not supported for EfficientNetModelConstructor."

    def _init_model(self, **kwargs) -> torch.Any:
        if "num_labels" in kwargs:
            kwargs = copy.deepcopy(kwargs)
            kwargs["num_classes"] = kwargs.pop("num_labels")

        if self.pretrained:
            return EfficientNet.from_pretrained(
                self.model_name,
                **kwargs,
                **self.init_args,
            )
        else:
            return EfficientNet.from_name(
                self.model_name,
                **kwargs,
                **self.init_args,
            )
