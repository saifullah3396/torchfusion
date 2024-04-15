from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from torchfusion.core.models.constructors.base import ModelConstructor


@dataclass
class TorchvisionModelConstructor(ModelConstructor):
    def __post_init__(self):
        assert self.model_task in [
            "image_classification",
        ], f"Task {self.model_task} not supported for TorchvisionModelConstructor."

    def _init_model(
        self,
        **kwargs,
    ) -> torch.Any:
        os.environ["TORCH_HOME"] = self.cache_dir
        model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            self.model_name,
            pretrained=self.pretrained,
            # cache_dir=self.cache_dir, # they don't take cache_dir here
            verbose=False,
            **self.init_args,
        )

        if "num_labels" in kwargs:
            num_labels = kwargs["num_labels"]
            if self.model_name == "alexnet":
                model.classifier[6] = torch.nn.Linear(
                    model.classifier[6].in_features, num_labels
                )
            elif self.model_name == "vgg16":
                model.classifier[6] = torch.nn.Linear(
                    model.classifier[6].in_features, num_labels
                )
            elif self.model_name == "resnet50":
                model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
            elif self.model_name == "inception_v3":
                model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
            elif self.model_name == "googlenet":
                model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
            else:
                raise ValueError(
                    f"No classification head replacer defined for the model {self.model_name}."
                )

        return model
