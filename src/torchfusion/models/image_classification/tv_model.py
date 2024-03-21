""" Base Model class for the any model from Timm Repository. """


from dataclasses import dataclass, field

import torch

from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.image_classification.fusion_nn_model import (
    FusionNNModelForImageClassification,
)


class TorchvisionModelForImageClassification(FusionNNModelForImageClassification):
    @dataclass
    class Config(FusionModelConfig):
        tv_name: str = "alexnet"
        tv_kwargs: dict = field(default_factory=lambda: {})

    def _build_classification_model(self):
        model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            self.config.tv_name,
            pretrained=self.model_args.pretrained,
            verbose=False,
            **self.config.tv_kwargs,
        )

        if self.config.tv_name == "alexnet":
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, self.num_labels)
        elif self.config.tv_name == "vgg16":
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, self.num_labels)
        elif self.config.tv_name == "resnet50":
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_labels)
        elif self.config.tv_name == "inception_v3":
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_labels)
        elif self.config.tv_name == "googlenet":
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_labels)
        else:
            raise ValueError(f"No classification head replacer defined for the model {self.config.tv_name}.")

        return model
