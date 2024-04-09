""" Base Model class for the any model from Timm Repository. """

from dataclasses import dataclass, field

import timm

from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.classification.image import (
    FusionNNModelForImageClassification,
)


class TimmModelForImageClassification(FusionNNModelForImageClassification):
    @dataclass
    class Config(FusionNNModelForImageClassification.Config):
        timm_name: str = "alexnet"
        timm_kwargs: dict = field(default_factory=lambda: {})

    def _build_classification_model(self):
        return timm.create_model(
            self.config.timm_name,
            pretrained=self.model_args.pretrained,
            **self.config.timm_kwargs,
            num_classes=self.num_labels,
        )
