""" Base Model class for the any model from Timm Repository. """

from dataclasses import dataclass, field

from torch import nn
from transformers import AutoConfig, AutoModelForImageClassification

from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.image_classification.fusion_nn_model import (
    FusionNNModelForImageClassification,
)


class HuggingfaceModelForImageClassification(FusionNNModelForImageClassification):
    @dataclass
    class Config(FusionNNModelForImageClassification.Config):
        hf_name: str = ""
        hf_kwargs: dict = field(default_factory={})

    def _build_classification_model(self):
        if self.model_args.pretrained:
            hf_config = AutoConfig.from_pretrained(
                self.config.hf_name,
                cache_dir=self.model_args.cache_dir,
                **self.config.hf_kwargs,
            )
            model = AutoModelForImageClassification.from_pretrained(
                self.config.hf_name,
                cache_dir=self.model_args.cache_dir,
                config=hf_config,
            )
        else:
            hf_config = AutoConfig(
                self.config.hf_name,
                cache_dir=self.model_args.cache_dir,
            )
            model = AutoModelForImageClassification(
                self.config.hf_name,
                cache_dir=self.model_args.cache_dir,
                config=hf_config,
            )

        model.classifier = nn.Linear(model.classifier.in_features, self.num_labels)
        return model
