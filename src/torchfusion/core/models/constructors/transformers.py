from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from torchfusion.core.models.constructors.base import ModelConstructor
from torchfusion.core.models.tasks import ModelTasks
from torchfusion.models.utilities import (
    find_layer_in_model,
    freeze_layers,
    freeze_layers_by_name,
)


@dataclass
class TransformersModelConstructor(ModelConstructor):
    n_frozen_encoder_layers: int = 0  # first N frozen layers in the encoder
    encoder_layer_name: str = (
        None  # for example layoutlmv3, this changes based on the model
    )
    frozen_layers: List[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        assert self.model_task in [
            ModelTasks.sequence_classification,
            ModelTasks.token_classification,
            ModelTasks.image_classification,
        ], f"Task {self.model_task} not supported for TransformersModelConstructor."

        if self.n_frozen_encoder_layers > 0:
            assert (
                self.encoder_layer_name is not None
            ), "Please provide the encoder encoder_layer_name to unfreeze last N layers of."

    def _init_model(self, **kwargs) -> torch.Any:
        if self.model_task == ModelTasks.sequence_classification:
            initializer_class = AutoModelForSequenceClassification
        elif self.model_task == ModelTasks.token_classification:
            initializer_class = AutoModelForTokenClassification
        elif self.model_task == ModelTasks.image_classification:
            initializer_class = AutoModelForImageClassification
        else:
            raise ValueError(f"Task {self.model_task} not supported.")

        assert (
            "num_labels" in kwargs
        ), "num_labels must be provided for token classification."
        num_labels = kwargs.pop("num_labels")

        if initializer_class in [
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        ]:
            # pass labels for these but for image classification we got to update it later
            kwargs = dict(
                num_labels=num_labels,
                return_dict=True,
            )

        if self.pretrained:
            hf_config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **self.init_args,
                **kwargs,
            )
            model = initializer_class.from_pretrained(
                self.model_name,
                config=hf_config,
                cache_dir=self.cache_dir,
            )
        else:
            hf_config = AutoConfig(
                self.model_name,
                cache_dir=self.cache_dir,
                **self.init_args,
                **kwargs,
            )
            model = initializer_class(
                self.model_name,
                config=hf_config,
                cache_dir=self.cache_dir,
            )

        if initializer_class == AutoModelForImageClassification:
            # reset the classifier head to match the labels
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)

        if self.n_frozen_encoder_layers > 0:
            encoder_layer = find_layer_in_model(model, self.encoder_layer_name)
            freeze_layers(encoder_layer[: self.n_frozen_encoder_layers])

        if self.frozen_layers is not None and len(self.frozen_layers) > 0:
            freeze_layers_by_name(model, self.frozen_layers)

        return model
