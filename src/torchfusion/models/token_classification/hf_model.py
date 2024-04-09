from dataclasses import dataclass, field
from typing import List, Optional
from uu import encode

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from torchfusion.core.models.classification.sequence import (
    FusionNNModelForSequenceClassification,
)
from torchfusion.core.models.classification.tokens import (
    FusionNNModelForTokenClassification,
)
from torchfusion.models.utilities import (
    find_layer_in_model,
    freeze_layers,
    freeze_layers_by_name,
)


class HuggingfaceModelForSequenceClassification(FusionNNModelForSequenceClassification):
    @dataclass
    class Config(FusionNNModelForSequenceClassification.Config):
        hf_name: str = ""
        hf_kwargs: dict = field(default_factory={})
        n_frozen_encoder_layers: Optional[int] = (
            None  # first N frozen layers in the encoder
        )
        encoder_layer_name: str = (
            None  # for example layoutlmv3, this changes based on the model
        )
        frozen_layers: List[str] = field(default_factory=lambda: [])

        def __post_init__(self):
            if self.n_frozen_encoder_layers > 0:
                assert (
                    self.encoder_layer_name is not None
                ), "Please provide the encoder encoder_layer_name to unfreeze last N layers of."

    def _build_classification_model(self):
        if self.model_args.pretrained:
            hf_config = AutoConfig.from_pretrained(
                self.config.hf_name,
                num_labels=self.num_labels,
                cache_dir=self.model_args.cache_dir,
                **self.config.hf_kwargs,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.hf_name,
                cache_dir=self.model_args.cache_dir,
                config=hf_config,
            )
        else:
            hf_config = AutoConfig(
                self.config.hf_name,
                num_labels=self.num_labels,
                cache_dir=self.model_args.cache_dir,
                **self.config.hf_kwargs,
            )
            model = AutoModelForSequenceClassification(
                self.config.hf_name,
                cache_dir=self.model_args.cache_dir,
                config=hf_config,
            )

        if self.config.n_frozen_encoder_layers > 0:
            encoder_layer = find_layer_in_model(model, self.config.encoder_layer_name)
            freeze_layers(encoder_layer[: self.config.n_frozen_encoder_layers])

        if self.config.frozen_layers is not None and len(self.config.frozen_layers) > 0:
            freeze_layers_by_name(model, self.config.frozen_layers)

        return model


class HuggingfaceModelForTokenClassification(FusionNNModelForTokenClassification):
    @dataclass
    class Config(FusionNNModelForTokenClassification.Config):
        hf_name: str = ""
        hf_kwargs: dict = field(default_factory={})
        n_frozen_encoder_layers: int = 0  # first N frozen layers in the encoder
        encoder_layer_name: str = (
            None  # for example layoutlmv3, this changes based on the model
        )
        frozen_layers: List[str] = field(default_factory=lambda: [])

        def __post_init__(self):
            if self.n_frozen_encoder_layers > 0:
                assert (
                    self.encoder_layer_name is not None
                ), "Please provide the encoder encoder_layer_name to unfreeze last N layers of."

    def _build_classification_model(self):
        if self.model_args.pretrained:
            hf_config = AutoConfig.from_pretrained(
                self.config.hf_name,
                num_labels=self.num_labels,
                cache_dir=self.model_args.cache_dir,
                **self.config.hf_kwargs,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.config.hf_name,
                cache_dir=self.model_args.cache_dir,
                config=hf_config,
            )
        else:
            hf_config = AutoConfig(
                self.config.hf_name,
                num_labels=self.num_labels,
                cache_dir=self.model_args.cache_dir,
                **self.config.hf_kwargs,
            )
            model = AutoModelForTokenClassification(
                self.config.hf_name,
                cache_dir=self.model_args.cache_dir,
                config=hf_config,
            )

        if self.config.n_frozen_encoder_layers > 0:
            encoder_layer = find_layer_in_model(model, self.config.encoder_layer_name)
            freeze_layers(encoder_layer[: self.config.n_frozen_encoder_layers])

        if self.config.frozen_layers is not None and len(self.config.frozen_layers) > 0:
            freeze_layers_by_name(model, self.config.frozen_layers)

        return model
