"""
Defines the dataclass for holding model arguments.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from torchfusion.core.args.args_base import ArgumentsBase
from torchfusion.utilities.module_import import ModuleLazyImporter


@dataclass
class ModelArguments(ArgumentsBase):
    """
    Dataclass that holds the model arguments.
    """

    name: str = field(
        default="",
        metadata={
            "help": "The name of the model to use.",
            "choices": [e for e in ModuleLazyImporter.get_fusion_models().keys()],
        },
    )
    model_task: str = field(
        default="image_classification",
        metadata={"help": "Training task for which the model is loaded."},
    )
    required_training_functionality: str = field(
        default="default",
        metadata={"help": "The required functionality for the model."},
    )
    cache_dir: str = field(
        default=os.environ.get("TORCH_FUSION_CACHE_DIR", "./cache/") + "/pretrained/",
        metadata={"help": "The location to store pretrained or cached models."},
    )
    convert_bn_to_gn: bool = field(
        default=False,
        metadata={"help": "If true, converts all batch norm layers to group norm."},
    )
    remove_lora_layers: bool = field(
        default=False,
        metadata={"help": "If true, converts all batch norm layers to group norm."},
    )
    model_directory_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the model directory."},
    )
    bypass_params_creation: bool = field(
        default=False,
        metadata={
            "help": (
                "If this is true, the the mapping of the groups with optimizers is not generated."
                "It can be used for customized parameter groups for example for lr decay in some layers."
            )
        },
    )
    return_dict: bool = field(
        default=True,
        metadata={"help": ("Whether the outputs of the model return a dictionary.")},
    )
    model_config: dict = field(
        default_factory=lambda: {},
        metadata={"help": "The model configuration."},
    )

    def __post_init__(self):
        from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
        from torchfusion.core.models.factory import ModelFactory
        from torchfusion.utilities.dataclasses.dacite_wrapper import from_dict

        # if model directory is none set it to name
        if self.model_directory_name is None:
            self.model_directory_name = self.name

        # update config
        model_class = ModelFactory.get_fusion_model_class(self)
        config_class = FusionModelConfig
        if hasattr(model_class, "Config"):
            config_class = model_class.Config
            if not issubclass(model_class.Config, FusionModelConfig):
                raise ValueError(
                    f"Model configuration [{model_class.Config}] must be a "
                    f"child of the [{FusionModelConfig}] class."
                )

        if self.model_config is None:
            raise ValueError("Model configuration is required to initialize the model.")

        self.model_config["model_constructor_args"]["model_task"] = self.model_task
        self.model_config["model_constructor_args"]["cache_dir"] = self.cache_dir
        self.model_config = from_dict(
            data_class=config_class,
            data=self.model_config,
        )
