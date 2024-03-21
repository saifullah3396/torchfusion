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
            "choices": [e for e in ModuleLazyImporter.get_models().keys()],
        },
    )
    model_task: str = field(
        default="image_classification",
        metadata={"help": "Training task for which the model is loaded."},
    )
    cache_dir: str = field(
        default=os.environ.get("TORCH_FUSION_CACHE_DIR", "./cache/") + "/pretrained/",
        metadata={"help": "The location to store pretrained or cached models."},
    )
    pretrained_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint file name to load the model weights from."},
    )
    pretrained: bool = field(
        default=True,
        metadata={"help": ("Whether to load the model weights if available.")},
    )
    checkpoint_state_dict_key: str = field(
        default="state_dict",
        metadata={"help": "The state dict key for checkpoint"},
    )
    config: dict = field(
        default_factory=lambda: {},
        metadata={"help": "The model configuration."},
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

    def __post_init__(self):
        from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
        from torchfusion.core.models.factory import ModelFactory
        from torchfusion.utilities.dataclasses.dacite_wrapper import from_dict

        # if model directory is none set it to name
        if self.model_directory_name is None:
            self.model_directory_name = self.name

        # update config
        model_class = ModelFactory.get_fusion_nn_model_class(self)
        config_class = FusionModelConfig
        if hasattr(model_class, "Config"):
            config_class = model_class.Config
            if not issubclass(model_class.Config, FusionModelConfig):
                raise ValueError(
                    f"Model configuration [{model_class.Config}] must be a "
                    f"child of the [{FusionModelConfig}] class."
                )
        if self.config is None:
            self.config = config_class()
        elif isinstance(self.config, dict):
            self.config = from_dict(
                data_class=config_class,
                data=self.config,
            )
