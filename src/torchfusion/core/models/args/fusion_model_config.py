"""
Defines the base TrainValSampler class for defining training/validation split samplers.
"""

from copyreg import constructor
from dataclasses import dataclass, field
from typing import Optional

from click import Option

from torchfusion.core.args.args_base import ArgumentsBase, ClassInitializerArgs
from torchfusion.core.models.constructors.fusion import FusionModelConstructor
from torchfusion.core.models.constructors.timm import TimmModelConstructor
from torchfusion.core.models.constructors.torchvision import TorchvisionModelConstructor
from torchfusion.core.models.constructors.transformers import (
    TransformersModelConstructor,
)
from torchfusion.utilities.dataclasses.abstract_dataclass import AbstractDataclass
from torchfusion.utilities.dataclasses.dacite_wrapper import from_dict


@dataclass
class FusionModelConfig(ArgumentsBase):
    """
    Base model configuration.
    """

    model_constructor: str = field(
        default=str,
        metadata={
            "help": "The type of initializer to use for the model. Options are 'custom', 'transformers', 'torchvision'."
        },
    )
    model_constructor_args: dict = field(
        default_factory=dict,
        metadata={"help": "The arguments for the model constructor."},
    )
