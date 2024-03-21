"""
Defines the base TrainValSampler class for defining training/validation split samplers.
"""

from dataclasses import dataclass, field

from torchfusion.utilities.dataclasses.abstract_dataclass import AbstractDataclass


@dataclass
class FusionModelConfig(AbstractDataclass):
    """
    Base model configuration.
    """

    return_dict: bool = field(
        default=True,
        metadata={"help": ("Whether the outputs of the model return a dictionary.")},
    )
    required_training_functionality: str = "default"
    bypass_params_creation: bool = field(
        default=False,
        metadata={
            "help": (
                "If this is true, the the mapping of the groups with optimizers is not generated."
                "It can be used for customized parameter groups for example for lr decay in some layers."
            )
        },
    )
