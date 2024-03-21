"""
Defines the optimizer arguments
"""
from dataclasses import dataclass, field
from typing import List

from torchfusion.core.training.optim.constants import OptimizerType


@dataclass
class GroupParameters:
    group_name: str
    kwargs: dict


@dataclass
class OptimizerArguments:
    name: str = OptimizerType.ADAM.value
    group_params: List[GroupParameters] = field(
        default_factory=lambda: [
            GroupParameters(
                group_name="default",
                kwargs={
                    "lr": 2e-5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 1e-2,
                },
            )
        ],
    )

    def __post_init__(self):
        self.name = OptimizerType(self.name)
