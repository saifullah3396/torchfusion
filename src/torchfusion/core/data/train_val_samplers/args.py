"""
Defines the dataclass for holding train/val sampling arguments.
"""

from dataclasses import dataclass
from typing import Optional

from torchfusion.core.args.args_base import ArgumentsBase


@dataclass
class TrainValSamplerArguments(ArgumentsBase):
    """
    Dataclass that holds the train/validation sampling arguments.
    """

    # Train validation strategy to use
    strategy: str = ""

    # Strategy config
    config: Optional[dict] = None
