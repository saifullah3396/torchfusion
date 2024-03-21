from dataclasses import dataclass, field
from typing import Optional

from torchfusion.core.args.args_base import ArgumentsBase
from torchfusion.core.args.general_args import GeneralArguments
from torchfusion.core.data.args.data_args import DataArguments
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.core.training.args.training import TrainingArguments


@dataclass
class FusionArguments(ArgumentsBase):
    """
    A aggregated container for all the arguments required for model configuration,
    data loading, and training.
    """

    # General arguments
    general_args: GeneralArguments = field(default=GeneralArguments())

    # Model related arguments
    model_args: Optional[ModelArguments] = field(default=None)

    # Data related arguments
    data_args: DataArguments = field(default=DataArguments())

    # Training related arguments
    training_args: TrainingArguments = field(default=TrainingArguments())

    def __repr__(self) -> str:
        return super().__repr__()
