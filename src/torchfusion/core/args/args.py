from dataclasses import dataclass, field
from typing import List, Optional, Union

from torchfusion.core.args.args_base import ArgumentsBase, ClassInitializerArgs
from torchfusion.core.args.general_args import GeneralArguments
from torchfusion.core.data.args.data_args import DataArguments
from torchfusion.core.data.args.data_loader_args import DataLoaderArguments
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

    # Arguments related to defining default data augmentations for training.
    train_preprocess_augs: Optional[
        Union[ClassInitializerArgs, List[ClassInitializerArgs]]
    ] = field(default=None)

    # Arguments related to defining default data augmentations for evaluation.
    eval_preprocess_augs: Optional[
        Union[ClassInitializerArgs, List[ClassInitializerArgs]]
    ] = field(default=None)

    # Arguments related to defining default data augmentations for training.
    train_realtime_augs: Optional[
        Union[ClassInitializerArgs, List[ClassInitializerArgs]]
    ] = field(default=None)

    # Arguments related to defining default data augmentations for training.
    eval_realtime_augs: Optional[
        Union[ClassInitializerArgs, List[ClassInitializerArgs]]
    ] = field(default=None)

    # Arguments related to data loading or specifically torch dataloaders.
    data_loader_args: DataLoaderArguments = field(
        default_factory=lambda: DataLoaderArguments(),
    )

    # Train validation sampling arguments
    train_val_sampler: Optional[ClassInitializerArgs] = field(default=None)

    # Training related arguments
    training_args: TrainingArguments = field(default=TrainingArguments())

    def __repr__(self) -> str:
        return super().__repr__()
