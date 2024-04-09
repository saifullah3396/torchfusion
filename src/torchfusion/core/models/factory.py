"""
Defines the factory for TrainValSampler class and its children.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

from torchfusion.core.args.args import FusionArguments
from torchfusion.core.data.args.data_args import DataArguments
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.fusion_nn_model import FusionNNModel
from torchfusion.core.models.utilities.checkpoints import setup_checkpoint
from torchfusion.core.training.args.training import TrainingArguments
from torchfusion.utilities.module_import import ModuleLazyImporter

if TYPE_CHECKING:
    from torchfusion.core.models.fusion_model import FusionModel


class ModelFactory:
    """
    The model factory that initializes the model based on its name, subtype, and
    training task.
    """

    @staticmethod
    def get_fusion_nn_model_class(model_args: ModelArguments) -> Type[FusionNNModel]:
        """
        Find the model given the task and its name
        """

        models_in_task = ModuleLazyImporter.get_models().get(
            model_args.model_task, None
        )
        if models_in_task is None:
            raise ValueError(f"Task [{model_args.model_task}] is not supported.")
        model_class = models_in_task.get(model_args.name, None)
        if model_class is None:
            raise ValueError(
                f"Model [{model_args.model_task}/{model_args.name}] "
                "is not supported."
            )
        model_class = model_class()
        return model_class

    @staticmethod
    def create_fusion_model(
        args: FusionArguments,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        wrapper_class=FusionModel,
        load_checkpoint_if_available: bool = True,
        **model_kwargs,
    ) -> FusionModel:
        """
        Initialize the model
        """

        model = wrapper_class(args=args, **model_kwargs)
        if load_checkpoint_if_available:
            checkpoint = (
                args.model_args.pretrained_checkpoint
                if checkpoint is None
                else checkpoint
            )
            checkpoint_state_dict_key = args.model_args.checkpoint_state_dict_key
            setup_checkpoint(
                model.torch_model,
                checkpoint=checkpoint,
                checkpoint_state_dict_key=checkpoint_state_dict_key,
                strict=strict,
            )

        return model
        if load_checkpoint_if_available:
            setup_checkpoint(model, model_args, checkpoint=checkpoint, strict=strict)

        return model

    @staticmethod
    def create_fusion_nn_model(
        model_args: ModelArguments,
        training_args: TrainingArguments,
        data_args: DataArguments,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        dataset_features: Optional[dict] = None,
        **model_kwargs,
    ) -> FusionNNModel:
        """
        Initialize the model
        """

        model_class = ModelFactory.get_fusion_nn_model_class(model_args)
        model = model_class(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            dataset_features=dataset_features,
            **model_kwargs,
        )
        model.build_model()
        setup_checkpoint(model, model_args, checkpoint=checkpoint, strict=strict)

        return model
