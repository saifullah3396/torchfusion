"""
Defines the factory for TrainValSampler class and its children.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

from torchfusion.core.args.args import FusionArguments
from torchfusion.core.models.args.model_args import ModelArguments
from torchfusion.utilities.module_import import ModuleLazyImporter

if TYPE_CHECKING:
    from torchfusion.core.models.fusion_model import FusionModel


class ModelFactory:
    """
    The model factory that initializes the model based on its name, subtype, and
    training task.
    """

    @staticmethod
    def get_torch_model_class(model_name, model_task) -> Type[FusionModel]:
        """
        Find the model given the task and its name
        """
        models_in_task = ModuleLazyImporter.get_torch_models().get(model_task, None)
        if models_in_task is None:
            raise ValueError(f"Task [{model_task}] is not supported.")
        model_class = models_in_task.get(model_name, None)
        if model_class is None:
            raise ValueError(
                f"Model [{model_task}/{model_name}] "
                f"is not supported. Supported models are: {models_in_task}."
            )
        model_class = model_class()
        return model_class

    @staticmethod
    def get_fusion_model_class(model_args: ModelArguments) -> Type[FusionModel]:
        """
        Find the model given the task and its name
        """

        models_in_task = ModuleLazyImporter.get_fusion_models().get(
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
        **model_kwargs,
    ) -> FusionModel:
        """
        Initialize the model
        """

        model_class = ModelFactory.get_fusion_model_class(args.model_args)
        fusion_model = model_class(args=args, **model_kwargs)
        fusion_model.build_model(checkpoint=checkpoint, strict=strict)
        return fusion_model

    # @staticmethod
    # def create_fusion_kd_model(
    #     args: FusionArguments,
    #     teacher_checkpoint: Optional[str] = None,
    #     student_checkpoint: Optional[str] = None,
    #     strict: bool = False,
    #     wrapper_class=FusionKDModel,
    #     load_checkpoint_if_available: bool = True,
    #     **model_kwargs,
    # ) -> FusionKDModel:
    #     """
    #     Initialize the model
    #     """

    #     model = wrapper_class(
    #         args=args,
    #         teacher_model_class=ModelFactory.get_fusion_nn_model_class(
    #             args.teacher_model_args
    #         ),
    #         student_model_class=ModelFactory.get_fusion_nn_model_class(
    #             args.student_model_args
    #         ),
    #         **model_kwargs,
    #     )
    #     if load_checkpoint_if_available:
    #         teacher_checkpoint = (
    #             args.teacher_model_args.checkpoint
    #             if teacher_checkpoint is None
    #             else teacher_checkpoint
    #         )
    #         checkpoint_state_dict_key = args.model_args.checkpoint_state_dict_key
    #         setup_checkpoint(
    #             model.torch_teacher_model,
    #             teacher_model_args,
    #             checkpoint=teacher_checkpoint,
    #             strict=strict,
    #         )
    #         setup_checkpoint(
    #             model.torch_student_model,
    #             student_model_args,
    #             checkpoint=student_checkpoint,
    #             strict=strict,
    #         )

    #     return model
