from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torchfusion.core.models.constructors.base import ModelConstructor
from torchfusion.core.models.factory import ModelFactory
from torchfusion.core.models.tasks import ModelTasks


@dataclass
class FusionModelConstructor(ModelConstructor):
    def __post_init__(self):
        super().__post_init__()
        assert self.model_task in [
            ModelTasks.image_classification,
            ModelTasks.token_classification,
            ModelTasks.gan,
            ModelTasks.autoencoding,
        ], f"Task {self.model_task} not supported for FusionModelConstructor."

    def _init_model(self, **kwargs) -> torch.Any:
        model_class = ModelFactory.get_torch_model_class(
            self.model_name, self.model_task
        )
        return model_class(**self.init_args, **kwargs)


@dataclass
class FusionModelWithBackboneConstructor(FusionModelConstructor):
    def _init_model(self, **kwargs) -> torch.Any:
        from torchfusion.core.models.constructors.factory import ModelConstructorFactory

        # get init args
        init_args = copy.deepcopy(self.init_args)

        # setup backbone model from its own constructor
        backbone_model_constructor = init_args.pop("backbone_model_constructor", None)
        backbone_model_constructor_args = init_args.pop(
            "backbone_model_constructor_args", None
        )
        backbone_model_constructor_args["model_task"] = self.model_task
        backbone_model_constructor = ModelConstructorFactory.create(
            name=backbone_model_constructor,
            kwargs=backbone_model_constructor_args,
        )
        backbone_model = backbone_model_constructor(**kwargs)

        # setup the model that takes backbone model as input
        model_class = ModelFactory.get_torch_model_class(
            self.model_name, self.model_task
        )
        return model_class(
            **init_args,
            backbone_model_name=backbone_model_constructor.model_name,
            backbone_model=backbone_model,
            **kwargs,
        )
