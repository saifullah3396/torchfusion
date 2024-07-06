from torchfusion.core.models.constructors.detectron2 import Detectron2ModelConstructor
from torchfusion.core.models.constructors.diffusers import DiffusersModelConstructor
from torchfusion.core.models.constructors.fusion import (
    FusionModelConstructor,
    FusionModelWithBackboneConstructor,
)
from torchfusion.core.models.constructors.timm import TimmModelConstructor
from torchfusion.core.models.constructors.torchvision import TorchvisionModelConstructor
from torchfusion.core.models.constructors.transformers import (
    TransformersModelConstructor,
)
from torchfusion.core.utilities.dataclasses.dacite_wrapper import from_dict


class ModelConstructorFactory:
    @staticmethod
    def create(name: str, kwargs: dict):
        if name == "fusion_model":
            model_constructor_class = FusionModelConstructor
        elif name == "fusion_model_with_backbone":
            model_constructor_class = FusionModelWithBackboneConstructor
        elif name == "transformers":
            model_constructor_class = TransformersModelConstructor
        elif name == "torchvision":
            model_constructor_class = TorchvisionModelConstructor
        elif name == "timm":
            model_constructor_class = TimmModelConstructor
        elif name == "detectron2":
            model_constructor_class = Detectron2ModelConstructor
        elif name == "diffusers":
            model_constructor_class = DiffusersModelConstructor
        else:
            raise ValueError(f"Model constructor type [{name}] not supported.")

        return from_dict(
            data_class=model_constructor_class,
            data=kwargs,
        )
