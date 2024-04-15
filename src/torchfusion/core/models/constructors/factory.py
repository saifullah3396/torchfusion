from torchfusion.core.models.constructors.diffusers import DiffusersModelConstructor
from torchfusion.core.models.constructors.fusion import FusionModelConstructor
from torchfusion.core.models.constructors.timm import TimmModelConstructor
from torchfusion.core.models.constructors.torchvision import TorchvisionModelConstructor
from torchfusion.core.models.constructors.transformers import (
    TransformersModelConstructor,
)
from torchfusion.utilities.dataclasses.dacite_wrapper import from_dict


class ModelConstructorFactory:
    @staticmethod
    def create(name: str, kwargs: dict):
        if name == "fusion_model":
            model_constructor_class = FusionModelConstructor
        elif name == "transformers":
            model_constructor_class = TransformersModelConstructor
        elif name == "torchvision":
            model_constructor_class = TorchvisionModelConstructor
        elif name == "timm":
            model_constructor_class = TimmModelConstructor
        elif name == "diffusers":
            model_constructor_class = DiffusersModelConstructor
        else:
            raise ValueError(f"Model constructor type [{name}] not supported.")

        return from_dict(
            data_class=model_constructor_class,
            data=kwargs,
        )
