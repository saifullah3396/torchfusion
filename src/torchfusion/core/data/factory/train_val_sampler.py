from __future__ import annotations

from torchfusion.core.utilities.module_import import ModuleLazyImporter


class TrainValSamplerFactory:
    @staticmethod
    def create(name: str, kwargs: dict):
        from torchfusion.core.utilities.dataclasses.dacite_wrapper import from_dict

        sampler_class = ModuleLazyImporter.get_train_val_samplers().get(name, None)
        if sampler_class is None:
            raise ValueError(
                f"TrainValSampler [{name}] is not supported. Choices = "
                f"{ModuleLazyImporter.get_train_val_samplers().keys()}"
            )
        sampler_class = sampler_class()
        return from_dict(
            data_class=sampler_class,
            data=kwargs,
        )
