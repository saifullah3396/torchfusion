from __future__ import annotations

import importlib
from copy import deepcopy

from torchfusion.core.data.data_augmentations.general import DictTransform
from torchfusion.core.utilities.dataclasses.dacite_wrapper import from_dict
from torchfusion.core.utilities.module_import import ModuleLazyImporter


class DataAugmentationFactory:
    @staticmethod
    def create(
        name: str,
        kwargs: dict,
    ):
        aug_class = ModuleLazyImporter.get_augmentations().get(name, None)

        if aug_class is not None:
            # lazy load class
            aug_class = aug_class()
        else:
            try:
                aug_class = getattr(
                    importlib.import_module(".".join(name.split(".")[:-1])),
                    name.split(".")[-1],
                )
            except Exception as e:
                raise ValueError(f"DataAugmentation [{name}] is not supported.")

        kwargs = deepcopy(kwargs)
        if "key" in kwargs:
            key = kwargs.pop("key")
            if isinstance(kwargs, dict):
                return DictTransform(
                    key=key,
                    transform=from_dict(
                        data_class=aug_class,
                        data=kwargs,
                    ),
                )
        else:
            if isinstance(kwargs, dict):
                return from_dict(
                    data_class=aug_class,
                    data=kwargs,
                )
