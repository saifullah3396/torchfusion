from __future__ import annotations

import importlib
from dataclasses import dataclass

from torchfusion.utilities.general import str_to_underscored_lower


@dataclass
class ModuleRegistryItem:
    cls: str
    name: str = ""
    task: str = None


class ModuleLazyImporter:
    MODULES_REGISTRY = {}

    @staticmethod
    def get_module(name: str, module_name: str, cls_name: str):
        try:
            return lambda: getattr(
                importlib.import_module(name + "." + module_name), cls_name
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {name}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    @staticmethod
    def register_modules(name, import_structure, registry_key):
        for key, value in import_structure.items():
            for item in value:
                if isinstance(item, ModuleRegistryItem):
                    (reg_name, task, cls) = (
                        item.name,
                        item.task,
                        item.cls,
                    )
                else:
                    task = None
                    reg_name = item
                    cls = item

                reg_dict = ModuleLazyImporter.MODULES_REGISTRY
                if registry_key not in ModuleLazyImporter.MODULES_REGISTRY:
                    reg_dict[registry_key] = {}

                reg_dict = reg_dict[registry_key]

                if task is not None:
                    if task not in reg_dict:
                        reg_dict[task] = {}
                    reg_dict = reg_dict[task]

                if reg_name != "":
                    reg_dict[reg_name] = ModuleLazyImporter.get_module(name, key, cls)
                else:
                    reg_dict[str_to_underscored_lower(cls.__name__)] = (
                        lambda: ModuleLazyImporter.get_module(name, key, cls)
                    )

    @staticmethod
    def register_fusion_models(name, import_structure, registry_key="fusion_models"):
        ModuleLazyImporter.register_modules(name, import_structure, registry_key)

    @staticmethod
    def register_torch_models(name, import_structure, registry_key="torch_models"):
        ModuleLazyImporter.register_modules(name, import_structure, registry_key)

    @staticmethod
    def register_datasets(name, import_structure, registry_key="datasets"):
        ModuleLazyImporter.register_modules(name, import_structure, registry_key)

    @staticmethod
    def register_batch_samplers(name, import_structure, registry_key="batch_samples"):
        ModuleLazyImporter.register_modules(name, import_structure, registry_key)

    @staticmethod
    def register_train_val_samplers(
        name, import_structure, registry_key="train_val_samplers"
    ):
        ModuleLazyImporter.register_modules(name, import_structure, registry_key)

    @staticmethod
    def register_augmentations(name, import_structure, registry_key="augmentations"):
        ModuleLazyImporter.register_modules(name, import_structure, registry_key)

    @staticmethod
    def register_analyzer_tasks(name, import_structure, registry_key="analyzer_tasks"):
        ModuleLazyImporter.register_modules(name, import_structure, registry_key)

    @staticmethod
    def register_tokenizers(name, import_structure, registry_key="tokenizers"):
        ModuleLazyImporter.register_modules(name, import_structure, registry_key)

    @staticmethod
    def get_fusion_models():
        if "fusion_models" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["fusion_models"]
        else:
            return {}

    @staticmethod
    def get_torch_models():
        if "torch_models" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["torch_models"]
        else:
            return {}

    @staticmethod
    def get_datasets():
        if "datasets" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["datasets"]
        else:
            return {}

    @staticmethod
    def get_batch_samplers():
        if "batch_samplers" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["batch_samplers"]
        else:
            return {}

    @staticmethod
    def get_data_cachers():
        if "data_cachers" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["data_cachers"]
        else:
            return {}

    @staticmethod
    def get_train_val_samplers():
        if "train_val_samplers" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["train_val_samplers"]
        else:
            return {}

    @staticmethod
    def get_augmentations():
        if "augmentations" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["augmentations"]
        else:
            return {}

    @staticmethod
    def get_tokenizers():
        if "tokenizers" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["tokenizers"]
        else:
            return {}

    @staticmethod
    def get_analyzer_tasks():
        if "analyzer_tasks" in ModuleLazyImporter.MODULES_REGISTRY:
            return ModuleLazyImporter.MODULES_REGISTRY["analyzer_tasks"]
        else:
            return {}
