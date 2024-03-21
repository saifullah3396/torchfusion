from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure = {
    "hf_model": [
        ModuleRegistryItem("HuggingfaceModelForImageClassification", "hf_model", ModelTasks.image_classification),
    ],
    "tv_model": [
        ModuleRegistryItem("TorchvisionModelForImageClassification", "tv_model", ModelTasks.image_classification),
    ],
    "timm_model": [
        ModuleRegistryItem("TimmModelForImageClassification", "timm_model", ModelTasks.image_classification),
    ],
    "toy_models": [
        ModuleRegistryItem("ToyModelForCifar10Classification", "toy_model_cifar10", ModelTasks.image_classification),
        ModuleRegistryItem(
            "ToyModelForMNISTClassification",
            "toy_model_mnist",
            ModelTasks.image_classification,
        ),
    ],
}

ModuleLazyImporter.register_models(__name__, _import_structure)
