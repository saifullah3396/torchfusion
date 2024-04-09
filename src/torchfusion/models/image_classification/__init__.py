from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure_torch = {
    "toy_models": [
        ModuleRegistryItem(
            "ToyModelCifar10",
            "toy_model_cifar10",
            ModelTasks.image_classification,
        ),
        ModuleRegistryItem(
            "ToyModelMNIST",
            "toy_model_mnist",
            ModelTasks.image_classification,
        ),
    ],
}
ModuleLazyImporter.register_torch_models(__name__, _import_structure_torch)
