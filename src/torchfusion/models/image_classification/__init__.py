from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure_torch = {
    # path: class name, class key, task
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
    # path: class name, class key, task
    "kd_wide_residual_network": [
        ModuleRegistryItem(
            "StudentWideResidualNetwork",
            "kd_wide_residual_network_student",
            ModelTasks.image_classification,
        ),
        ModuleRegistryItem(
            "TeacherWideResidualNetwork",
            "kd_wide_residual_network_teacher",
            ModelTasks.image_classification,
        ),
    ],
}
ModuleLazyImporter.register_torch_models(__name__, _import_structure_torch)
