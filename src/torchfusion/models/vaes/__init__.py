from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure_torch = {
    # path: class name, class key, task
    "compvis_autoencoder": [
        ModuleRegistryItem(
            "CompVisAutoEncoder",
            "compvis_autoencoder",
            ModelTasks.autoencoding,
        ),
    ],
}
ModuleLazyImporter.register_torch_models(__name__, _import_structure_torch)
