from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure = {
    "image": [
        ModuleRegistryItem(
            "FusionModelForVariationalImageAutoEncoding",
            "fusion_model_image_vae",
            ModelTasks.autoencoding,
        ),
    ],
}

ModuleLazyImporter.register_models(__name__, _import_structure)
