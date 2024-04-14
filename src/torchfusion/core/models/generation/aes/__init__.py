from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure = {
    "image": [
        ModuleRegistryItem(
            "FusionModelForImageAutoEncoding",
            "fusion_model_image_ae",
            ModelTasks.autoencoding,
        ),
    ],
}

ModuleLazyImporter.register_fusion_models(__name__, _import_structure)
