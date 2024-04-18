from torchfusion.core.models.tasks import ModelTasks
from torchfusion.core.utilities.module_import import (
    ModuleLazyImporter,
    ModuleRegistryItem,
)

_import_structure = {
    "vaegan": [
        ModuleRegistryItem(
            "FusionModelForVAEGAN",
            "fusion_model_image_vaegan",
            ModelTasks.gan,
        ),
    ],
}

ModuleLazyImporter.register_fusion_models(__name__, _import_structure)
