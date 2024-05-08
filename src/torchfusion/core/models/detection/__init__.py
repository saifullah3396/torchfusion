from torchfusion.core.models.tasks import ModelTasks
from torchfusion.core.utilities.module_import import (
    ModuleLazyImporter,
    ModuleRegistryItem,
)

_import_structure = {
    "image": [
        ModuleRegistryItem(
            "FusionModelForImageObjectDetection",
            "fusion_model",
            ModelTasks.object_detection,
        ),
    ],
}


ModuleLazyImporter.register_fusion_models(__name__, _import_structure)
