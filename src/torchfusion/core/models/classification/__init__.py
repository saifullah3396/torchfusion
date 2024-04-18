from torchfusion.core.models.tasks import ModelTasks
from torchfusion.core.utilities.module_import import (
    ModuleLazyImporter,
    ModuleRegistryItem,
)

_import_structure = {
    "image": [
        ModuleRegistryItem(
            "FusionModelForImageClassification",
            "fusion_model",
            ModelTasks.image_classification,
        ),
    ],
    "sequence": [
        ModuleRegistryItem(
            "FusionModelForSequenceClassification",
            "fusion_model",
            ModelTasks.sequence_classification,
        ),
    ],
    "tokens": [
        ModuleRegistryItem(
            "FusionModelForTokenClassification",
            "fusion_model",
            ModelTasks.token_classification,
        ),
    ],
}


ModuleLazyImporter.register_fusion_models(__name__, _import_structure)
