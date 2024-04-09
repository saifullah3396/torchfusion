from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure = {
    "hf_model": [
        ModuleRegistryItem(
            "HuggingfaceModelForSequenceClassification",
            "hf_model",
            ModelTasks.sequence_classification,
        ),
    ],
    "hf_model": [
        ModuleRegistryItem(
            "HuggingfaceModelForTokenClassification",
            "hf_model",
            ModelTasks.token_classification,
        ),
    ],
}

ModuleLazyImporter.register_models(__name__, _import_structure)
