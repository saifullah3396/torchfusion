from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure = {
    "kl_aegan": [
        ModuleRegistryItem(
            "FusionModelForKLAEGAN",
            "fusion_model_kl_aegan",
            ModelTasks.gan,
        ),
    ],
}

ModuleLazyImporter.register_fusion_models(__name__, _import_structure)
