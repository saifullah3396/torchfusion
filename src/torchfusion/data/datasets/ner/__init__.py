from torchfusion.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "sroie": [
        "SROIE",
    ],
}
ModuleLazyImporter.register_datasets(__name__, _import_structure)
