from torchfusion.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "sroie": [
        "SROIE",
    ],
    "cord": [
        "CORD",
    ],
    "funsd": [
        "FUNSD",
    ],
    "wild_receipts": [
        "WildReceipts",
    ],
    "docile": [
        "DOCILE",
    ],
}
ModuleLazyImporter.register_datasets(__name__, _import_structure)
