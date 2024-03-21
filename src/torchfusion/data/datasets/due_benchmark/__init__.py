from torchfusion.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "due_benchmark": [
        "DueBenchmark",
    ],
}
ModuleLazyImporter.register_datasets(__name__, _import_structure)
