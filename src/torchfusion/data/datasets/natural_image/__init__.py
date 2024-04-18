from torchfusion.core.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "cifar10": [
        "Cifar10",
    ],
    "cifar10_torch_dataset": [
        "Cifar10TorchDataset",
    ],
}
ModuleLazyImporter.register_datasets(__name__, _import_structure)
