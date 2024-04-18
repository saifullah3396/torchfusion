from torchfusion.core.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "kfolds_cross_val": ["KFoldCrossValSampler"],
    "random_split": ["RandomSplitSampler"],
}

ModuleLazyImporter.register_train_val_samplers(__name__, _import_structure)
