from torchfusion.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "ar_group_batch_sampler": [
        "AspectRatioGroupBatchSampler",
    ],
    "group_batch_sampler": ["GroupBatchSampler"],
}

ModuleLazyImporter.register_batch_samplers(__name__, _import_structure)
