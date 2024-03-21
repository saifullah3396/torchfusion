from torchfusion.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "prepare_dataset": [
        "PrepareDataset",
    ],
    "test_models": [
        "TestModels",
    ],
    "generate_metrics": [
        "GenerateMetrics",
    ],
    "generate_reconstruction_samples": [
        "GenerateReconstructionSamples",
    ],
    "generate_vae_features": [
        "GenerateVAEFeatures",
    ],
}
ModuleLazyImporter.register_analyzer_tasks(__name__, _import_structure)
