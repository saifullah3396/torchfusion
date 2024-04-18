from torchfusion.core.utilities.module_import import ModuleLazyImporter

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
    "evaluate_image_reconstruction": [
        "EvaluateImageReconstruction",
    ],
    "save_vae_features": [
        "SaveVAEFeatures",
    ],
}
ModuleLazyImporter.register_analyzer_tasks(__name__, _import_structure)
