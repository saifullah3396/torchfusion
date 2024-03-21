from torchfusion.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "general": ["ToTensor", "Brightness" "Contrast" "GrayScaleToRGB" "RGBToBGR"],
    "noise": ["GaussianNoiseRGB" "ShotNoiseRGB" "FibrousNoise" "MultiscaleNoise"],
    "transforms": [
        "Translation",
        "Scale",
        "Rotation",
        "RandomChoiceAffine",
        "Elastic",
        "Rescale",
        "RescaleOneDim",
        "RescaleWithAspectAndPad",
        "RandomRescale",
        "RandomResizedCrop",
        "RandomCrop",
        # "RandomResizedMaskedCrop
    ],
    "advanced": [
        "PreprocessAug",
        "BasicImageAug",
        "RandAug",
        "Moco",
        "BarlowTwins",
        "MultiCrop",
        "BinarizationAug",
        # "TwinDocs",
    ],
    "blur": [
        "GaussianBlur",
        "GaussianBlurPIL",
        "BinaryBlur",
        "NoisyBinaryBlur",
        "DefocusBlur",
        "MotionBlur",
        "ZoomBlur",
    ],
    "distortions": [
        "RandomDistortion",
        "RandomBlotches",
        "SurfaceDistortion",
        "Threshold",
        "Pixelate",
        "JPEGCompression",
        "Solarization",
    ],
}

ModuleLazyImporter.register_augmentations(__name__, _import_structure)
