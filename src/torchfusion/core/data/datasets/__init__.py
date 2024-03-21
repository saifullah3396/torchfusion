import inspect

from datasets.packaged_modules import (
    _EXTENSION_TO_MODULE,
    _MODULE_SUPPORTS_METADATA,
    _PACKAGED_DATASETS_MODULES,
    _hash_python_lines,
)

# here instead of adding it to our own registery, we add it to huggingface packaged_modules
# this is necessary to properly handle data_files automatically from folder structure
import torchfusion.core.data.datasets.fusion_image_folder_dataset as fusion_image_folder_dataset

_FILE = fusion_image_folder_dataset
_PACKAGED_DATASETS_MODULES["fusion_image_folder"] = (_FILE.__name__, _hash_python_lines(inspect.getsource(_FILE)))
_EXTENSION_TO_MODULE.update(
    {ext[1:]: ("fusion_image_folder", {}) for ext in fusion_image_folder_dataset.FusionImageFolderDataset.EXTENSIONS}
)
_MODULE_SUPPORTS_METADATA.update("fusion_image_folder")
