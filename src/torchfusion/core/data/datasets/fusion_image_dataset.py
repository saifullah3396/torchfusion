import dataclasses
from typing import List

import datasets
from datasets.features import Image
from datasets.tasks import ImageClassification

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_dataset import FusionDataset, FusionDatasetConfig

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class FusionImageDatasetConfig(FusionDatasetConfig):
    """BuilderConfig for FusionImageDataset"""

    labels: List[str] = None


class FusionImageDataset(FusionDataset):
    BUILDER_CONFIGS = [
        FusionImageDatasetConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default image classification dataset config",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=self._dataset_features(),
            supervised_keys=(DataKeys.IMAGE, DataKeys.LABEL),
            homepage=self.config.homepage,
            citation=self.config.citation,
            task_templates=ImageClassification(image_column=DataKeys.IMAGE, label_column=DataKeys.LABEL),
        )

    def _dataset_features(self):
        return datasets.Features(
            {
                DataKeys.IMAGE: Image(decode=True),
                DataKeys.IMAGE_FILE_PATH: datasets.features.Value("string"),
                DataKeys.LABEL: datasets.features.ClassLabel(names=self.config.labels),
            }
        )
