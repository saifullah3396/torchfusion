import dataclasses
from typing import List

import datasets
from datasets.features import Image
from datasets.tasks import ImageClassification

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_dataset import (
    FusionDataset,
    FusionDatasetConfig,
)

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class FusionImageTextLayoutDatasetConfig(FusionDatasetConfig):
    """BuilderConfig for FusionImageDataset"""

    pass


class FusionImageTextLayoutDataset(FusionDataset):
    BUILDER_CONFIGS = [
        FusionImageTextLayoutDatasetConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default image classification dataset config",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=self._dataset_features(),
            homepage=self.config.homepage,
            citation=self.config.citation,
        )

    def _dataset_features(self):
        return datasets.Features(
            {
                DataKeys.IMAGE: Image(decode=True),
                DataKeys.IMAGE_FILE_PATH: datasets.features.Value("string"),
                DataKeys.WORDS: datasets.Sequence(datasets.Value(dtype="string")),
                DataKeys.WORD_BBOXES: datasets.Sequence(
                    datasets.Value("float32"), length=4
                ),
            }
        )
