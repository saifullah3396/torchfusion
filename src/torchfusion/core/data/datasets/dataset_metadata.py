import dataclasses
from typing import Optional

from datasets import DatasetInfo
from datasets.features import Features
from torchfusion.core.constants import DataKeys
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class FusionDatasetMetaData:
    """This is similar to the DatasetInfo class in the datasets library. It is used to store metadata about a dataset."""

    description: str = dataclasses.field(default_factory=str)
    citation: str = dataclasses.field(default_factory=str)
    homepage: str = dataclasses.field(default_factory=str)
    license: str = dataclasses.field(default_factory=str)
    features: Optional[Features] = None
    dataset_name: Optional[str] = None
    config_name: Optional[str] = None
    splits: Optional[dict] = None

    @classmethod
    def from_info(cls, dataset_info: DatasetInfo):
        return cls(
            description=dataset_info.description,
            citation=dataset_info.citation,
            homepage=dataset_info.homepage,
            license=dataset_info.license,
            features=dataset_info.features,
            dataset_name=dataset_info.dataset_name,
            config_name=dataset_info.config_name,
            splits=dataset_info.splits,
        )

    def get_labels(self):
        data_labels = None
        logger.debug(f"Dataset loaded with following features: {self.features}")
        if DataKeys.LABEL in self.features:
            from datasets.features import ClassLabel, Sequence

            if isinstance(self.features[DataKeys.LABEL], ClassLabel):
                data_labels = self.features[DataKeys.LABEL].names
            elif isinstance(self.features[DataKeys.LABEL], Sequence):
                data_labels = self.features[DataKeys.LABEL].feature.names

        if DataKeys.OBJECTS in self.features:
            from datasets.features import ClassLabel, Sequence

            # get catogory_id labels
            catogory_id = self.features[DataKeys.OBJECTS][0]["category_id"]

            if isinstance(catogory_id, ClassLabel):
                data_labels = catogory_id.names
            else:
                raise ValueError(f"Unsupported category_id type: {type(catogory_id)}")

        logger.info(f"Data labels = {data_labels}")
        if isinstance(data_labels, list):
            logger.info(f"Number of labels = {len(data_labels)}")
        return data_labels
