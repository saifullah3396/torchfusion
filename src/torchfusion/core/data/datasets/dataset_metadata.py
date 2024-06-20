import dataclasses
from collections import OrderedDict
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from datasets import DatasetInfo
from datasets.features import Features
from torchfusion.core.constants import DataKeys
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


def get_labels_color_map(n, name="tab20"):
    return plt.get_cmap(name, n)


@dataclasses.dataclass
class FusionDatasetMetaData:
    """This is similar to the DatasetInfo class in the datasets library. It is used to store metadata about a dataset."""

    description: Optional[str] = None
    citation: Optional[str] = None
    homepage: Optional[str] = None
    license: Optional[str] = None
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
        if DataKeys.LABEL in self.features:
            from datasets.features import ClassLabel, Sequence

            if isinstance(self.features[DataKeys.LABEL], ClassLabel):
                data_labels = self.features[DataKeys.LABEL].names
            elif isinstance(self.features[DataKeys.LABEL], Sequence):
                data_labels = self.features[DataKeys.LABEL].feature.names
            elif isinstance(self.features[DataKeys.LABEL], list):
                data_labels = self.features[DataKeys.LABEL]

        if DataKeys.OBJECTS in self.features:
            from datasets.features import ClassLabel, Sequence

            # get catogory_id labels
            catogory_id = self.features[DataKeys.OBJECTS][0]["category_id"]

            if isinstance(catogory_id, ClassLabel):
                data_labels = catogory_id.names
            else:
                raise ValueError(f"Unsupported category_id type: {type(catogory_id)}")
        return data_labels

    def generate_label_colors(self):
        # get dataset entity labels
        labels = self.get_labels()

        # get cmap
        cmap = get_labels_color_map(len(labels))

        # generate label colors
        labels_colors = {
            label: mpl.colors.rgb2hex(cmap(idx), keep_alpha=False)
            for idx, label in enumerate(labels)
        }
        return labels_colors

    def generate_ner_label_colors(self):
        # get dataset entity labels
        labels = self.get_labels()

        # get all labels without bio tagging
        total_entities = set()
        for label in labels:
            total_entities.add(label.split("-")[-1])
        total_entities = list(total_entities)

        # get cmap
        cmap = get_labels_color_map(len(total_entities))

        # generate label colors
        labels_colors = OrderedDict()
        for label in labels:
            base_label = label.split("-")[-1]
            color_index = total_entities.index(base_label)
            color = cmap(color_index)

            # if it is I just slightly reduce color brightness
            if label.startswith("B"):
                color = [c * 0.8 for c in color]
            labels_colors[label] = mpl.colors.rgb2hex(color, keep_alpha=False)
        return labels_colors

    def get_labels_with_colors(self):
        labels = self.get_labels()
        # check if any label starts with B- or I- tags
        if any([label.startswith("B-") or label.startswith("I-") for label in labels]):
            return self.generate_ner_label_colors()
        else:
            return self.generate_label_colors()
