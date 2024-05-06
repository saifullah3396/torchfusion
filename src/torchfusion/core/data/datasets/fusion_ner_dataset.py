import dataclasses
import typing
from abc import abstractmethod
from typing import Optional

import datasets
import numpy as np
from datasets.features import Features, Image
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.features import FusionClassLabel
from torchfusion.core.data.datasets.fusion_dataset import FusionDataset
from torchfusion.core.data.datasets.fusion_dataset_config import FusionDatasetConfig
from torchfusion.core.data.text_utils.tokenizers.factory import TokenizerFactory

logger = datasets.logging.get_logger(__name__)

_NER_LABELS_PER_SCHEME = {
    # This mapping should be defined for each child dataset
    "IOB": []
}


@dataclasses.dataclass
class FusionNERDatasetConfig(FusionDatasetConfig):
    """Base BuilderConfig for NER datasets."""

    ner_labels: dict = dataclasses.field(default_factory=lambda: {"IOB": []})
    ner_scheme: str = "IOB"
    tokenizer_config: Optional[dict] = None
    segment_level_layout: bool = False

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: typing.Optional[Features] = None,
    ) -> str:
        if self.tokenizer_config is None:
            raise ValueError("Tokenizer config must be provided for this dataset!")

        tokenizer_name = (
            self.tokenizer_config["kwargs"]["model_name"]
            if self.tokenizer_config["name"] == "HuggingfaceTokenizer"
            else self.tokenizer_config["name"]
        )
        return self.name + "-" + tokenizer_name + "-" + self.cache_file_name


class FusionNERDataset(FusionDataset):
    """Base Dataset for NER datasets."""

    BUILDER_CONFIGS = [
        FusionNERDatasetConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default dataset config",
            ner_labels=_NER_LABELS_PER_SCHEME,
            ner_scheme="IOB",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_tokenizer()

    @property
    def ner_labels(self):
        return self.config.ner_labels[self.config.ner_scheme]

    def _initialize_tokenizer(self):
        # create tokenizer
        tokenizer_name = (
            self.config.tokenizer_config["name"]
            if self.config.tokenizer_config is not None
            else "HuggingfaceTokenizer"
        )
        tokenizer_kwargs = (
            self.config.tokenizer_config["kwargs"]
            if self.config.tokenizer_config is not None
            else {
                "model_name": "bert-base-uncased",
                "keys_to_add_on_overflow": [
                    DataKeys.IMAGE_FILE_PATH,
                    DataKeys.IMAGE,
                ],
            }
        )
        self.tokenizer = TokenizerFactory.create(tokenizer_name, tokenizer_kwargs)

    def get_line_bbox(self, bboxes):
        x = [
            bboxes[i][j]
            for i in range(len(bboxes))
            for j in range(0, len(bboxes[i]), 2)
        ]
        y = [
            bboxes[i][j]
            for i in range(len(bboxes))
            for j in range(1, len(bboxes[i]), 2)
        ]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxes))]
        return bbox

    def _preprocess_dataset(self, data, batch_size=100):
        batch_tokenized_data = []
        batches = data.groupby(np.arange(len(data)) // batch_size)
        for _, g in batches:
            batch_tokenized_data.extend(self.tokenizer(g.to_dict("list")))
        return batch_tokenized_data

    def _update_ner_labels(self, data):
        if self.config.ner_scheme == "IOB":
            # the dataset is already in iob format
            return data
        else:
            raise ValueError(f"NER scheme {self._ner_scheme} not supported!")

    def _dataset_features(self):
        return datasets.Features(
            {
                DataKeys.IMAGE: Image(decode=True),
                DataKeys.IMAGE_FILE_PATH: datasets.features.Value("string"),
                DataKeys.TOKEN_IDS: datasets.Sequence(datasets.Value(dtype="int32")),
                DataKeys.TOKEN_BBOXES: datasets.Sequence(
                    datasets.Sequence(datasets.Value("float32"), length=4)
                ),
                DataKeys.ATTENTION_MASKS: datasets.Sequence(
                    datasets.Value(dtype="uint8")
                ),
                DataKeys.LABEL: datasets.Sequence(
                    FusionClassLabel(
                        names=self.ner_labels,
                        num_classes=len(self.ner_labels),
                    ),
                ),
            }
        )

    @abstractmethod
    def _generate_examples_impl(self, *args, **kwargs):
        raise NotImplementedError()
