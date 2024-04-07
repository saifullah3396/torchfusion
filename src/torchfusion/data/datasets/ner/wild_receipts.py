# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WildReceipts dataset"""


import dataclasses
import json
import os
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import PIL

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_ner_dataset import (
    FusionNERDataset,
    FusionNERDatasetConfig,
)
from torchfusion.core.data.text_utils.utilities import normalize_bbox

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """WildReceipts Dataset"""

_HOMEPAGE = ""

_LICENSE = "Apache-2.0 license"

_URLS = ["https://download.openmmlab.com/mmocr/data/wildreceipt.tar"]

_NER_LABELS_PER_SCHEME = {
    "IOB": [
        "B-Store_name_value",
        "B-Store_name_key",
        "B-Store_addr_value",
        "B-Store_addr_key",
        "B-Tel_value",
        "B-Tel_key",
        "B-Date_value",
        "B-Date_key",
        "B-Time_value",
        "B-Time_key",
        "B-Prod_item_value",
        "B-Prod_item_key",
        "B-Prod_quantity_value",
        "B-Prod_quantity_key",
        "B-Prod_price_value",
        "B-Prod_price_key",
        "B-Subtotal_value",
        "B-Subtotal_key",
        "B-Tax_value",
        "B-Tax_key",
        "B-Tips_value",
        "B-Tips_key",
        "B-Total_value",
        "B-Total_key",
        "O",
    ]
}


def convert_to_list(row):
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            row[k] = v.tolist()
            if isinstance(v[0], np.ndarray):
                row[k] = [x.tolist() for x in v]
    return row


@dataclasses.dataclass
class WildReceiptsConfig(FusionNERDatasetConfig):
    pass


class WildReceipts(FusionNERDataset):
    """WildReceipts dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        WildReceiptsConfig(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            ner_labels=_NER_LABELS_PER_SCHEME,
            ner_scheme="IOB",
        ),
    ]

    def _split_generators(self, dl_manager):
        for file in ["train.txt", "test.txt"]:
            assert (
                Path(self.config.data_dir) / file
            ).exists(), f"Data directory {self.config.data_dir} {file} does not exist."
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "train.txt"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "test.txt"},
            ),
        ]

    def _load_dataset_to_pandas(self, filepath):
        item_list = []
        with open(filepath, "r") as f:
            for line in f:
                item_list.append(line.rstrip("\n\r"))

        class_list = pd.read_csv(
            filepath.parent / "class_list.txt", delimiter="\s", header=None
        )
        id2labels = dict(zip(class_list[0].tolist(), class_list[1].tolist()))

        data = []
        for guid, fname in enumerate(item_list):
            ann = json.loads(fname)
            image_path = filepath.parent / ann["file_name"]
            image_size = PIL.Image.open(image_path).size

            words = []
            labels = []
            bboxes = []
            for i in ann["annotations"]:
                label = id2labels[i["label"]]
                if label == "Ignore":  # label 0 is attached to ignore so we skip it
                    continue
                if label in ["Others"]:
                    label = "O"
                else:
                    label = "B-" + label
                labels.append(label)
                words.append(i["text"])
                bboxes.append(
                    normalize_bbox(
                        [i["box"][6], i["box"][7], i["box"][2], i["box"][3]], image_size
                    )
                )

            data.append(
                {
                    DataKeys.WORDS: words,
                    DataKeys.WORD_BBOXES: bboxes,
                    DataKeys.LABEL: [self.config.ner_labels.index(l) for l in labels],
                    DataKeys.IMAGE_FILE_PATH: image_path,
                    # we don't load all images here to save memory
                }
            )
        return pd.DataFrame(data)

    def _generate_examples_impl(
        self,
        filepath,
    ):
        data = self._load_dataset_to_pandas(filepath)
        self._logger.info("Base dataset pandas dataframe loaded:")
        self._logger.info(data)
        try:
            data = data.apply(convert_to_list, axis=1)
            self._update_ner_labels(data)
            tokenized_data = self._preprocess_dataset(data)
            for idx, sample in enumerate(tokenized_data):
                if DataKeys.IMAGE_FILE_PATH in sample:
                    sample[DataKeys.IMAGE] = PIL.Image.open(
                        sample[DataKeys.IMAGE_FILE_PATH]
                    )
                yield idx, sample
        except Exception as e:
            print(e)
