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

"""SROIE dataset"""


import collections
import csv
import dataclasses
import itertools
import json
import os
from dataclasses import field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import datasets
import numpy as np
import pandas as pd
import PIL
import tqdm
from attr import dataclass
from datasets import load_dataset
from datasets.features import Features, Image

from torchfusion.core.args.args_base import ClassInitializerArgs
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_dataset import FusionDataset
from torchfusion.core.data.datasets.fusion_dataset_config import FusionDatasetConfig
from torchfusion.core.data.datasets.fusion_image_dataset import (
    FusionImageDataset,
    FusionImageDatasetConfig,
)
from torchfusion.core.data.datasets.fusion_ner_dataset import (
    FusionNERDataset,
    FusionNERDatasetConfig,
)
from torchfusion.core.data.text_utils.tokenizers.factory import TokenizerFactory
from torchfusion.core.data.text_utils.tokenizers.hf_tokenizer import DataPadder
from torchfusion.core.data.text_utils.utilities import normalize_bbox

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """SROIE Receipts Dataset"""

_HOMEPAGE = "https://rrc.cvc.uab.es/?ch=13"

_LICENSE = "Apache-2.0 license"

_NER_LABELS_PER_SCHEME = {
    "IOB": [
        "O",
        "B-COMPANY",
        "I-COMPANY",
        "B-DATE",
        "I-DATE",
        "B-ADDRESS",
        "I-ADDRESS",
        "B-TOTAL",
        "I-TOTAL",
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
class SROIEConfig(FusionNERDatasetConfig):
    pass


class SROIE(FusionNERDataset):
    """SROIE dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        SROIEConfig(
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
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "test"},
            ),
        ]

    def _load_dataset_to_pandas(self, filepath):
        ann_dir = os.path.join(filepath, "tagged")
        image_dir = os.path.join(filepath, "images")

        data = []
        for fname in sorted(os.listdir(image_dir)):
            name, ext = os.path.splitext(fname)
            file_path = os.path.join(ann_dir, name + ".json")
            with open(file_path, "r", encoding="utf8") as f:
                sample = json.load(f)
            image_path = os.path.join(image_dir, fname)
            image = PIL.Image.open(image_path)
            boxes = [normalize_bbox(box, image.size) for box in sample["bbox"]]

            data.append(
                {
                    DataKeys.WORDS: sample["words"],
                    DataKeys.WORD_BBOXES: boxes,
                    DataKeys.LABEL: [
                        self.ner_labels.index(l) for l in sample["labels"]
                    ],
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
