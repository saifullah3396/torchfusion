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

"""FUNSD dataset"""


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
_DESCRIPTION = """FUNSD Dataset"""

_HOMEPAGE = "https://guillaumejaume.github.io/FUNSD/"

_LICENSE = "Apache-2.0 license"

_NER_LABELS_PER_SCHEME = {
    "IOB": [
        "O",
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
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
class FUNSDConfig(FusionNERDatasetConfig):
    pass


class FUNSD(FusionNERDataset):
    """FUNSD dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        FUNSDConfig(
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
        for dir in ["training_data", "testing_data"]:
            assert (
                Path(self.config.data_dir) / dir
            ).exists(), f"Data directory {self.config.data_dir} {dir} does not exist."
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "training_data"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "testing_data"},
            ),
        ]

    def _quad_to_box(self, quad):
        # test 87 is wrongly annotated
        box = (max(0, quad["x1"]), max(0, quad["y1"]), quad["x3"], quad["y3"])
        if box[3] < box[1]:
            bbox = list(box)
            tmp = bbox[3]
            bbox[3] = bbox[1]
            bbox[1] = tmp
            box = tuple(bbox)
        if box[2] < box[0]:
            bbox = list(box)
            tmp = bbox[2]
            bbox[2] = bbox[0]
            bbox[0] = tmp
            box = tuple(bbox)
        return box

    def _load_dataset_to_pandas(self, filepath):
        ann_dir = os.path.join(filepath, "annotations")
        image_dir = os.path.join(filepath, "images")

        data = []
        for fname in sorted(os.listdir(image_dir)):
            name, ext = os.path.splitext(fname)
            ann_path = os.path.join(ann_dir, name + ".json")
            image_path = os.path.join(image_dir, fname)
            with open(ann_path, "r", encoding="utf8") as f:
                annotation = json.load(f)
            image_size = PIL.Image.open(image_path).size

            words_list = []
            bboxes = []
            labels = []
            # get annotations
            for item in annotation["form"]:
                cur_line_bboxes = []
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue

                if label == "other":
                    for w in words:
                        words_list.append(w["text"])
                        labels.append("O")
                        cur_line_bboxes.append(normalize_bbox(w["box"], image_size))
                else:
                    words_list.append(words[0]["text"])
                    labels.append("B-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(words[0]["box"], image_size))
                    for w in words[1:]:
                        words_list.append(w["text"])
                        labels.append("I-" + label.upper())
                        cur_line_bboxes.append(normalize_bbox(w["box"], image_size))
                if self.config.segment_level_layout:
                    cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)
            data.append(
                {
                    DataKeys.WORDS: words_list,
                    DataKeys.WORD_BBOXES: bboxes,
                    DataKeys.LABEL: [self.ner_labels.index(l) for l in labels],
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
