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

"""CORD dataset"""


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
_DESCRIPTION = """CORD Dataset"""

_HOMEPAGE = "https://github.com/clovaai/cord"

_LICENSE = "Apache-2.0 license"

_NER_LABELS_PER_SCHEME = {
    "IOB": [
        "O",
        "B-MENU.NM",
        "B-MENU.NUM",
        "B-MENU.UNITPRICE",
        "B-MENU.CNT",
        "B-MENU.DISCOUNTPRICE",
        "B-MENU.PRICE",
        "B-MENU.ITEMSUBTOTAL",
        "B-MENU.VATYN",
        "B-MENU.ETC",
        "B-MENU.SUB.NM",
        "B-MENU.SUB.UNITPRICE",
        "B-MENU.SUB.CNT",
        "B-MENU.SUB.PRICE",
        "B-MENU.SUB.ETC",
        "B-VOID_MENU.NM",
        "B-VOID_MENU.PRICE",
        "B-SUB_TOTAL.SUBTOTAL_PRICE",
        "B-SUB_TOTAL.DISCOUNT_PRICE",
        "B-SUB_TOTAL.SERVICE_PRICE",
        "B-SUB_TOTAL.OTHERSVC_PRICE",
        "B-SUB_TOTAL.TAX_PRICE",
        "B-SUB_TOTAL.ETC",
        "B-TOTAL.TOTAL_PRICE",
        "B-TOTAL.TOTAL_ETC",
        "B-TOTAL.CASHPRICE",
        "B-TOTAL.CHANGEPRICE",
        "B-TOTAL.CREDITCARDPRICE",
        "B-TOTAL.EMONEYPRICE",
        "B-TOTAL.MENUTYPE_CNT",
        "B-TOTAL.MENUQTY_CNT",
        "I-MENU.NM",
        "I-MENU.NUM",
        "I-MENU.UNITPRICE",
        "I-MENU.CNT",
        "I-MENU.DISCOUNTPRICE",
        "I-MENU.PRICE",
        "I-MENU.ITEMSUBTOTAL",
        "I-MENU.VATYN",
        "I-MENU.ETC",
        "I-MENU.SUB.NM",
        "I-MENU.SUB.UNITPRICE",
        "I-MENU.SUB.CNT",
        "I-MENU.SUB.PRICE",
        "I-MENU.SUB.ETC",
        "I-VOID_MENU.NM",
        "I-VOID_MENU.PRICE",
        "I-SUB_TOTAL.SUBTOTAL_PRICE",
        "I-SUB_TOTAL.DISCOUNT_PRICE",
        "I-SUB_TOTAL.SERVICE_PRICE",
        "I-SUB_TOTAL.OTHERSVC_PRICE",
        "I-SUB_TOTAL.TAX_PRICE",
        "I-SUB_TOTAL.ETC",
        "I-TOTAL.TOTAL_PRICE",
        "I-TOTAL.TOTAL_ETC",
        "I-TOTAL.CASHPRICE",
        "I-TOTAL.CHANGEPRICE",
        "I-TOTAL.CREDITCARDPRICE",
        "I-TOTAL.EMONEYPRICE",
        "I-TOTAL.MENUTYPE_CNT",
        "I-TOTAL.MENUQTY_CNT",
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
class CORDConfig(FusionNERDatasetConfig):
    pass


class CORD(FusionNERDataset):
    """CORD dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        CORDConfig(
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
        for dir in ["train", "test", "dev"]:
            assert (
                Path(self.config.data_dir) / dir
            ).exists(), f"Data directory {self.config.data_dir} {dir} does not exist."

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "dev"},
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
        ann_dir = os.path.join(filepath, "json")
        image_dir = os.path.join(filepath, "image")

        data = []
        for fname in sorted(os.listdir(image_dir)):
            name, ext = os.path.splitext(fname)
            ann_path = os.path.join(ann_dir, name + ".json")
            image_path = os.path.join(image_dir, fname)
            with open(ann_path, "r", encoding="utf8") as f:
                annotation = json.load(f)
            image = PIL.Image.open(image_path)

            words = []
            bboxes = []
            labels = []
            for item in annotation["valid_line"]:
                cur_line_bboxes = []
                line_words, label = item["words"], item["category"]
                line_words = [w for w in line_words if w["text"].strip() != ""]
                if len(line_words) == 0:
                    continue
                if label == "other":
                    for w in line_words:
                        words.append(w["text"])
                        labels.append("O")
                        cur_line_bboxes.append(
                            normalize_bbox(self._quad_to_box(w["quad"]), image.size)
                        )
                else:
                    words.append(line_words[0]["text"])
                    label = label.upper().replace("MENU.SUB_", "MENU.SUB.")
                    labels.append("B-" + label)
                    cur_line_bboxes.append(
                        normalize_bbox(
                            self._quad_to_box(line_words[0]["quad"]), image.size
                        )
                    )
                    for w in line_words[1:]:
                        words.append(w["text"])
                        label = label.upper().replace("MENU.SUB_", "MENU.SUB.")
                        labels.append("I-" + label)
                        cur_line_bboxes.append(
                            normalize_bbox(self._quad_to_box(w["quad"]), image.size)
                        )
                if self.config.segment_level_layout:
                    cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)
            data.append(
                {
                    DataKeys.WORDS: words,
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
