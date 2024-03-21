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

"""DockBank dataset"""


import collections
import csv
import dataclasses
import json
import os
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import PIL
import tqdm

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_coco_dataset import (
    FusionCocoDataset, FusionCocoDatasetConfig)
from torchfusion.core.data.datasets.fusion_image_dataset import (
    FusionImageDataset, FusionImageDatasetConfig)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@misc{li2020docbank,
    title={DocBank: A Benchmark Dataset for Document Layout Analysis},
    author={Minghao Li and Yiheng Xu and Lei Cui and Shaohan Huang and Furu Wei and Zhoujun Li and Ming Zhou},
    year={2020},
    eprint={2006.01038},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

# You can copy an official description
_DESCRIPTION = """\
DocBank is a new large-scale dataset that is constructed using a weak supervision approach.
It enables models to integrate both the textual and layout information for downstream tasks.
The current DocBank dataset totally includes 500K document pages, where 400K for training, 50K for validation and 50K for testing.
"""

_HOMEPAGE = "https://doc-analysis.github.io/docbank-page/index.html"

_LICENSE = "Apache-2.0 license"

_CLASSES = [
    "abstract",
    "author",
    "caption",
    "equation",
    "figure",
    "footer",
    "list",
    "paragraph",
    "reference",
    "section",
    "table",
    "title",
    "date",
]


def create_coco_dict():
    coco_dict = {}
    coco_dict["info"] = {
        "year": 2020,
        "version": "1.0",
        "description": "The MS COCO format version of DocBank.",
        "contributor": "Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, Ming Zhou",
        "url": "https://github.com/doc-analysis/DocBank",
    }
    coco_dict["licenses"] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-NoDerivs License",
            "url": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
        }
    ]
    coco_dict["categories"] = [
        {"id": i, "name": _CLASSES[i].upper(), "supercategory": ""}
        for i in range(len(_CLASSES))
    ]
    coco_dict["images"] = []
    coco_dict["annotations"] = []

    return coco_dict


def create_coco_image_dict(file_name, image_id):
    image_path, image_name = os.path.split(file_name)
    image_dict = {
        "image_id": image_id,
        DataKeys.IMAGE_FILE_PATH: image_name,
        DataKeys.IMAGE_HEIGHT: "",
        DataKeys.IMAGE_WIDTH: "",
    }
    if os.path.exists(file_name):
        image = PIL.Image.open(file_name)
        image_dict[DataKeys.IMAGE_HEIGHT] = image.height
        image_dict[DataKeys.IMAGE_WIDTH] = image.width
        image_dict[DataKeys.IMAGE] = image
    return image_dict


def create_coco_object_dict(doc_object, doc_object_id, image_id):
    object_dict = {}
    object_dict["id"] = doc_object_id
    object_dict["image_id"] = image_id
    object_dict["iscrowd"] = 0
    object_dict["segmentation"] = []
    object_dict["text"] = doc_object[0]
    object_width = (doc_object[3] - doc_object[1],)
    object_height = (doc_object[4] - doc_object[2],)
    object_dict["bbox"] = [
        int(doc_object[1]),
        int(doc_object[2]),
        int(object_width[0]),
        int(object_height[0]),
    ]
    object_dict["area"] = int(object_width[0] * object_height[0])
    object_dict["category_id"] = _CLASSES.index(doc_object[9])
    object_dict["color"] = [int(doc_object[5]), int(doc_object[6]), int(doc_object[7])]
    object_dict["font"] = doc_object[8]
    return object_dict


def extract_archive(path):
    import tarfile

    root_path = path.parent
    folder_name = path.name.replace(".tar.gz", "")

    def extract_nonexisting(archive):
        for member in archive.members:
            name = member.name
            if not (root_path / folder_name / name).exists():
                archive.extract(name, path=root_path / folder_name)

    # print(f"Extracting {path.name} into {root_path / folder_name}...")
    with tarfile.open(path) as archive:
        extract_nonexisting(archive)


def folder_iterator(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            yield os.path.join(subdir, file)


@dataclasses.dataclass
class DocBankConfig(FusionCocoDatasetConfig):
    """BuilderConfig for DocBank."""

    pass


class DocBank(FusionCocoDataset):
    """DocBank dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DocBankConfig(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            category_names=_CLASSES,
        ),
    ]

    def _dataset_features(self):
        dataset_features = super()._dataset_features()
        dataset_features["objects"][0][DataKeys.COLOR] = datasets.Sequence(
            datasets.Sequence(datasets.Value("uint8"))
        )
        dataset_features["objects"][0][DataKeys.FONT] = datasets.Value("string")
        return dataset_features

    def _get_image_dir(self, split):
        return f"{self.config.data_dir}/DocBank_500K_ori_img/"

    def _get_ann_dir(self):
        return f"{self.config.data_dir}/DocBank_500K_txt/"

    def _get_json_path(self, split):
        return f"{self.config.data_dir}/{split}.json"

    def _generate_examples_impl(
        self,
        split,
    ):
        self._logger.info("Loading annotations file...")
        image_idx = 0
        doc_object_idx = 0
        ann_dir = Path(self.config.data_dir) / "DocBank_500K_txt"
        image_dir = Path(self.config.data_dir) / "DocBank_500K_ori_img"

        # check if image files exist and if not, raise a warning
        if not Path(image_dir).exists():
            self._logger.warning(
                f"Image directory {image_dir} does not exist. Loading data without images..."
            )

        with open(Path(self.config.data_dir) / f"{split}.jsonl", "rt") as fp:
            for file in fp:
                index, basename = eval(file)
                ann_file = ann_dir / f"{basename}.txt"
                image_file = image_dir / f"{basename}_ori.jpg"

                coco_dict = create_coco_dict()
                image_dict = create_coco_image_dict(
                    image_file,
                    image_idx,
                )
                ann_table = pd.read_table(
                    ann_file,
                    header=None,
                    names=[
                        "token",
                        "x0",
                        "y0",
                        "x1",
                        "y1",
                        "R",
                        "G",
                        "B",
                        "name",
                        "label",
                    ],
                    quoting=csv.QUOTE_NONE,
                    encoding="utf-8",
                )
                for doc_object in ann_table.values:
                    object_dict = create_coco_object_dict(
                        doc_object, doc_object_idx, image_idx
                    )
                    coco_dict["annotations"].append(object_dict)
                    doc_object_idx += 1
                image_idx += 1
                coco_dict["images"].append(image_dict)
                yield image_idx, coco_dict
