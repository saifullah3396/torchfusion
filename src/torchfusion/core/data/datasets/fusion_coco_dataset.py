import collections
import dataclasses
import json
import os
from pathlib import Path
from typing import List, Optional

import datasets
import numpy as np
import PIL
import tqdm
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_dataset import (
    FusionDataset,
    FusionDatasetConfig,
)
from torchfusion.core.data.utilities.coco_utils import load_coco_json

# we do not use the logger from torchfusion, but the one from datasets here
logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class FusionCocoDatasetConfig(FusionDatasetConfig):
    """BuilderConfig for FusionCocoDataset"""

    category_names: Optional[List[str]] = None
    splits: List[str] = dataclasses.field(
        default_factory=lambda: ["train", "dev", "test"]
    )

    def __post_init__(self):
        assert (
            self.category_names is not None
        ), "category_names must be provided for Coco type datasets."


class FusionCocoDataset(FusionDataset):
    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=self._dataset_features(),
            homepage=self.config.homepage,
            citation=self.config.citation,
        )

    def _get_image_dir(self, split):
        return f"{self.config.data_dir}/{split}"

    def _get_json_path(self, split):
        return f"{self.config.data_dir}/{split}.json"

    def _dataset_features(self):
        # we prepare coco style features as output
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                DataKeys.IMAGE_FILE_PATH: datasets.features.Value("string"),
                DataKeys.IMAGE_WIDTH: datasets.Value("int32"),
                DataKeys.IMAGE_HEIGHT: datasets.Value("int32"),
            }
        )

        features[DataKeys.IMAGE] = datasets.Image()

        object_dict = {
            "category_id": datasets.ClassLabel(
                names=self.config.category_names,
                num_classes=len(self.config.category_names),
            ),
            "image_id": datasets.Value("string"),
            "id": datasets.Value("int64"),
            "area": datasets.Value("int64"),
            "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
            "bbox_mode": datasets.Value("int32"),
            "segmentation": datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"))
            ),
            "iscrowd": datasets.Value("bool"),
        }
        features["objects"] = [object_dict]
        return features

    def _dataset_features(self):
        features = datasets.Features(
            {
                DataKeys.IMAGE_ID: datasets.Value("int64"),
                DataKeys.IMAGE_WIDTH: datasets.Value("int32"),
                DataKeys.IMAGE_HEIGHT: datasets.Value("int32"),
                DataKeys.IMAGE_FILE_PATH: datasets.features.Value("string"),
                DataKeys.IMAGE: datasets.Image(decode=True),
            }
        )
        object_dict = {
            "category_id": datasets.ClassLabel(names=self.config.category_names),
            "area": datasets.Value("int64"),
            "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
            "bbox_mode": datasets.Value("int32"),
            "segmentation": datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"))
            ),
            "iscrowd": datasets.Value("bool"),
        }
        features["objects"] = [object_dict]
        return features

    def _image_info_to_example(self, image_info, split):
        image_file_path = os.path.join(
            self._get_image_dir(split), image_info["file_name"]
        )
        example = {
            "image_id": image_info["id"],
            DataKeys.IMAGE_WIDTH: image_info["width"],
            DataKeys.IMAGE_HEIGHT: image_info["height"],
            DataKeys.IMAGE_FILE_PATH: str(image_file_path),
        }
        if Path(image_file_path).exists():
            image = PIL.Image.open(image_file_path)
            example[DataKeys.IMAGE] = image
        return example

    def _get_image_to_ann_dict(self, json_path):
        with open(json_path, encoding="utf8") as f:
            annotation_data = json.load(f)
            image_info = annotation_data["images"]
            annotations = annotation_data["annotations"]
            image_id_to_annotations = collections.defaultdict(list)
            logger.info("Preparing annotations...")
            for annotation in tqdm.tqdm(annotations):
                image_id_to_annotations[annotation["image_id"]].append(annotation)
        return image_info, image_id_to_annotations

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        splits = []
        for split in self.config.splits:
            if split == "train":
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "split": "train",
                    },
                )
            elif split in ["val", "valid", "validation", "dev"]:
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "split": "val",
                    },
                )
            elif split == "test":
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "split": "test",
                    },
                )
            else:
                continue

            splits.append(dataset)
        return splits

    def _generate_examples_impl(
        self,
        split,
    ):
        """Yields examples as (key, example) tuples."""

        logger.info("Loading annotations file...")

        # check if image files exist and if not, raise a warning
        if not Path(self._get_image_dir(split)).exists():
            logger.warning(
                f"Image directory {self._get_image_dir(split)} does not exist. Loading data without images..."
            )

        # load all coco samples from json file
        samples_list, category_names = load_coco_json(
            json_file=self._get_json_path(split), image_root=self._get_image_dir(split)
        )

        assert (
            category_names == self.config.category_names
        ), "Category names do not match between dataset config and json file."

        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        def _sample_from_coco_sample(sample):
            image_file_path = self._get_image_dir(split) / sample["file_name"]
            image = np.array(PIL.Image.open(image_file_path))
            return {
                DataKeys.IMAGE_ID: sample["image_id"],
                DataKeys.IMAGE_WIDTH: sample["width"],
                DataKeys.IMAGE_HEIGHT: sample["height"],
                DataKeys.IMAGE_FILE_PATH: sample["file_name"],
                DataKeys.IMAGE: image,
            }

        for idx, coco_sample in enumerate(samples_list):
            sample = _sample_from_coco_sample(coco_sample)
            objects = []
            for annotation in coco_sample["annotations"]:
                if "bbox_mode" in annotation:
                    annotation["bbox_mode"] = annotation["bbox_mode"].value

                if "modified" in annotation:
                    del annotation["modified"]
                objects.append(annotation)
            sample["objects"] = objects

            yield idx, sample
