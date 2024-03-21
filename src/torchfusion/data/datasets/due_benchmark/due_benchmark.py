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

"""Due-Benchmark datasets"""


import collections
import csv
import json
import os
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import datasets
import numpy as np
import pandas as pd
import PIL
import torch
import tqdm
from benchmarker.data.reader import Corpus, qa_strategies
from benchmarker.data.reader.benchmark_dataset import BenchmarkDataset
from benchmarker.data.slicer import LongPageStrategy
from benchmarker.data.t5 import T5DownstreamDataConverter, data_instance_2_feature
from benchmarker.model import MODEL_CLASSES
from benchmarker.utils.training import load_tokenizer
from datasets.features import Image
from pdf2image import convert_from_path
from regex import F
from transformers.tokenization_utils_fast import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_dataset import FusionDataset
from torchfusion.core.data.datasets.fusion_dataset_config import FusionDatasetConfig
from torchfusion.core.data.datasets.fusion_image_text_layout_dataset import (
    FusionImageTextLayoutDataset,
    FusionImageTextLayoutDatasetConfig,
)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """https://github.com/due-benchmark/baselines"""

# You can copy an official description
_DESCRIPTION = """\
Due Benchmark is a benchmark for document understanding and analysis.
It contains a large-scale dataset and a set of standard evaluation metrics and baselines for document understanding
and analysis tasks, including document classification, named entity recognition, table structure recognition,
and table structure recognition.
"""

_HOMEPAGE = "https://github.com/due-benchmark/baselines"

_LICENSE = "MIT license"

_DATASETS = [
    "DocVQA",
    "PWC",
    "DeepForm",
    "TabFact",
    "WikiTableQuestions",
    "InfographicsVQA",
    "KleisterCharity",
]
IMAGE_SIZE_DIVISIBILITY = 64


def load_tokenizer(
    model_path: Path,
    model_type: str = "bert",
    do_lower_case: Optional[bool] = None,
    convert_to_fast_tokenizer: bool = False,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Loads BertTokenizer from Bert model directory.

    If `do_lower_case` is explicitly passed, tokenizer will be loaded using that value.
    Otherwise, it is looked up in model's config. If config doesn't contain this parameter,
    BertTokenizer is loaded using `transformers` default behaviour (which is
    checking model identifier for `-cased` or `-uncased` substrings).
    :param model_type: type of the architecture to use for loading i.e. "bert", "roberta"
    :param model_path: model path or identifier. If path, has to contain config.json
    :param do_lower_case: Optional boolean value. Controls BertTokenizer's `do_lower_case`.
    :return: BertTokenizer, RobertaTokenizer or T5Tokenizer
    """
    if do_lower_case is not None:
        tokenizer = MODEL_CLASSES[model_type]["tokenizer"].from_pretrained(
            str(model_path), do_lower_case=do_lower_case
        )
    else:
        config = MODEL_CLASSES[model_type]["config"].from_pretrained(str(model_path))
        if config is None:
            raise FileNotFoundError(
                f"Provided model or identifier {model_path} is not valid"
            )
        if hasattr(config, "do_lower_case"):
            tokenizer = MODEL_CLASSES[model_type]["tokenizer"].from_pretrained(
                str(model_path), do_lower_case=config.do_lower_case
            )
        else:
            tokenizer = MODEL_CLASSES[model_type]["tokenizer"].from_pretrained(
                str(model_path)
            )

    if not convert_to_fast_tokenizer or isinstance(tokenizer, PreTrainedTokenizerFast):
        return tokenizer
    return PreTrainedTokenizerFast(__slow_tokenizer=tokenizer)  # Dirty, but worth it


class DueBenchmarkConfig(FusionDatasetConfig):
    """BuilderConfig for DueBenchmark. This configuration is taken from generate_memmaps from the DueBenchmark library."""

    # tokenizer args
    model_path: str = "google-t5/t5-large"
    model_type: str = "t5"
    use_fast_tokenizer: bool = True
    max_encoder_length: int = 1024

    # corpus args
    unescape_prefix: bool = False
    unescape_values: bool = True
    use_prefix: bool = True
    prefix_separator: str = ":"
    values_separator: str = "|"
    single_property: bool = True
    use_none_answers: bool = False
    use_fast_tokenizer: bool = True
    limit: int = -1
    case_augmentation: bool = False
    segment_levels: tuple = (
        "tokens",
        "pages",
    )
    long_page_strategy: str = "FIRST_PART"
    ocr_engine: str = "tesseract"
    lowercase_expected: bool = False
    lowercase_input: bool = False
    train_strategy: str = "all_items"
    dev_strategy: str = "concat"
    test_strategy: str = "concat"
    augment_tokens_from_file: str = ""
    img_matrix_order: int = 0
    processes = 1
    imap_chunksize = 100
    skip_text_tokens = False

    # image loading config
    target_image_width: int = 1024
    target_image_height: int = 1024
    target_image_channels: int = 3


# #!/bin/bash

# DATASETS_ROOT="/home/ataraxia/Datasets/documents/DueBenchmark"
# TOKENIZER="/home/ataraxia/Datasets/documents/DueBenchmark/models/T5-large/"
# MAX_LENGTHS=(1024 6144 6144 1024 4096 1024 6144)
# TRAIN_STRATEGIES=(all_items concat all_items all_items concat all_items all_items)
# # DATASETS=(DocVQA PWC DeepForm TabFact WikiTableQuestions InfographicsVQA KleisterCharity)
# DATASETS=(DocVQA)
# OCRS=(microsoft_cv)


def validate_image_configuration(target_image_height, target_image_width):
    divisable_msg = f"should be divisable by {IMAGE_SIZE_DIVISIBILITY}"
    assert (
        target_image_width % IMAGE_SIZE_DIVISIBILITY == 0
    ), f"Incorect width: {target_image_width}, {divisable_msg}"
    assert (
        target_image_height % IMAGE_SIZE_DIVISIBILITY == 0
    ), f"Incorect max height size: {target_image_height}, {divisable_msg}"


def get_image(
    image_file_path,
    feature,
    target_image_height,
    target_image_width,
    target_image_channels,
):
    validate_image_configuration(
        target_image_height=target_image_height,
        target_image_width=target_image_width,
    )
    images = []
    if image_file_path:
        images.extend(
            read_real_images(
                image_file_path,
                feature,
                target_image_height=target_image_height,
                target_image_width=target_image_width,
                target_image_channels=target_image_channels,
            )
        )
    else:
        # do not waste memory for empty images and create 1px height image
        images.append(
            create_dummy_image(
                target_image_height=target_image_height,
                target_image_width=target_image_width,
                target_image_channels=target_image_channels,
            )
        )

    # simply to single image for usage in this case
    return images[0]


def read_real_images(
    image_file_path,
    feature,
    target_image_height,
    target_image_width,
    target_image_channels,
):
    mask = feature.seg_data["pages"]["masks"]
    num_pages = feature.seg_data["pages"]["ordinals"]
    page_sizes = feature.seg_data["pages"]["bboxes"]
    page_sizes = page_sizes[mask].tolist()
    page_lst = num_pages[mask].tolist()
    return [
        get_page_image(
            image_file_path,
            page_no,
            page_size,
            target_image_height=target_image_height,
            target_image_width=target_image_width,
            target_image_channels=target_image_channels,
        )
        for page_no, page_size in zip(page_lst, page_sizes)
    ]


def get_page_image(
    image_file_path,
    page_no,
    page_size,
    target_image_height,
    target_image_width,
    target_image_channels,
):
    page_path = image_file_path / f"{page_no}.png"
    if page_path.is_file():
        return load_image(
            page_path,
            target_image_height=target_image_height,
            target_image_width=target_image_width,
            target_image_channels=target_image_channels,
        )
    else:
        return create_dummy_image(
            target_image_height=target_image_height,
            target_image_width=target_image_width,
            target_image_channels=target_image_channels,
        )


def create_dummy_image(
    target_image_height, target_image_width, target_image_channels
) -> np.ndarray:
    arr_sz = (
        (target_image_height, target_image_width, 3)
        if target_image_channels == 3
        else (target_image_height, target_image_width)
    )
    return np.full(arr_sz, 255, dtype=np.uint8)


def load_image(
    page_path, target_image_height, target_image_width, target_image_channels
):
    image = Image.open(page_path)
    if image.mode != "RGB" and target_image_channels == 3:
        image = image.convert("RGB")
    if image.mode != "L" and target_image_channels == 1:
        image = image.convert("L")
    return np.array(image.resize((target_image_width, target_image_height)))


class DueBenchmark(FusionDataset):
    """DocBank dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DueBenchmarkConfig(
            name=dataset,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )
        for dataset in _DATASETS
    ]

    def _dataset_features(self):
        return datasets.Features(
            {
                DataKeys.IMAGE: Image(
                    decode=True
                ),  # a pdf can have multiple page images
                DataKeys.IMAGE_FILE_PATH: datasets.features.Value("string"),
                DataKeys.TOKEN_IDS: datasets.Sequence(datasets.Value(dtype="int32")),
                DataKeys.TOKEN_BBOXES: datasets.Sequence(
                    datasets.Sequence(datasets.Value("float32"), length=4)
                ),
                DataKeys.ATTENTION_MASKS: datasets.Sequence(
                    datasets.Value(dtype="uint8")
                ),
                DataKeys.TARGET_TOKEN_IDS: datasets.Sequence(
                    datasets.Value(dtype="int32")
                ),
            }
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": "dev"},
            ),
        ]

    def _create_example_from_feature(self, feature):
        # we remap all due benchmark keys to what we require in our datasets
        image_file_path = (
            Path(self.config.data_dir)
            / self.config.name
            / "pdfs"
            / (feature.doc_id + ".pdf")
        )
        example = {
            # this is how the due benchmark loads the images
            DataKeys.IMAGE: get_image(
                image_file_path,
                feature,
                target_image_height=self.config.target_image_height,
                target_image_width=self.config.target_image_width,
                target_image_channels=self.config.target_image_channels,
            ),
            # this is how the udop code due benchmark loads the images
            # DataKeys.IMAGE: convert_from_path(image_file_path)[
            #     0
            # ],  # we take the first image of each pdf
            DataKeys.IMAGE_FILE_PATH: str(image_file_path),
            DataKeys.TOKEN_IDS: feature.input_ids,
            DataKeys.TOKEN_BBOXES: feature.seg_data["tokens"]["bboxes"],
            DataKeys.ATTENTION_MASKS: feature.input_masks,
            DataKeys.TARGET_TOKEN_IDS: feature.lm_label_ids,
            DataKeys.TARGET_TOKEN_LABEL_STRING: feature.label_name,
        }
        return example

    def _generate_examples_impl(
        self,
        split,
    ):
        # load t5 tokenizer and data converter
        tokenizer = load_tokenizer(
            self.config.model_path,
            model_type=self.config.model_type,
            convert_to_fast_tokenizer=self.config.use_fast_tokenizer,
        )
        data_converter = T5DownstreamDataConverter(
            tokenizer,
            segment_levels=self.config.segment_levels,
            max_seq_length=self.config.max_encoder_length,
            long_page_strategy=LongPageStrategy(self.config.long_page_strategy),
            img_matrix_order=self.config.img_matrix_order,
            processes=self.config.processes,
            skip_text_tokens=self.config.skip_text_tokens,
        )
        corpus = Corpus(
            unescape_prefix=self.config.unescape_prefix,
            unescape_values=self.config.unescape_values,
            use_prefix=self.config.use_prefix,
            prefix_separator=self.config.prefix_separator,
            values_separator=self.config.values_separator,
            single_property=self.config.single_property,
            use_none_answers=self.config.use_none_answers,
            case_augmentation=self.config.case_augmentation,
            lowercase_expected=self.config.lowercase_expected,
            lowercase_input=self.config.lowercase_input,
            train_strategy=getattr(qa_strategies, self.config.train_strategy),
            dev_strategy=getattr(qa_strategies, self.config.dev_strategy),
            test_strategy=getattr(qa_strategies, self.config.test_strategy),
            augment_tokens_from_file=self.config.augment_tokens_from_file,
        )
        self._logger.info(
            f"Loading dataset from path {Path(self.config.data_dir) / self.config.name} with OCR engine {self.config.ocr_engine}"
        )
        benchmark_dataset = BenchmarkDataset(
            directory=Path(self.config.data_dir) / self.config.name,
            split=split,
            ocr=self.config.ocr_engine,
            segment_levels=self.config.segment_levels,
        )
        setattr(corpus, "_" + split, benchmark_dataset)
        subset = getattr(corpus, split)
        if subset is None:
            raise ValueError(f"Split {split} not found in corpus")

        # get tokenizer features
        features = data_converter.generate_features(subset)

        for idx, feature in enumerate(features):
            # print(feature.seg_data)
            if idx == self.config.limit:
                break
            if feature is None:
                continue
            example = self._create_example_from_feature(feature)
            # debugging
            # for k, v in example.items():
            #     print(k, type(v))
            #     if isinstance(v, np.ndarray):
            #         print(v.shape)
            #         print(v.dtype)
            yield idx, example
