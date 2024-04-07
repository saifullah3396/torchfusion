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

"""DOCILE dataset"""


import base64
import dataclasses
import io
import json
import os
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import PIL
import pyarrow
import pyarrow_hotfix
from datasets import Dataset as ArrowDataset
from docile.dataset import KILE_FIELDTYPES, LIR_FIELDTYPES

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_ner_dataset import (
    FusionNERDataset,
    FusionNERDatasetConfig,
)
from torchfusion.core.data.text_utils.utilities import normalize_bbox

pyarrow_hotfix.uninstall()
pyarrow.PyExtensionType.set_auto_load(True)


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """DOCILE Dataset"""

_HOMEPAGE = "https://github.com/rossumai/docile"

_LICENSE = "Apache-2.0 license"

_ALL_LABELS = [
    "O-KILE",
    "O-LI",
    "O-LIR",
    "B-LI",
    "B-account_num",
    "B-amount_due",
    "B-amount_paid",
    "B-amount_total_gross",
    "B-amount_total_net",
    "B-amount_total_tax",
    "B-bank_num",
    "B-bic",
    "B-currency_code_amount_due",
    "B-customer_billing_address",
    "B-customer_billing_name",
    "B-customer_delivery_address",
    "B-customer_delivery_name",
    "B-customer_id",
    "B-customer_order_id",
    "B-customer_other_address",
    "B-customer_other_name",
    "B-customer_registration_id",
    "B-customer_tax_id",
    "B-date_due",
    "B-date_issue",
    "B-document_id",
    "B-iban",
    "B-line_item_amount_gross",
    "B-line_item_amount_net",
    "B-line_item_code",
    "B-line_item_currency",
    "B-line_item_date",
    "B-line_item_description",
    "B-line_item_discount_amount",
    "B-line_item_discount_rate",
    "B-line_item_hts_number",
    "B-line_item_order_id",
    "B-line_item_person_name",
    "B-line_item_position",
    "B-line_item_quantity",
    "B-line_item_tax",
    "B-line_item_tax_rate",
    "B-line_item_unit_price_gross",
    "B-line_item_unit_price_net",
    "B-line_item_units_of_measure",
    "B-line_item_weight",
    "B-order_id",
    "B-payment_reference",
    "B-payment_terms",
    "B-tax_detail_gross",
    "B-tax_detail_net",
    "B-tax_detail_rate",
    "B-tax_detail_tax",
    "B-vendor_address",
    "B-vendor_email",
    "B-vendor_name",
    "B-vendor_order_id",
    "B-vendor_registration_id",
    "B-vendor_tax_id",
    "E-LI",
    "I-LI",
    "I-account_num",
    "I-amount_due",
    "I-amount_paid",
    "I-amount_total_gross",
    "I-amount_total_net",
    "I-amount_total_tax",
    "I-bank_num",
    "I-bic",
    "I-currency_code_amount_due",
    "I-customer_billing_address",
    "I-customer_billing_name",
    "I-customer_delivery_address",
    "I-customer_delivery_name",
    "I-customer_id",
    "I-customer_order_id",
    "I-customer_other_address",
    "I-customer_other_name",
    "I-customer_registration_id",
    "I-customer_tax_id",
    "I-date_due",
    "I-date_issue",
    "I-document_id",
    "I-iban",
    "I-line_item_amount_gross",
    "I-line_item_amount_net",
    "I-line_item_code",
    "I-line_item_currency",
    "I-line_item_date",
    "I-line_item_description",
    "I-line_item_discount_amount",
    "I-line_item_discount_rate",
    "I-line_item_hts_number",
    "I-line_item_order_id",
    "I-line_item_person_name",
    "I-line_item_position",
    "I-line_item_quantity",
    "I-line_item_tax",
    "I-line_item_tax_rate",
    "I-line_item_unit_price_gross",
    "I-line_item_unit_price_net",
    "I-line_item_units_of_measure",
    "I-line_item_weight",
    "I-order_id",
    "I-payment_reference",
    "I-payment_terms",
    "I-tax_detail_gross",
    "I-tax_detail_net",
    "I-tax_detail_rate",
    "I-tax_detail_tax",
    "I-vendor_address",
    "I-vendor_email",
    "I-vendor_name",
    "I-vendor_order_id",
    "I-vendor_registration_id",
    "I-vendor_tax_id",
]

_KILE_LABELS = (
    [
        "O",
    ]
    + [f"B-{x}" for x in KILE_FIELDTYPES]
    + [f"I-{x}" for x in KILE_FIELDTYPES]
)


_LIR_LABELS = (
    [
        "O",
    ]
    + [f"B-{x}" for x in LIR_FIELDTYPES]
    + [f"I-{x}" for x in LIR_FIELDTYPES]
)


def convert_to_list(row):
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            row[k] = v.tolist()
            if isinstance(v[0], np.ndarray):
                row[k] = [x.tolist() for x in v]
    return row


def process(examples, config):
    import numpy as np

    all_labels = np.array(_ALL_LABELS)
    if config == "kile":
        label_map = _KILE_LABELS
    elif config == "lir":
        label_map = _LIR_LABELS

    labels_to_idx = dict(zip(label_map, range(len(label_map))))
    labels = examples["ner_tags"]
    updated_labels = []
    for idx, label in enumerate(labels):
        # get label id in original map
        indices = np.nonzero(label)[0]

        # get_kile_label
        sample_label = [x for x in all_labels[indices] if x in label_map]
        if len(sample_label) > 0:
            updated_labels.append(labels_to_idx[sample_label[0]])
        else:
            updated_labels.append(labels_to_idx[label_map[0]])

    updated = {
        DataKeys.WORDS: examples["tokens"],
        DataKeys.WORD_BBOXES: examples["bboxes"],
        DataKeys.LABEL: updated_labels,
        DataKeys.IMAGE: examples["img"],
    }
    return updated


@dataclasses.dataclass
class DOCILEConfig(FusionNERDatasetConfig):
    synthetic: bool = False


class DOCILE(FusionNERDataset):
    """DOCILE dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DOCILEConfig(
            name="kile",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            ner_labels=_KILE_LABELS,
            ner_scheme="",
            synthetic=False,
        ),
        DOCILEConfig(
            name="lir",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            ner_labels=_LIR_LABELS,
            ner_scheme="",
            synthetic=False,
        ),
    ]

    def _initialize_config(self):
        pass

    def _split_generators(self, dl_manager):
        if not self.config.synthetic:
            for dir in ["train_withImgs", "val_withImgs"]:
                assert (
                    Path(self.config.data_dir) / dir
                ).exists(), (
                    f"Data directory {self.config.data_dir} {dir} does not exist."
                )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "train_withImgs"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "val_withImgs"},
            ),
        ]

    def _load_dataset_to_pandas(self, filepath):
        if not self.config.synthetic:
            base_dataset = ArrowDataset.load_from_disk(filepath)
            base_dataset = base_dataset.map(
                partial(process, config=self.config.name), batched=False
            )
            base_dataset = base_dataset.remove_columns(
                ["id", "img", "tokens", "bboxes", "ner_tags"]
            )

            # for dataset
            data = base_dataset.to_dict()
            return pd.DataFrame.from_dict(data)

    def _generate_examples_impl(
        self,
        filepath,
    ):
        data = self._load_dataset_to_pandas(filepath)
        self._logger.info("Base dataset pandas dataframe loaded:")
        self._logger.info(data)
        try:
            data = data.apply(convert_to_list, axis=1)
            tokenized_data = self._preprocess_dataset(data)
            for idx, sample in enumerate(tokenized_data):
                if DataKeys.IMAGE in sample:
                    base64_decoded = base64.b64decode(sample["image"])
                    sample[DataKeys.IMAGE] = np.array(
                        PIL.Image.open(io.BytesIO(base64_decoded))
                    )
                yield idx, sample
        except Exception as e:
            print(e)
