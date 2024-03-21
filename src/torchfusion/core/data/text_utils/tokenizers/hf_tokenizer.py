from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

from transformers import AutoTokenizer

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.text_utils.tokenizers.base import TorchFusionTokenizer
from torchfusion.core.data.text_utils.utilities import remove_keys, rename_key


def pad_sequences(sequences, padding_side, max_length, padding_elem):
    if padding_side == "right":
        return [seq + [padding_elem] * (max_length - len(seq)) for seq in sequences]
    else:
        return [[padding_elem] * (max_length - len(seq)) + seq for seq in sequences]


@dataclass
class HuggingfaceTokenizer(TorchFusionTokenizer):
    model_name: str = ""
    init_kwargs: dict = field(default_factory=lambda: {})
    call_kwargs: dict = field(default_factory=lambda: {})
    padding_required: bool = True
    padding_side: str = "right"
    pad_max_length: int = 512
    keys_to_add_on_overflow: Optional[List[str]] = field(
        default_factory=lambda: [DataKeys.IMAGE_FILE_PATH, DataKeys.IMAGE]
    )

    def __post_init__(self):

        self.default_init_kwargs = {
            "cache_dir": f"{os.environ['TORCH_FUSION_CACHE_DIR']}/.huggingface/",
            "local_files_only": False,
        }
        self.init_kwargs = {**self.default_init_kwargs, **self.init_kwargs}

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **self.init_kwargs
        )

        self.output_keys = [
            DataKeys.TOKEN_IDS,
            DataKeys.ATTENTION_MASKS,
            DataKeys.TOKEN_TYPE_IDS,
            DataKeys.TOKEN_BBOXES,
            DataKeys.LABEL,
            DataKeys.OVERFLOW_MAPPING,
            DataKeys.WORD_IDS,
        ]

    def _apply_tokenizer(self, sample: dict):
        data = sample[DataKeys.WORDS]

        self.default_call_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "stride": 0,
            "pad_to_multiple_of": 8,
        }
        self.call_kwargs = {**self.default_call_kwargs, **self.call_kwargs}

        fixed_kwargs = dict(
            is_split_into_words=True,
            return_overflowing_tokens=True,  # set some arguments that we need to stay fixed for our case
            return_token_type_ids=None,
            return_attention_mask=None,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            return_length=False,
            return_tensors=None,
            verbose=True,
        )

        if "layoutlm" in self.model_name:
            fixed_kwargs.pop("is_split_into_words")
            fixed_kwargs["boxes"] = sample[DataKeys.WORD_BBOXES]
            fixed_kwargs["word_labels"] = sample[DataKeys.LABEL]

        kwargs = {**fixed_kwargs, **self.call_kwargs}

        # tokenize the words
        if isinstance(data, dict):
            tokenized_data = self.tokenizer(
                **data,
                **kwargs,
            )
        else:
            tokenized_data = self.tokenizer(
                data,
                **kwargs,
            )
        return tokenized_data

    def _process_samples(self, samples: Union[dict, List[dict]]):
        tokenized_data = self._apply_tokenizer(samples)

        # remap ids
        rename_key(tokenized_data, "bbox", DataKeys.TOKEN_BBOXES)
        rename_key(tokenized_data, "labels", DataKeys.LABEL)

        # remove unneeded ids
        remove_keys(samples, [DataKeys.WORDS, DataKeys.WORD_BBOXES])

        # add word ids
        tokenized_data[DataKeys.WORD_IDS] = [
            tokenized_data.word_ids(i)
            for i in range(len(tokenized_data[DataKeys.TOKEN_IDS]))
        ]

        for k in self.output_keys:
            if k in tokenized_data:
                samples[k] = tokenized_data[k]

        return samples

    def __call__(self, samples: Union[dict, List[dict]]):
        tokenized_samples = self._process_samples(samples)

        # post process here we repete all additional ids if they are matched to the same original file
        # for example if we have 2 overflowed samples from the same original sample we need to repeat the image file path
        overflowed_data = {k: [] for k in self.keys_to_add_on_overflow}
        for batch_index in range(len(tokenized_samples[DataKeys.TOKEN_IDS])):
            org_batch_index = tokenized_samples[DataKeys.OVERFLOW_MAPPING][batch_index]
            for k in overflowed_data.keys():
                if k in samples:
                    overflowed_data[k].append(samples[k][org_batch_index])
        for k in overflowed_data.keys():
            tokenized_samples[k] = overflowed_data[k]

        # convert dict of lists to list of dicts
        tokenized_samples = [
            dict(zip(tokenized_samples, t)) for t in zip(*tokenized_samples.values())
        ]

        # pad the samples
        if self.padding_required:
            data_padder = DataPadder(
                self.padding_side,
                self.pad_max_length,
                self.tokenizer.pad_token_type_id,
                self.tokenizer.pad_token_type_id,
            )
            tokenized_samples = data_padder(tokenized_samples)

        return tokenized_samples


@dataclass
class DataPadder:
    def __init__(
        self,
        padding_side="right",
        max_length=512,
        pad_token_id=-100,
        pad_token_type_id=-100,
    ):
        self.padding_side = padding_side
        self.max_length = max_length

        self.data_padding_dict = {}

        # sequence keys dict
        if DataKeys.TOKEN_IDS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_IDS] = pad_token_id
        if DataKeys.TOKEN_TYPE_IDS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_TYPE_IDS] = pad_token_type_id
        if DataKeys.ATTENTION_MASKS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.ATTENTION_MASKS] = 0
        if DataKeys.LABEL not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.LABEL] = -100
        if DataKeys.TOKEN_BBOXES not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_BBOXES] = [0, 0, 0, 0]
        if DataKeys.TOKEN_ANGLES not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_ANGLES] = 0
        if DataKeys.POSITION_IDS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.POSITION_IDS] = pad_token_id

    def _pad_sample(self, sample: dict):
        for k, padding_elem in self.data_padding_dict.items():
            if k in sample:
                if isinstance(sample, list):
                    sample[k] = pad_sequences(
                        sample[k],
                        self.padding_side,
                        self.max_length,
                        padding_elem,
                    )
                else:
                    sample[k] = pad_sequences(
                        [sample[k]],
                        self.padding_side,
                        self.max_length,
                        padding_elem,
                    )[0]
        return sample

    def __call__(self, samples: Union[dict, List[dict]]):
        for idx, sample in enumerate(samples):
            samples[idx] = self._pad_sample(sample)

        return samples
