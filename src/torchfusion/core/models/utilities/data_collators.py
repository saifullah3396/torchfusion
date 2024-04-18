import numbers
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from torchfusion.core.constants import DataKeys
from torchfusion.utilities.logging import get_logger


def pad_sequences(sequences, padding_side, max_length, padding_elem):
    if padding_side == "right":
        if isinstance(sequences[0][0], list):
            padded_sequences = []
            for seq_list in sequences:
                padded_sequences.append([])
                for seq in seq_list:
                    padded_sequences[-1].append(
                        seq + [padding_elem] * (max_length - len(seq))
                    )
            return padded_sequences
        else:
            return [seq + [padding_elem] * (max_length - len(seq)) for seq in sequences]
    else:
        if isinstance(sequences[0][0], list):
            padded_sequences = []
            for seq_list in sequences:
                padded_sequences.append([])
                for seq in seq_list:
                    padded_sequences[-1].append(
                        [padding_elem] * (max_length - len(seq)) + seq
                    )
            return padded_sequences
        else:
            return [[padding_elem] * (max_length - len(seq)) + seq for seq in sequences]


def list_to_tensor(list_of_items, dtype=None):
    if isinstance(list_of_items, torch.Tensor):  # if it is already a tensor
        output = list_of_items
    elif isinstance(list_of_items[0], numbers.Number):  # if it is a list of number type
        output = torch.tensor(list_of_items)
    elif isinstance(list_of_items[0], torch.Tensor):  # if it is a list of torch tensors
        output = torch.stack(list_of_items)
    elif isinstance(list_of_items[0], np.ndarray):  # if it is a list of numpy arrays
        output = torch.from_numpy(np.array(list_of_items))
    elif isinstance(list_of_items[0], str):  # if it is a list of strings, leave it
        output = list_of_items
    elif isinstance(list_of_items, list):  # if it is a list of list
        output = torch.stack([list_to_tensor(l) for l in list_of_items])
    else:
        output = torch.tensor(list_of_items)
    output = output.to(dtype)
    return output


def feature_to_maybe_tensor(features, feature_key):
    try:
        if isinstance(features[0][feature_key], torch.Tensor):
            return torch.stack([sample[feature_key] for sample in features])
        elif isinstance(features[0][feature_key], np.ndarray):
            return torch.from_numpy(
                np.array([sample[feature_key] for sample in features])
            )
        elif isinstance(features[0][feature_key], list):  # if its a list
            if isinstance(features[0][feature_key][0], torch.Tensor):  # list of tensors
                return torch.stack([sample[feature_key] for sample in features])
            elif isinstance(
                features[0][feature_key][0], list
            ):  # list of lists, convert to tensor we assume all lists are of the same length and have numbers, this will break for other cases
                # possible use case [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
                return torch.stack(
                    [torch.tensor(sample[feature_key]) for sample in features]
                )
            elif isinstance(
                features[0][feature_key][0], str
            ):  # for strings we just let it pass. This could be for example a list of words
                return [sample[feature_key] for sample in features]
            else:
                return torch.tensor([sample[feature_key] for sample in features])
        elif isinstance(features[0][feature_key], str):
            return [sample[feature_key] for sample in features]
        else:
            return torch.tensor([sample[feature_key] for sample in features])
    except Exception as e:
        raise RuntimeError(
            f"Found a feature during batch collation that cannot be automatically converted to a tensor: {features[0][feature_key]}."
            f" Did you forget to set transforms on the input data?"
        )


@dataclass
class PassThroughCollator:
    """
    Data collator for converting data in the batch to a dictionary of pytorch tensors.
    """

    def __call__(self, features):
        batch = {}
        for k in features[0].keys():
            batch[k] = [sample[k] for sample in features]
        return batch


@dataclass
class BatchToTensorDataCollator:
    """
    Data collator for converting data in the batch to a dictionary of pytorch tensors.
    """

    allowed_keys: Optional[list] = None
    type_map: Optional[list] = None

    def __call__(self, features):
        batch = {}
        if isinstance(features[0], list):
            for idx, feature in enumerate(features):
                features[idx] = {k: [dic[k] for dic in feature] for k in feature[0]}

        keys = features[0].keys()
        for k in keys:
            if self.allowed_keys is not None and k not in self.allowed_keys:
                continue
            batch[k] = feature_to_maybe_tensor(features, feature_key=k)

        if len(batch.keys()) == 0:
            logger = get_logger()
            logger.warning(
                "Batch is empty after collation as no empty allowed_keys=[] was passed. "
                "If you wish to automatically collate batch, provide keys or set allowed_keys=None"
            )

        if self.type_map is not None:
            for k, dtype in self.type_map.items():
                if k in batch:
                    batch[k] = batch[k].to(dtype)

        return batch


@dataclass
class BaseSequenceDataCollator:
    data_key_type_map: dict = field(default_factory=lambda: {})
    data_padding_dict: dict = field(default_factory=lambda: {})
    tokenizer: Optional[PreTrainedTokenizer] = None
    tokenizer_call_kwargs: Optional[dict] = field(default_factory=lambda: {})
    tokenizer_apply_key: Optional[str] = None
    tokenizer_apply_dict_keys: Optional[list] = field(default_factory=lambda: [])
    return_tensors: str = "pt"

    def __post_init__(self) -> None:
        # sequence keys dict
        if DataKeys.TOKEN_IDS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_IDS] = self.tokenizer.pad_token_id
        if DataKeys.TOKEN_TYPE_IDS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_TYPE_IDS] = (
                self.tokenizer.pad_token_type_id
            )
        if DataKeys.ATTENTION_MASKS not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.ATTENTION_MASKS] = 0
        if DataKeys.TOKEN_BBOXES not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_BBOXES] = [0, 0, 0, 0]
        if DataKeys.TOKEN_ANGLES not in self.data_padding_dict:
            self.data_padding_dict[DataKeys.TOKEN_ANGLES] = 0

    def __call__(self, features):
        # both cannot be used at the same time
        assert not (
            self.tokenizer_apply_dict_keys is not None
            and self.tokenizer_apply_key is not None
        ), "Both tokenizer_apply_dict_keys and tokenizer_apply_key cannot be used at the same time"

        batch = {k: [dic[k] for dic in features] for k in features[0]}
        if self.tokenizer_apply_dict_keys is not None:
            # generate input from our own keys
            tokenizer_input = {
                mapped_key: batch[key]
                for mapped_key, key in self.tokenizer_apply_dict_keys.items()
            }

            tokenizer_output = self.tokenizer(
                **tokenizer_input,
                **self.tokenizer_call_kwargs,
                return_tensors=self.return_tensors,
            )

            # fix the keys
            if "bbox" in tokenizer_output.keys():
                tokenizer_output[DataKeys.TOKEN_BBOXES] = tokenizer_output["bbox"]
                del tokenizer_output["bbox"]
        else:
            tokenizer_output = self.tokenizer(
                batch[self.tokenizer_apply_key],
                **self.tokenizer_call_kwargs,
                return_tensors=self.return_tensors,
            )

        for k in tokenizer_output:
            batch[k] = tokenizer_output[k]

        # convert all objects in batch to torch tensors
        filtered_batch = {}
        for k, v in batch.items():
            if k not in self.data_key_type_map.keys():  # filter out uneeded items
                continue
            filtered_batch[k] = list_to_tensor(v, dtype=self.data_key_type_map[k])
        return filtered_batch
