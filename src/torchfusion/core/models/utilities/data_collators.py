import numbers
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from detectron2.structures import Instances
from torchfusion.core.data.text_utils.tokenizers.hf_tokenizer import (
    HuggingfaceTokenizer,
)
from torchfusion.core.utilities.logging import get_logger


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
            tensor_list = [sample[feature_key] for sample in features]
            tensor_shapes = [t.shape for t in tensor_list]
            if tensor_shapes.count(tensor_shapes[0]) == len(
                tensor_shapes
            ):  # all tensors have equal shape, we make a batch tensor
                return torch.stack(tensor_list)
            else:  # otherwise we return a list of tensors, this is not tested!
                return tensor_list
        elif isinstance(features[0][feature_key], np.ndarray):
            array_list = [sample[feature_key] for sample in features]
            array_shapes = [a.shape for a in array_list]
            if array_shapes.count(array_shapes[0]) == len(
                array_shapes
            ):  # all tensors have equal shape, we make a batch tensor
                return torch.from_numpy(np.array(array_list))
            else:  # otherwise we return a list of tensors, this is not tested!
                return [torch.from_numpy(a) for a in array_list]
            return
        elif isinstance(features[0][feature_key], list):  # if its a list
            if isinstance(features[0][feature_key][0], torch.Tensor):  # list of tensors
                return torch.stack([sample[feature_key] for sample in features])
            elif isinstance(
                features[0][feature_key][0], list
            ):  # list of lists, convert to tensor we assume all lists are of the same length and have numbers, this will break for other cases
                # possible use case [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
                # see if all tensors have same size
                size_check_fail = False
                for sample in features:
                    if len(sample[feature_key]) != len(features[0][feature_key]):
                        size_check_fail = True
                        break
                if size_check_fail:
                    return [torch.tensor(sample[feature_key]) for sample in features]
                else:
                    return torch.stack(
                        [torch.tensor(sample[feature_key]) for sample in features]
                    )
            elif isinstance(
                features[0][feature_key][0], str
            ):  # for strings we just let it pass. This could be for example a list of words
                return [sample[feature_key] for sample in features]
            elif (
                None in features[0][feature_key]
            ):  # for strings we just let it pass. This could be for example a list of words
                return [sample[feature_key] for sample in features]
            else:
                # see if all tensors have same size
                size_check_fail = False
                for sample in features:
                    if len(sample[feature_key]) != len(features[0][feature_key]):
                        size_check_fail = True
                        break
                if size_check_fail:
                    return [sample[feature_key] for sample in features]
                else:
                    return torch.tensor([sample[feature_key] for sample in features])
        elif isinstance(features[0][feature_key], (str, Instances)):
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

    return_batch_dict: bool = True

    def __call__(self, features):
        if self.return_batch_dict:
            batch = {}
            for k in features[0].keys():
                batch[k] = [sample[k] for sample in features]
            return batch
        else:
            return features


@dataclass
class BatchToTensorDataCollator:
    """
    Data collator for converting data in the batch to a dictionary of pytorch tensors.
    """

    allowed_keys: Optional[list] = None
    data_key_type_map: Optional[list] = None

    def __call__(self, features):
        batch = {}
        if isinstance(features[0], list):
            for idx, feature in enumerate(features):
                features[idx] = {k: [dic[k] for dic in feature] for k in feature[0]}

        # by default we make sure to always include INDEX in tensor collation as it is useful for storing data
        if DataKeys.INDEX not in self.allowed_keys:
            self.allowed_keys.append(DataKeys.INDEX)
            self.data_key_type_map[DataKeys.INDEX] = torch.long

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

        if self.data_key_type_map is not None:
            for k, dtype in self.data_key_type_map.items():
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(dtype)
        return batch


@dataclass
class SequenceTokenizerDataCollator:
    data_key_type_map: dict = field(default_factory=lambda: {})
    tokenizer: Optional[HuggingfaceTokenizer] = None
    keys_to_add_on_overflow: Optional[str] = None
    return_tensors: str = "pt"
    overflow_sampling: str = "return_all"

    def __call__(self, features):
        batch = {}
        keys = features[0].keys()
        for k in keys:
            batch[k] = [sample[k] for sample in features]

        # update tokenizer args here if required
        self.tokenizer.keys_to_add_on_overflow = self.keys_to_add_on_overflow
        self.tokenizer.overflow_sampling = self.overflow_sampling

        tokenizer_output = self.tokenizer(batch, return_dict=True)

        for k in tokenizer_output:
            batch[k] = tokenizer_output[k]

        # convert all objects in batch to torch tensors
        filtered_batch = {}
        for k, v in batch.items():
            if k not in self.data_key_type_map.keys():  # filter out uneeded items
                continue
            filtered_batch[k] = list_to_tensor(v, dtype=self.data_key_type_map[k])
        return filtered_batch
