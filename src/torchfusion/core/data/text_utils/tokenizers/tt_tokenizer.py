from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.text_utils.tokenizers.base import TorchFusionTokenizer


@dataclass
class TorchTextTokenizer(TorchFusionTokenizer):
    tokenizer_name: str
    vocab_file: Optional[str] = None

    def __post_init__(self):
        self._tokenizer = get_tokenizer(self.tokenizer_name)

        # initialize vocabulary
        self._vocab = None
        if self.vocab_file is not None:
            self.vocab_file = Path(self.vocab_file)
            if not self.vocab_file.exists():
                return ValueError(
                    f"Vocabulary file [{self.vocab_file}] does not exist."
                )

            if self.vocab_file.suffix == ".pth":
                self.vocab = torch.load(self.vocab_file)
            elif (
                self.vocab_file.suffix == ".pickle"
            ):  # generate vocabulary if it is not already present

                with open(self.vocab_file.suffix, "rb") as f:
                    data = pickle.load(f)
                self._vocab = self.generate_vocab_from_data(data)
            else:
                return ValueError(
                    f"Vocabulary file of type [{self.vocab_file.suffix}] is not supported."
                )

    def yield_tokens(self, data_iter: Iterable):
        for text in data_iter:
            yield self._tokenizer(text)

    def generate_vocab_from_data(self, data: Union[Iterable, List]):
        vocab = build_vocab_from_iterator(self.yield_tokens(data), specials=["<unk>"])
        vocab.set_default_index(self._vocab["<unk>"])
        return vocab

    def __call__(self, sample: dict):
        sample[DataKeys.TOKEN_IDS] = self._vocab(
            self._tokenizer(sample[DataKeys.WORDS])
        )
        return sample
