"""
Defines the base Tokenizer class for defining custom tokenizers.
"""

from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class TorchFusionTokenizer:
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
