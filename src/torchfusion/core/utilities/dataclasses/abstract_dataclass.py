"""
Defines dataclasses related utility functions/classes.
"""

from abc import ABC
from dataclasses import dataclass


@dataclass
class AbstractDataclass(ABC):
    def __new__(cls, *args, **kwargs):
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError(f"Cannot instantiate abstract class: [{cls.__name__}].")
        return super().__new__(cls)

    @classmethod
    def get_subclasses(cls):
        return tuple(cls.__subclasses__())


__all__ = ["get_fields", "from_dict", "Config"]
