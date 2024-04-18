import typing
from dataclasses import dataclass

from torch import nn

from torchfusion.core.models.utilities.data_collators import BatchToTensorDataCollator


@dataclass
class CollateFnDict:
    train: typing.Optional[typing.Callable] = BatchToTensorDataCollator()
    validation: typing.Optional[typing.Callable] = BatchToTensorDataCollator()
    test: typing.Optional[typing.Callable] = BatchToTensorDataCollator()

    def __getitem__(self, name):
        return self.__dict__[name]


@dataclass
class TransformsDict:
    train: typing.Optional[
        typing.Union[typing.Callable, typing.List[typing.Callable]]
    ] = None
    validation: typing.Optional[
        typing.Union[typing.Callable, typing.List[typing.Callable]]
    ] = None
    test: typing.Optional[
        typing.Union[typing.Callable, typing.List[typing.Callable]]
    ] = None

    def __getitem__(self, name):
        return self.__dict__[name]


@dataclass
class MetricsDict:
    train: typing.Optional[
        typing.Union[typing.Callable, typing.Mapping[str, typing.Callable]]
    ] = None
    validation: typing.Optional[
        typing.Union[typing.Callable, typing.Mapping[str, typing.Callable]]
    ] = None
    test: typing.Optional[
        typing.Union[typing.Callable, typing.Mapping[str, typing.Callable]]
    ] = None
    predict: typing.Optional[
        typing.Union[typing.Callable, typing.Mapping[str, typing.Callable]]
    ] = None

    def __getitem__(self, name):
        return self.__dict__[name]


@dataclass
class ModulesDict:
    train: typing.List[str] = None
    validation: typing.List[str] = None
    test: typing.List[str] = None

    def __getitem__(self, name):
        return self.__dict__[name]
