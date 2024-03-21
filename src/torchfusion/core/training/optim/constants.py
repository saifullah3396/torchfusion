"""
Defines the optimizer related constants.
"""
from enum import Enum


class OptimizerType(Enum):
    ADAM = "adam"
    ADAM_W = "adam_w"
    SGD = "sgd"
    RMS_PROP = "rmsprop"
    LARS = "lars"
