"""
Defines the scheduler related constants.
"""
from enum import Enum


class LRSchedulerType(Enum):
    LAMBDA_LR = "lambda_lr"
    STEP_LR = "step_lr"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_ANNEALING_LR = "cosine_annealing_lr"
    CYCLIC_LR = "cyclic_lr"
    POLYNOMIAL_DECAY_LR = "poly_decay_lr"
    EXPONENTIAL_LR = "exponential_lr"


class WDSchedulerType(Enum):
    COSINE = "cosine"
