"""
Defines the base TrainValSampler class for defining training/validation split samplers.
"""

from dataclasses import dataclass

from torchfusion.utilities.dataclasses.abstract_dataclass import AbstractDataclass


@dataclass
class AnalyzerTaskConfig(AbstractDataclass):
    """
    Base task configuration.
    """

    data_split: str = "test"
