"""Holds the constants related to training."""


from typing import Any


class TrainingStage:
    train = "train"
    validation = "validation"
    test = "test"
    predict = "predict"
    visualization = "visualization"

    @classmethod
    def get(cls, name: str) -> Any:
        return getattr(cls, name)


class GANStage:
    train_gen = "train_gen"
    train_disc = "train_disc"
