"""
Defines the scheduler arguments
"""
from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class LRSchedulerArguments:
    # Name of the lr scheduler to use. If none
    name: str = ""
    params: dict = field(default_factory=lambda: {})
    restarts: bool = False

    def __post_init__(self):
        from torchfusion.core.training.sch.constants import LRSchedulerType

        if self.name != "":
            self.name = LRSchedulerType(self.name)


@dataclass
class WDGroupParams:
    warmup_epochs: int = 0
    wd_end: float = 0.001


@dataclass
class WDSchedulerArguments:
    name: str = ""
    params: Mapping[str, WDGroupParams] = field(default_factory=lambda: {"all": WDGroupParams()})

    def __post_init__(self):
        from torchfusion.core.training.sch.constants import WDSchedulerType

        if self.name != "":
            self.name = WDSchedulerType(self.name)
