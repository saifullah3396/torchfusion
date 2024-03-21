import dataclasses
from dataclasses import dataclass
from typing import Optional

from torchfusion.utilities.dataclasses.abstract_dataclass import AbstractDataclass


@dataclass
class ArgumentsBase(AbstractDataclass):
    def __repr__(self) -> str:
        """
        Pretty prints the arguments.
        """

        def print_recursive_args(d, depth=0):
            msg = ""
            for k, v in d.items():
                tabs = "  " * depth
                msg += f"{tabs}{k}: "
                if isinstance(v, dict):
                    msg += "\n"
                    msg += print_recursive_args(v, depth + 1)
                else:
                    msg += f"{v}\n"
            return msg

        # print arguments for verbosity
        msg = ""
        msg += print_recursive_args(dataclasses.asdict(self))
        return msg


@dataclass
class ClassInitializerArgs(ArgumentsBase):
    """
    Dataclass that holds the arguments related to any subclass.
    """

    # Class name
    name: str = ""

    # Strategy kwargs
    kwargs: Optional[dict] = None
