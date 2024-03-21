"""
Defines the dataclass for holding analyzer arguments.
"""

import os
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Union

from torchfusion.core.args.args_base import ArgumentsBase
from torchfusion.utilities.module_import import ModuleLazyImporter


@dataclass
class AnalyzerArguments(ArgumentsBase):
    """
    Dataclass that holds the analyzer related arguments.
    """

    analyzer_output_dir: str = (
        f"{os.environ.get('TORCH_FUSION_OUTPUT_DIR', './output/')}/analyzer",
    )
    model_checkpoints: Optional[Union[List[List[str]], List[str]]] = ""
    tasks: Union[Mapping[str, dict], List[Mapping[str, dict]]] = field(
        default=None,
    )

    def __post_init__(self):
        if self.tasks is not None:
            for task_name, config in self.tasks.items():
                if "task_type" not in config.keys():
                    raise ValueError(
                        f"Analyzer config task item [{task_name}] must have 'task_type' argument "
                        f"from the following choices: {[e for e in ModuleLazyImporter.get_analyzer_tasks().keys()]}"
                    )
                if "task_config" not in config.keys():
                    raise ValueError(
                        f"Analyzer config task item [{task_name}] must have 'task_config' argument with task config dict."
                    )
