"""
Defines the factory for DataAugmentation class and its children.
"""


from __future__ import annotations

from typing import Type

from torchfusion.core.analyzer.args import AnalyzerArguments
from torchfusion.core.analyzer.tasks.base import AnalyzerTask
from torchfusion.core.args.args import FusionArguments
from torchfusion.utilities.module_import import ModuleLazyImporter


class AnalyzerTaskFactory:
    @staticmethod
    def get_class(analyzer_args: AnalyzerArguments) -> Type[AnalyzerTask]:
        """
        Find the model given the task and its name
        """

        if isinstance(analyzer_args.config, list):
            tasks = []
            for task, config in zip(analyzer_args.task, analyzer_args.config):
                cls = ModuleLazyImporter.get_analyzer_tasks().get(task, None)
                if cls is None:
                    raise ValueError(f"Analyzer task [{task}] is not supported.")
                cls = cls()
                task.append(cls)
            return tasks
        else:
            cls = ModuleLazyImporter.get_analyzer_tasks().get(analyzer_args.task, None)
            if cls is None:
                raise ValueError(f"Analyzer task [{analyzer_args.task}] is not supported.")
            cls = cls()
            return cls

    @staticmethod
    def create(
        args: FusionArguments,
        hydra_config,
    ):
        pass

        tasks = dict()
        for task_name, config in args.analyzer_args.tasks.items():
            task_type = config["task_type"]
            task_config = config["task_config"]

            cls = ModuleLazyImporter.get_analyzer_tasks().get(task_type, None)
            if cls is None:
                raise ValueError(
                    f"Analyzer task [{task_name}:{task_type}] is not supported. Possible choices = {ModuleLazyImporter.get_analyzer_tasks()}"
                )
            cls = cls()
            tasks[task_name] = cls(args, hydra_config, task_config)
        return tasks
