"""
Defines the feature attribution generation task.
"""

from dataclasses import dataclass

import ignite.distributed as idist

from torchfusion.core.analyzer.tasks.base import AnalyzerTask
from torchfusion.core.analyzer.tasks.base_config import AnalyzerTaskConfig
from torchfusion.core.training.utilities.general import initialize_torch


class PrepareDataset(AnalyzerTask):
    @dataclass
    class Config(AnalyzerTaskConfig):
        visualize: bool = False

    def _setup_analysis(self, task_name: str):
        # initialize training
        initialize_torch(
            self._args,
            seed=self._args.general_args.seed,
            deterministic=self._args.general_args.deterministic,
        )

        # initialize torch device (cpu or gpu)
        self._device = idist.device()

        # get device rank
        self._rank = idist.get_rank()

    def setup(self, task_name: str):
        # setup training
        self._setup_analysis(task_name)

        # setup datamodule
        self._datamodule = self._setup_datamodule(stage=None)

        # setup dataloaders
        self._train_dataloader = self._datamodule.train_dataloader(
            self._args.data_args.data_loader_args.per_device_eval_batch_size,
            dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_args.data_loader_args.pin_memory,
        )
        self._test_dataloader = self._datamodule.test_dataloader(
            self._args.data_args.data_loader_args.per_device_eval_batch_size,
            dataloader_num_workers=self._args.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_args.data_loader_args.pin_memory,
        )

    def cleanup(self):
        pass

    def run(self):
        if self.config.visualize:
            # visualize data points
            if self._train_dataloader is not None:
                for batch in self._train_dataloader:
                    self._datamodule.show_batch(batch)
                    break

            # visualize data points
            if self._test_dataloader is not None:
                for batch in self._test_dataloader:
                    self._datamodule.show_batch(batch)
                    break
