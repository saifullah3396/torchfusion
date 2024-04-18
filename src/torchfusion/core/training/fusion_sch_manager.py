from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import torch

from torchfusion.core.utilities.logging import get_logger

if TYPE_CHECKING:
    import torch

    from torchfusion.core.args.args import FusionArguments
    from torchfusion.core.training.fusion_opt_manager import FusionOptimizerManager
    from torchfusion.core.training.fusion_trainer import FusionTrainer


class FusionSchedulersManager:
    def __init__(
        self,
        args: FusionArguments,
        optimizer_manager: FusionOptimizerManager,
        trainer: FusionTrainer,
    ) -> None:
        self._args = args
        self._trainer = trainer
        self._optimizers = optimizer_manager.optimizers
        self._lr_schedulers = None
        self._wd_schedulers = None
        self._logger = get_logger()

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def lr_schedulers(self):
        return self._lr_schedulers

    @property
    def wd_schedulers(self):
        return self._wd_schedulers

    def _setup_lr_schedulers(self, optimizers: Mapping[str, torch.optim.Optimizer]):
        from torchfusion.core.training.sch.factory import LRSchedulerFactory

        # configure schedulers
        lr_schedulers = {}
        for k, sch in self._args.training_args.lr_schedulers.items():
            lr_schedulers[k] = LRSchedulerFactory.create(
                sch,
                optimizers[k],
                self._trainer.total_training_steps,
                self._trainer.warmup_steps,
                self._args.training_args.max_epochs,
            )
        return lr_schedulers

    def _setup_wd_schedulers(self, optimizers: Mapping[str, torch.optim.Optimizer]):
        from torchfusion.core.training.sch.factory import WdSchedulerFactory

        # configure schedulers
        wd_schedulers = {}
        for k, sch in self._args.training_args.wd_schedulers.items():
            wd_schedulers[k] = WdSchedulerFactory.create(
                sch,
                optimizers[k],
                self._args.training_args.max_epochs,
                self._trainer.steps_per_epoch,
            )
        return wd_schedulers

    def setup(self, init_schedulers: bool = True):
        if init_schedulers:
            # setup learning rate schedulers
            if self._args.training_args.lr_schedulers is not None:
                self._lr_schedulers = self._setup_lr_schedulers(
                    self._optimizers,
                )

            # setup weight decay schedulers
            if self._args.training_args.wd_schedulers is not None:
                self._wd_schedulers = self._setup_wd_schedulers(
                    self._optimizers,
                )

        # print information about optimizers and schedulers
        self._pretty_print()

    def _pretty_print(self):
        import ignite.distributed as idist

        if idist.get_rank() == 0:
            # print information
            if self._lr_schedulers is not None:
                msg = f"Configured learning rate schedulers: \n"
                for k, v in self._lr_schedulers.items():
                    msg += f"{k}:"
                    msg += " " * 4 + v.__class__.__name__ + "\n"
                self._logger.info(msg)
            if self._wd_schedulers is not None:
                msg = f"Configured weight decay schedulers:\n"
                for k, v in self._wd_schedulers.items():
                    msg += f"{k}: {v.__class__.__name__}\n"
                self._logger.info(msg)

    def get_checkpoint_state_dict(self):
        checkpoint = {}

        # add lr schedulers to state
        if self._lr_schedulers is not None:
            for k, sch in self._lr_schedulers.items():
                if sch is None:
                    continue
                checkpoint[f"lr_sch_{k}"] = sch

        # add wd schedulers to state
        if self._wd_schedulers is not None:
            for k, sch in self._wd_schedulers.items():
                if sch is None:
                    continue
                checkpoint[f"wd_sch_{k}"] = sch

        return checkpoint
