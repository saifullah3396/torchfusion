"""
Defines the scheduler factory.
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Optional

from torchfusion.core.training.sch.args import WDSchedulerArguments

if TYPE_CHECKING:
    import torch

    from torchfusion.core.training.sch.args import LRSchedulerArguments


class LRSchedulerFactory:
    @staticmethod
    def create(
        scheduler_args: LRSchedulerArguments,
        optimizer: torch.optim.Optimizer,
        num_training_steps: Optional[int] = None,
        num_warmup_steps: Optional[int] = None,
        num_epochs: Optional[int] = None,
    ):
        import torch
        from ignite.handlers import ReduceLROnPlateauScheduler

        from torchfusion.core.training.sch.constants import LRSchedulerType

        scheduler = None
        if scheduler_args.name == "":
            return scheduler

        if scheduler_args.name == LRSchedulerType.STEP_LR:  # tested
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_args.params)
        elif scheduler_args.name == LRSchedulerType.EXPONENTIAL_LR:  # tested
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args.params)
        elif scheduler_args.name == LRSchedulerType.REDUCE_LR_ON_PLATEAU:  # tested
            scheduler = ReduceLROnPlateauScheduler(optimizer, **scheduler_args.params)
        elif scheduler_args.name == LRSchedulerType.LAMBDA_LR:
            type = scheduler_args.params["type"]
            lambda_fn = None
            if type == "linear":

                def lr_lambda(current_step: int):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    return max(
                        0.0,
                        float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
                    )

                lambda_fn = lr_lambda

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
        elif scheduler_args.name == LRSchedulerType.COSINE_ANNEALING_LR:  # tested
            if not scheduler_args.restarts:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, num_training_steps, **scheduler_args.params
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, num_training_steps // num_epochs, **scheduler_args.params
                )
        elif scheduler_args.name == LRSchedulerType.CYCLIC_LR:  # not tested
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_args.params)
        elif scheduler_args.name == LRSchedulerType.POLYNOMIAL_DECAY_LR:  # not tested
            from torchfusion.core.training.sch.schedulers.poly_decay_lr import (
                PolyDecayLR,
            )

            params = copy(scheduler_args.params)
            max_decay_steps = params.pop("max_decay_steps")
            max_decay_steps = num_training_steps if max_decay_steps == -1 else max_decay_steps
            scheduler = PolyDecayLR(optimizer, max_decay_steps=max_decay_steps, **params)
        else:
            raise ValueError(f"Learning rate scheduler with the name [{scheduler_args.name}] is not supported!")

        return scheduler


class SchedulersDict(object):
    def __init__(self) -> None:
        self.d = {}

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)


class WdSchedulerFactory:
    @staticmethod
    def create(
        scheduler_args: WDSchedulerArguments,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        steps_per_epoch: int,
    ):
        pass

        from torchfusion.core.training.sch.constants import WDSchedulerType

        if scheduler_args.name == "":
            return None

        wd_schedulers = SchedulersDict()
        for target_group_name, target_group_params in scheduler_args.params.items():
            for group in optimizer.param_groups:
                group_name = group["name"]
                initial_wd = group["weight_decay"]

                if group_name == target_group_name or target_group_name == "all" and initial_wd > 0:
                    if scheduler_args.name == WDSchedulerType.COSINE:
                        from torchfusion.core.training.sch.schedulers.poly_decay_lr import (
                            CosineScheduler,
                        )

                        wd_schedulers.d[group_name] = CosineScheduler(
                            initial_wd,
                            target_group_params.wd_end,
                            num_epochs,
                            steps_per_epoch,
                            warmup_epochs=target_group_params.warmup_epochs,
                        )

        return wd_schedulers
