# import numbers
# from typing import List, Optional, Union

# from ignite.handlers import (
#     ConcatScheduler,
#     LRScheduler,
#     ParamGroupScheduler,
#     ParamScheduler,
#     PiecewiseLinear,
# )
# from torch.optim.lr_scheduler import _LRScheduler, LRScheduler
from ignite.handlers import create_lr_scheduler_with_warmup

# def create_lr_scheduler_with_warmup(
#     lr_scheduler: Union[ParamScheduler, _LRScheduler, LRScheduler],
#     warmup_start_value: float,
#     warmup_duration: int,
#     warmup_end_value: Optional[float] = None,
#     save_history: bool = False,
#     output_simulated_values: Optional[List] = None,
# ) -> "ConcatScheduler":
#     if not isinstance(lr_scheduler, (ParamScheduler, _LRScheduler, LRScheduler)):
#         raise TypeError(
#             "Argument lr_scheduler should be a subclass of torch.optim.lr_scheduler._LRScheduler or "
#             f"ParamScheduler, but given {type(lr_scheduler)}"
#         )

#     if not isinstance(warmup_duration, numbers.Integral):
#         raise TypeError(f"Argument warmup_duration should be integer, but given {warmup_duration}")

#     if not (warmup_duration > 1):
#         raise ValueError(f"Argument warmup_duration should be at least 2 events, but given {warmup_duration}")

#     warmup_schedulers = []  # type: List[ParamScheduler]

#     for param_group_index, param_group in enumerate(lr_scheduler.optimizer.param_groups):
#         if warmup_end_value is None:
#             param_group_warmup_end_value = param_group["lr"]
#         else:
#             param_group_warmup_end_value = warmup_end_value

#         milestones_values = [
#             (0, warmup_start_value),
#             (warmup_duration - 1, param_group_warmup_end_value),
#         ]

#         if isinstance(lr_scheduler, _LRScheduler):
#             init_lr = param_group["lr"]
#             if init_lr != param_group_warmup_end_value:
#                 milestones_values.append((warmup_duration, init_lr))

#             # We need to advance torch lr_scheduler to avoid duplicated lr value
#             # given by PiecewiseLinear and LRScheduler.
#             # We suggest to attach output scheduler on ITERATION_STARTED but
#             # torch lr_scheduler works with ITERATION_COMPLETED
#             # See also https://github.com/pytorch/ignite/pull/2496#issuecomment-1065984440
#             lr_scheduler.last_epoch += 1
#             lr_scheduler = LRScheduler(lr_scheduler, save_history=save_history)
#         # else:
#         #     init_lr = lr_scheduler.get_param()
#         #     if init_lr == param_group_warmup_end_value:
#         #         if warmup_duration > 2:
#         #             d = (param_group_warmup_end_value - warmup_start_value) / (warmup_duration - 1)
#         #             milestones_values[-1] = (warmup_duration - 2, param_group_warmup_end_value - d)
#         #         else:
#         #             milestones_values.pop(-1)

#         warmup_schedulers.append(
#             PiecewiseLinear(
#                 lr_scheduler.optimizer,
#                 param_name="lr",
#                 milestones_values=milestones_values,
#                 param_group_index=param_group_index,
#                 save_history=save_history,
#             )
#         )

#     warmup_scheduler = ParamGroupScheduler(warmup_schedulers, save_history=save_history)
    
#     schedulers = [
#         warmup_scheduler,
#         lr_scheduler,
#     ]  # type: List[Union[ParamScheduler, ParamGroupScheduler, _LRScheduler]]

#     durations = [milestones_values[-1][0] + 1]
#     combined_scheduler = ConcatScheduler(schedulers, durations=durations, save_history=save_history)

#     if output_simulated_values is not None:
#         if not isinstance(output_simulated_values, list):
#             raise TypeError(
#                 "Argument output_simulated_values should be a list of None, e.g. `[None] * 100`, "
#                 f"but given {type(output_simulated_values)}."
#             )
#         num_events = len(output_simulated_values)
#         result = ConcatScheduler.simulate_values(num_events=num_events, schedulers=schedulers, durations=durations)
#         for i in range(num_events):
#             output_simulated_values[i] = result[i]
#     return combined_scheduler
