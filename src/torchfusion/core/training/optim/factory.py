"""
Defines the optimizer factory.
"""

from torchfusion.core.training.optim.args import OptimizerArguments
from torchfusion.core.training.optim.constants import OptimizerType


class OptimizerFactory:
    @staticmethod
    def create(
        args: OptimizerArguments,
        model_param_groups: dict,
        bypass_params_creation=False,
    ):
        from copy import copy

        import torch

        if not bypass_params_creation:
            model_parameters = []
            group_params = copy(args.group_params)
            if not isinstance(args.group_params, list):
                group_params = [args.group_params]
            for group_params_config in group_params:
                group = group_params_config.group_name
                if group in model_param_groups.keys():
                    model_parameters.append({})
                    model_parameters[-1]["params"] = model_param_groups[group]
                    model_parameters[-1]["name"] = group
                    for k, v in group_params_config.kwargs.items():
                        model_parameters[-1][k] = v
        else:
            model_parameters = model_param_groups

        # here we take the learning rate and other parameters of only first group
        # although in the backend these parameters are already set seperately for
        # each group and would work separately
        base_kwargs = copy(args.group_params[0].kwargs)

        opt = None
        if args.name == OptimizerType.ADAM:
            opt = torch.optim.Adam(model_parameters, **base_kwargs)
        elif args.name == OptimizerType.ADAM_W:
            opt = torch.optim.AdamW(model_parameters, **base_kwargs)
        elif args.name == OptimizerType.SGD:
            opt = torch.optim.SGD(model_parameters, **base_kwargs)
        elif args.name == OptimizerType.RMS_PROP:
            opt = torch.optim.RMSprop(model_parameters, **base_kwargs)
        elif args.name == OptimizerType.LARS:
            from torchfusion.core.training.optim.optimizers.lars import LARS

            if "lars_exclude" in base_kwargs:
                base_kwargs.pop("lars_exclude")
            opt = LARS(model_parameters, **base_kwargs)
        if opt is None:
            raise ValueError(f"Optimizer {args.name} is not supported!")

        # assign name to optimizer
        opt.name = args.name

        # set the learning rate scale if lr_scale is available per grop
        for param_group in opt.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] *= param_group["lr_scale"]

        return opt
