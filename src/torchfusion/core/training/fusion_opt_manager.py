from __future__ import annotations

from typing import TYPE_CHECKING

from torchfusion.utilities.logging import get_logger

if TYPE_CHECKING:
    from torchfusion.core.args.args import FusionArguments
    from torchfusion.core.models.fusion_model import FusionModel
    from torchfusion.core.training.fusion_trainer import FusionTrainer


class FusionOptimizerManager:
    def __init__(self, args: FusionArguments, model: FusionModel, trainer: FusionTrainer) -> None:
        self._args = args
        self._model = model
        self._trainer = trainer
        self._optimizers = None
        self._logger = get_logger()

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def param_groups(self):
        return self._model.torch_model.get_param_groups()

    def _setup_optimizers(self):
        from torchfusion.core.training.optim.factory import OptimizerFactory

        # get model parameter groups
        model_param_groups = self.param_groups

        # setup optimizers dictionary
        optimizers = {}
        for k, args in self._args.training_args.optimizers.items():
            if k not in model_param_groups.keys():
                raise ValueError(
                    f"Your optimizer configuration does not align the model optimizer "
                    f"parameter groups. {k} =/= {model_param_groups.keys()}"
                )

            # set optimizers
            if self._args.model_args.config.bypass_params_creation:
                optimizers[k] = OptimizerFactory.create(
                    args=args,
                    model_param_groups=model_param_groups[k],
                    bypass_params_creation=self._args.model_args.config.bypass_params_creation,
                )
            else:
                optimizers[k] = OptimizerFactory.create(
                    args=args,
                    model_param_groups=model_param_groups,
                    bypass_params_creation=self._args.model_args.config.bypass_params_creation,
                )

        return optimizers

    def setup(self):
        import ignite.distributed as idist

        # setup optimizers
        self._optimizers = self._setup_optimizers()
        for k, opt in self._optimizers.items():
            self._optimizers[k] = idist.auto_optim(opt)

        # print information about optimizers and schedulers
        self._pretty_print()

    def _pretty_print(self):
        import ignite.distributed as idist

        from torchfusion.utilities.general import indent_string

        if idist.get_rank() == 0:
            # print information
            msg = f"Configured optimizers:\n"
            for k, v in self._optimizers.items():
                opt_str = indent_string(str(v), " " * 4)
                msg += f"{k}:\n"
                msg += f"{opt_str}\n"
            self._logger.info(msg)

    def get_checkpoint_state_dict(self):
        checkpoint = {}

        # add optimizers to state
        if self._optimizers is not None:
            for k, opt in self._optimizers.items():
                checkpoint[f"opt_{k}"] = opt

        return checkpoint
