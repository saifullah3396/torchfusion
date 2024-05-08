from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Union

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, EventEnum
from torchfusion.core.args.args import FusionArguments
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.functionality.default import DefaultTrainingFunctionality
from torchfusion.core.training.fusion_opt_manager import FusionOptimizerManager
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.utilities.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torchfusion.core.models.fusion_model import FusionModel

logger = get_logger(__name__)


def log_training_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Training time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


def log_eval_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


class OptimizerEvents(EventEnum):
    OPTIMIZER_STEP_CALLED = "optimizer_step_called"


class ObjectDetectionTrainingFunctionality(DefaultTrainingFunctionality):
    @classmethod
    def initialize_training_engine(
        cls,
        args: FusionArguments,
        model: FusionModel,
        opt_manager: FusionOptimizerManager,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        scaler: Optional["torch.cuda.amp.GradScaler"] = None,
        tb_logger: TensorboardLogger = None,
    ) -> Callable:
        return super().initialize_training_engine(
            args,
            model,
            opt_manager,
            device,
            scaler,
            tb_logger,
            # for detection since we use detectron we let it handle the device assignment automatically
            put_batch_to_device=False,
        )

    @classmethod
    def initialize_validation_engine(
        cls,
        args: FusionArguments,
        model: FusionModel,
        training_engine: Engine,
        output_dir: str,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        tb_logger: TensorboardLogger = None,
    ) -> Callable:
        return super().initialize_validation_engine(
            args,
            model,
            training_engine,
            output_dir,
            device,
            tb_logger,
            # for detection since we use detectron we let it handle the device assignment automatically
            put_batch_to_device=False,
        )

    @classmethod
    def initialize_prediction_engine(
        cls,
        args: FusionArguments,
        model: FusionModel,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        tb_logger: TensorboardLogger = None,
        keys_to_device: list = [],
    ) -> Callable:
        return super().initialize_prediction_engine(
            # for detection since we use detectron we let it handle the device assignment automatically
            args,
            model,
            device,
            tb_logger,
            keys_to_device,
            put_batch_to_device=False,
        )

    @classmethod
    def initialize_test_engine(
        cls,
        args: FusionArguments,
        model: FusionModel,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        tb_logger: TensorboardLogger = None,
    ) -> Callable:
        return super().initialize_test_engine(
            # for detection since we use detectron we let it handle the device assignment automatically
            args,
            model,
            device,
            tb_logger,
            put_batch_to_device=False,
        )

    @classmethod
    def initialize_visualization_engine(
        cls,
        args: FusionArguments,
        model: FusionModel,
        training_engine: Engine,
        output_dir: str,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        tb_logger: TensorboardLogger = None,
    ) -> Callable:
        return super().initialize_visualization_engine(
            args,
            model,
            training_engine,
            output_dir,
            device,
            tb_logger,
            # for detection since we use detectron we let it handle the device assignment automatically
            put_batch_to_device=False,
        )

    @classmethod
    def configure_running_avg_logging(
        cls, args: FusionArguments, engine: Engine, stage: TrainingStage
    ):
        from ignite.metrics import RunningAverage

        def output_transform(x: Any, index: int, name: str) -> Any:
            import numbers

            import torch

            if isinstance(x, Mapping):
                return x[name]
            elif isinstance(x, Sequence):
                return x[index]
            elif isinstance(x, (torch.Tensor, numbers.Number)):
                return x
            else:
                raise TypeError(
                    "Unhandled type of update_function's output. "
                    f"It should either mapping or sequence, but given {type(x)}"
                )

        # detectron does not return any loss metrics during evaluation
        if stage == TrainingStage.train:
            # add loss as a running average metric
            for i, n in enumerate(args.training_args.outputs_to_metric):
                RunningAverage(
                    output_transform=partial(output_transform, index=i, name=n),
                    epoch_bound=True,
                ).attach(engine, f"{stage}/{n}")
