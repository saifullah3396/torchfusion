from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, Union

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, EventEnum
from torch.utils.data import DataLoader
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
    def setup_training_engine(
        cls,
        args,
        model,
        opt_manager,
        training_sch_manager,
        train_dataloader,
        val_dataloader,
        output_dir,
        tb_logger,
        device,
        do_val=True,
        checkpoint_state_dict_extras={},
        data_labels: Optional[Sequence[str]] = None,
    ):
        (
            training_engine,
            validation_engine,
            visualization_engine,
        ) = super().setup_training_engine(
            args,
            model,
            opt_manager,
            training_sch_manager,
            train_dataloader,
            val_dataloader,
            output_dir,
            tb_logger,
            device,
            do_val,
            checkpoint_state_dict_extras,
            data_labels,
        )

        assert (
            validation_engine is None
        ), "Validation engine is not needed for detection as Detectron2 evaluators are used."
        cls.attach_detectron2_validator(
            args=args,
            model=model,
            training_engine=training_engine,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            output_dir=output_dir,
        )

        return training_engine, validation_engine, visualization_engine

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
        # for detection purposes we use detectron2 based evaluators which follow their own logic so validation
        # engine is not needed
        return None

    @classmethod
    def attach_detectron2_validator(
        cls,
        args: FusionArguments,
        model: FusionModel,
        training_engine: Engine,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_dir: str,
    ):
        from ignite.engine import Events

        def validate(engine):
            # prepare model for validation
            model.update_ema_for_stage(stage=TrainingStage.validation)

            epoch = training_engine.state.epoch

            import ignite.distributed as idist
            from detectron2.evaluation import (
                COCOEvaluator,
                inference_on_dataset,
                print_csv_format,
            )

            split = (
                "test" if args.data_loader_args.use_test_set_for_val else "validation"
            )
            logger.info(
                f"Running evaluation on the dataset: {args.data_args.dataset_name}, config: {args.data_args.dataset_config_name}, split: {split}"
            )
            dataset_name = f"{args.data_args.dataset_name}_{args.data_args.dataset_config_name}_{split}"
            detectron2_evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
            eval_results = inference_on_dataset(
                model.torch_model, val_dataloader, detectron2_evaluator
            )
            if idist.get_rank() == 0:
                assert isinstance(
                    eval_results, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    eval_results
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(
                        args.data_args.dataset_name
                    )
                )
                print_csv_format(eval_results)
                metrics_output = "\n".join(
                    [f"\t{k}: {v}" for k, v in eval_results.items()]
                )
                logger.info(
                    f"\nEpoch {epoch} - {TrainingStage.validation} metrics:\n {metrics_output}"
                )

            # prepare model for training again
            model.update_ema_for_stage(stage=TrainingStage.train)

        if args.training_args.eval_every_n_epochs >= 1:
            cond = Events.EPOCH_COMPLETED(every=args.training_args.eval_every_n_epochs)
            cond = cond | Events.COMPLETED
            if args.training_args.eval_on_start:
                cond = cond | Events.STARTED
            training_engine.add_event_handler(cond, validate)
        else:
            steps_per_epoch = cls._get_steps_per_epoch(args, train_dataloader)
            cond = Events.ITERATION_COMPLETED(
                every=int(args.training_args.eval_every_n_epochs * steps_per_epoch)
            )
            cond = cond | Events.COMPLETED
            if args.training_args.eval_on_start:
                cond = cond | Events.STARTED
            training_engine.add_event_handler(
                cond,
                validate,
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
