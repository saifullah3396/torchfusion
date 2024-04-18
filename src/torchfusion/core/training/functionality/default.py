from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, EventEnum, Events
from ignite.handlers import Checkpoint
from ignite.handlers.checkpoint import BaseSaveHandler, DiskSaver
from torch.utils.data import DataLoader

from torchfusion.core.args.args import FusionArguments
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.fusion_opt_manager import FusionOptimizerManager
from torchfusion.core.training.metrics.factory import MetricsFactory
from torchfusion.core.training.sch.schedulers.warmup import (
    create_lr_scheduler_with_warmup,
)
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.general import (
    empty_cuda_cache,
    pretty_print_dict,
)
from torchfusion.core.training.utilities.progress_bar import TqdmToLogger
from torchfusion.core.utilities.logging import get_logger

if TYPE_CHECKING:
    import torch

    from torchfusion.core.models.fusion_model import FusionModel
    from torchfusion.core.training.fusion_sch_manager import FusionSchedulersManager


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


class DefaultTrainingFunctionality:
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
        if args.training_args.gradient_accumulation_steps <= 0:
            raise ValueError(
                "Gradient_accumulation_steps must be strictly positive. "
                "No gradient accumulation if the value set to one (default)."
            )

        from ignite.engine import Engine

        # get related arguments
        gradient_accumulation_steps = args.training_args.gradient_accumulation_steps
        non_blocking = args.training_args.non_blocking_tensor_conv

        if args.training_args.with_amp:
            try:
                from torch.cuda.amp import autocast
            except ImportError:
                raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

            if scaler is None:
                from torch.cuda.amp.grad_scaler import GradScaler

                scaler = GradScaler(enabled=True)

        def training_step(
            engine: Engine, batch: Sequence[torch.Tensor]
        ) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model training update step
            """

            import torch
            from ignite.utils import convert_tensor

            from torchfusion.core.constants import DataKeys

            # perform optimizers zero_grad() operation with gradient accumulation
            if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
                for opt in opt_manager.optimizers.values():
                    opt.zero_grad()

            # setup model for training
            model.torch_model.train()

            # put batch to device
            batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

            # forward pass
            model_output = model.training_step(
                engine=engine, batch=batch, tb_logger=tb_logger
            )

            # make sure we get a dict from the model
            assert isinstance(
                model_output, dict
            ), "Model must return an instance of dict."

            # get loss from the output dict
            loss = model_output[DataKeys.LOSS]

            # accumulate loss if required
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # backward pass
            loss.backward()

            # perform optimizer update for correct gradient accumulation step
            if engine.state.iteration % gradient_accumulation_steps == 0:
                # perform gradient clipping if needed
                if args.training_args.enable_grad_clipping:
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(
                        model.torch_model.parameters(), args.training_args.max_grad_norm
                    )

                for opt in opt_manager.optimizers.values():
                    opt.step()

                engine.fire_event(OptimizerEvents.OPTIMIZER_STEP_CALLED)
                engine.state.optimizer_step_called += 1

            # if on the go training evaluation is required, detach data from the graph
            if args.training_args.eval_training:
                return_dict = {}
                for key, value in model_output.items():
                    if key == DataKeys.LOSS:
                        return_dict[key] = value.item()
                    elif isinstance(value, torch.Tensor):
                        return_dict[key] = value.detach()
                return return_dict

            return {DataKeys.LOSS: model_output[DataKeys.LOSS].item()}

        def training_step_with_amp(
            engine: Engine, batch: Sequence[torch.Tensor]
        ) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model training update step
            """

            import torch
            from ignite.utils import convert_tensor

            from torchfusion.core.constants import DataKeys

            # perform optimizers zero_grad() operation with gradient accumulation
            if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
                for opt in opt_manager.optimizers.values():
                    opt.zero_grad()

            # setup model for training
            model.torch_model.train()

            # put batch to device
            batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

            with autocast(enabled=True):
                # forward pass
                model_output = model.training_step(
                    engine=engine, batch=batch, tb_logger=tb_logger
                )

                # make sure we get a dict from the model
                assert isinstance(
                    model_output, dict
                ), "Model must return an instance of dict."

                # get loss from the output dict
                loss = model_output[DataKeys.LOSS]

                # accumulate loss if required
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()

                # perform optimizer update for correct gradient accumulation step
                if engine.state.iteration % gradient_accumulation_steps == 0:
                    # perform gradient clipping if needed
                    if args.training_args.enable_grad_clipping:
                        # Unscales the gradients of optimizer's assigned params in-place
                        for opt in opt_manager.optimizers.values():
                            scaler.unscale_(opt)

                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(
                            model.torch_model.parameters(),
                            args.training_args.max_grad_norm,
                        )

                    for opt in opt_manager.optimizers.values():
                        scaler.step(opt)

                    # scaler update should be called only once. See https://pytorch.org/docs/stable/amp.html
                    scaler.update()

                    engine.fire_event(OptimizerEvents.OPTIMIZER_STEP_CALLED)
                    engine.state.optimizer_step_called += 1
            else:
                # backward pass
                loss.backward()

                # perform optimizer update for correct gradient accumulation step
                if engine.state.iteration % gradient_accumulation_steps == 0:
                    # perform gradient clipping if needed
                    if args.training_args.enable_grad_clipping:
                        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        torch.nn.utils.clip_grad_norm_(
                            model.torch_model.parameters(),
                            args.training_args.max_grad_norm,
                        )

                    for opt in opt_manager.optimizers.values():
                        opt.step()

                    engine.fire_event(OptimizerEvents.OPTIMIZER_STEP_CALLED)
                    engine.state.optimizer_step_called += 1

            # if on the go training evaluation is required, detach data from the graph
            if args.training_args.eval_training:
                return_dict = {}
                for key, value in model_output.items():
                    if key == DataKeys.LOSS:
                        return_dict[key] = value.item()
                    elif isinstance(value, torch.Tensor):
                        return_dict[key] = value.detach()
                return return_dict

            return {DataKeys.LOSS: model_output[DataKeys.LOSS].item()}

        if args.training_args.with_amp:
            return Engine(training_step_with_amp)
        else:
            return Engine(training_step)

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
        from ignite.engine import Engine

        # get related arguments
        non_blocking = args.training_args.non_blocking_tensor_conv

        def validation_step(
            engine: Engine, batch: Sequence[torch.Tensor]
        ) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model evaluation update step
            """

            import torch
            from ignite.utils import convert_tensor
            from torch.cuda.amp import autocast

            from torchfusion.core.training.utilities.constants import TrainingStage

            # ready model for evaluation
            model.torch_model.eval()

            with torch.no_grad():
                with autocast(enabled=args.training_args.with_amp_inference):
                    # put batch to device
                    batch = convert_tensor(
                        batch, device=device, non_blocking=non_blocking
                    )

                    # forward pass
                    return model.evaluation_step(
                        engine=engine,
                        training_engine=training_engine,
                        batch=batch,
                        tb_logger=tb_logger,
                        stage=TrainingStage.validation,
                    )

        return Engine(validation_step)

    @classmethod
    def initialize_prediction_engine(
        cls,
        args: FusionArguments,
        model: FusionModel,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        tb_logger: TensorboardLogger = None,
        keys_to_device: list = [],
    ) -> Callable:
        from ignite.engine import Engine

        # get related arguments
        non_blocking = args.training_args.non_blocking_tensor_conv

        def prediction_step(
            engine: Engine, batch: Sequence[torch.Tensor]
        ) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model evaluation update step
            """

            import torch
            from ignite.utils import convert_tensor

            # ready model for evaluation
            model.torch_model.eval()
            if args.training_args.with_amp_inference:
                model.torch_model.half()

            with torch.no_grad():
                if len(keys_to_device) > 0:
                    keys = list(batch.keys())
                    for k in keys:
                        if k in keys_to_device:
                            batch[k] = convert_tensor(
                                batch[k], device=device, non_blocking=non_blocking
                            )
                else:
                    # put batch to device
                    batch = convert_tensor(
                        batch, device=device, non_blocking=non_blocking
                    )

                # if fp16 is on
                if args.training_args.with_amp_inference:
                    for k, v in batch.items():
                        if not isinstance(v, torch.Tensor):
                            continue
                        batch[k] = v.half()

                return model.predict_step(
                    engine=engine, batch=batch, tb_logger=tb_logger
                )

        return Engine(prediction_step)

    @classmethod
    def initialize_test_engine(
        cls,
        args: FusionArguments,
        model: FusionModel,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        tb_logger: TensorboardLogger = None,
    ) -> Callable:
        from ignite.engine import Engine

        # get related arguments
        non_blocking = args.training_args.non_blocking_tensor_conv

        def test_step(
            engine: Engine, batch: Sequence[torch.Tensor]
        ) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model evaluation update step
            """

            import torch
            from ignite.utils import convert_tensor

            from torchfusion.core.training.utilities.constants import TrainingStage

            # ready model for evaluation
            model.torch_model.eval()
            if args.training_args.with_amp_inference:
                model.torch_model.half()
            with torch.no_grad():
                # put batch to device
                batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

                # if fp16 is on
                if args.training_args.with_amp_inference:
                    for k, v in batch.items():
                        if not isinstance(v, torch.Tensor):
                            continue
                        batch[k] = v.half()

                # forward pass
                return model.evaluation_step(
                    engine=engine,
                    training_engine=None,
                    batch=batch,
                    tb_logger=tb_logger,
                    stage=TrainingStage.test,
                )

        return Engine(test_step)

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
        from ignite.engine import Engine

        # get related arguments
        non_blocking = args.training_args.non_blocking_tensor_conv

        def visualization_step(
            engine: Engine, batch: Sequence[torch.Tensor]
        ) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model evaluation update step
            """

            import torch
            from ignite.utils import convert_tensor
            from torch.cuda.amp import autocast

            # ready model for evaluation
            model.torch_model.eval()
            if args.training_args.with_amp_inference:
                model.torch_model.half()

            with torch.no_grad():
                with autocast(enabled=args.training_args.with_amp_inference):
                    # put batch to device
                    batch = convert_tensor(
                        batch, device=device, non_blocking=non_blocking
                    )

                    # if fp16 is on
                    if args.training_args.with_amp_inference:
                        for k, v in batch.items():
                            if not isinstance(v, torch.Tensor):
                                continue
                            batch[k] = v.half()

                    # forward pass
                    return model.visualization_step(
                        engine=engine,
                        training_engine=training_engine,
                        batch=batch,
                        tb_logger=tb_logger,
                        output_dir=output_dir,
                    )

        return Engine(visualization_step)

    @classmethod
    def _get_batches_per_epoch(cls, args, train_dataloader):
        # if args.privacy_args.use_bmm:
        #     print(
        #         len(train_dataloader),
        #         args.privacy_args.max_physical_batch_size,
        #         args.data_loader_args.per_device_train_batch_size,
        #     )
        #     ratio = (
        #         args.data_loader_args.per_device_train_batch_size
        #         / args.privacy_args.max_physical_batch_size
        #     )
        #     return len(train_dataloader) * ratio
        # else:
        return len(train_dataloader)

    @classmethod
    def _get_steps_per_epoch(cls, args, train_dataloader) -> int:
        """Total training steps inferred from datamodule and devices."""

        # batches = cls._get_batches_per_epoch(train_dataloader)
        # effective_accum = args.training_args.gradient_accumulation_steps * idist.get_world_size()
        # return batches // effective_accum
        return (
            cls._get_batches_per_epoch(args, train_dataloader)
            // args.training_args.gradient_accumulation_steps
        )

    @classmethod
    def _get_total_training_steps(cls, args, train_dataloader) -> int:
        """Total number of training steps inferred from datamodule and devices."""

        return (
            cls._get_steps_per_epoch(args, train_dataloader)
            * args.training_args.max_epochs
        )

    @classmethod
    def _get_warmup_steps(cls, args, train_dataloader):
        """Total number of warmup steps to be used."""

        return (
            args.training_args.warmup_steps
            if args.training_args.warmup_steps > 0
            else math.ceil(
                cls._get_total_training_steps(args, train_dataloader)
                * args.training_args.warmup_ratio
            )
        )

    @classmethod
    def configure_nan_callback(cls, args: FusionArguments, training_engine: Engine):
        from ignite.engine import Events

        # setup nan termination callback if required
        if args.training_args.stop_on_nan:
            from ignite.handlers import TerminateOnNan

            training_engine.add_event_handler(
                Events.ITERATION_COMPLETED, TerminateOnNan()
            )

    @classmethod
    def configure_cuda_cache_callback(
        cls, args: FusionArguments, training_engine: Engine
    ):
        import torch
        from ignite.engine import Events

        # add cuda cache clear callback if required
        if torch.cuda.is_available() and args.training_args.clear_cuda_cache:
            training_engine.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    @classmethod
    def configure_gpu_stats_callback(
        cls, args: FusionArguments, training_engine: Engine
    ):
        import ignite.distributed as idist
        import torch

        # add gpu stats callback if required
        if idist.device() != torch.device("cpu") and args.training_args.log_gpu_stats:
            from ignite.contrib.metrics import GpuInfo
            from ignite.engine import Events

            GpuInfo().attach(
                training_engine,
                name="gpu",
                event_name=Events.ITERATION_COMPLETED(
                    every=args.training_args.logging_steps
                ),
            )

    @classmethod
    def configure_model_ema_callback(
        cls, args: FusionArguments, training_engine: Engine, model: FusionModel
    ):
        if model.ema_handler is not None:
            logger = get_logger()

            if isinstance(model.ema_handler, dict):
                for key, handler in model.ema_handler.items():
                    logger.info(
                        f"Attaching EMAHandler[{key}] with following configuration: {args.training_args.model_ema_args}"
                    )
                    handler.attach(
                        training_engine,
                        name=f"{key}_ema_momentum",
                        event=OptimizerEvents.OPTIMIZER_STEP_CALLED(
                            every=args.training_args.model_ema_args.update_every
                        ),
                    )

                # @training_engine.on(Events.ITERATION_COMPLETED)
                # def print_ema_momentum(engine):
                #     for key in model.ema_handler.keys():
                #         print(f"current momentum for {key}: {getattr(engine.state, f'{key}_ema_momentum')}")
            else:
                logger.info(
                    f"Attaching EMAHandler with following configuration: {args.training_args.model_ema_args}"
                )
                model.ema_handler.attach(
                    training_engine,
                    name="ema_momentum",
                    event=OptimizerEvents.OPTIMIZER_STEP_CALLED(
                        every=args.training_args.model_ema_args.update_every
                    ),
                )

    @classmethod
    def configure_early_stopping_callback(
        cls, args: FusionArguments, training_engine: Engine, validation_engine: Engine
    ):
        # add gpu stats callback if required
        if args.training_args.early_stopping_args.monitored_metric is not None:
            from ignite.engine import Events
            from ignite.handlers import Checkpoint, EarlyStopping

            cfg = args.training_args.early_stopping_args
            es_handler = EarlyStopping(
                patience=cfg.patience,
                score_function=Checkpoint.get_default_score_fn(
                    cfg.monitored_metric, -1 if cfg.mode == "min" else 1.0
                ),
                trainer=training_engine,
            )
            validation_engine.add_event_handler(Events.COMPLETED, es_handler)

    @classmethod
    def configure_train_sampler(
        cls,
        args: FusionArguments,
        training_engine: Engine,
        train_dataloader: DataLoader,
    ):
        import ignite.distributed as idist
        from torch.utils.data.distributed import DistributedSampler

        if idist.get_world_size() > 1:
            from ignite.engine import Engine, Events

            train_sampler = train_dataloader.sampler
            if not isinstance(train_sampler, DistributedSampler):
                raise TypeError(
                    "Train sampler should be torch DistributedSampler and have `set_epoch` method"
                )

            @training_engine.on(Events.EPOCH_STARTED)
            def distrib_set_epoch(engine: Engine) -> None:
                cast(DistributedSampler, train_sampler).set_epoch(
                    engine.state.epoch - 1
                )

        else:
            # check whether the correct training sample is being used
            if train_dataloader.sampler is not None and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                logger = get_logger()

                logger.warning(
                    "Argument train_sampler is a distributed sampler,"
                    " but either there is no distributed setting or world size is < 2. "
                    "Train sampler argument will be ignored",
                    UserWarning,
                )

    @classmethod
    def configure_tb_logger(
        cls,
        args: FusionArguments,
        training_engine: Engine,
        validation_engine: Engine,
        model: FusionModel,
        opt_manager: FusionOptimizerManager,
        tb_logger: TensorboardLogger,
    ):
        class_labels = None
        if hasattr(model, "labels"):
            class_labels = model.labels

        # setup tensorboard logging if required
        if args.training_args.log_to_tb is not None:
            from ignite.contrib.handlers import global_step_from_engine
            from ignite.engine import Events

            # generate output transform for metrics
            output_transform = lambda output: {
                k: v
                for k, v in output.items()
                if k in args.training_args.outputs_to_metric
            }

            # attach handler to plot trainer's loss every 'logging_steps' iterations
            tb_logger.attach_output_handler(
                training_engine,
                event_name=Events.ITERATION_COMPLETED(
                    every=args.training_args.logging_steps
                ),
                tag=f"step",
                metric_names="all",
                class_labels=class_labels,
                output_transform=output_transform,
            )

            # attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at every
            # 'logging_steps' iteration
            for param_name in ["lr", "weight_decay"]:
                for k, opt in opt_manager.optimizers.items():
                    tb_logger.attach_opt_params_handler(
                        training_engine,
                        event_name=Events.ITERATION_STARTED(
                            every=args.training_args.logging_steps
                        ),
                        optimizer=opt,
                        param_name=param_name,
                        tag=f"step/opt/{k}",
                    )

            # from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler, GradsScalarHandler
            # for handler in [GradsHistHandler(model._nn_model), GradsScalarHandler(model._nn_model, reduction=torch.norm)]:
            #     tb_logger.attach(
            #         training_engine,
            #         event_name=Events.ITERATION_COMPLETED,
            #         log_handler=handler
            #     )

            if validation_engine is not None:
                # attach tb logger to validation engine
                tb_logger.attach_output_handler(
                    validation_engine,
                    event_name=Events.EPOCH_COMPLETED,
                    metric_names="all",
                    tag="epoch",
                    global_step_transform=global_step_from_engine(training_engine),
                    class_labels=class_labels,
                    output_transform=output_transform,
                )

    @classmethod
    def configure_test_tb_logger(
        cls,
        args: FusionArguments,
        test_engine: Engine,
        model: FusionModel,
        tb_logger: TensorboardLogger,
        tag: str = "epoch",
    ):
        from ignite.engine import Events

        class_labels = None
        if hasattr(model, "labels"):
            class_labels = model.labels

        # attach tb logger to validation engine
        tb_logger.attach_output_handler(
            test_engine,
            event_name=Events.EPOCH_COMPLETED,
            metric_names="all",
            tag=tag,
            class_labels=class_labels,
        )

    @classmethod
    def configure_metrics(
        cls,
        args: FusionArguments,
        engine: Engine,
        model: FusionModel,
        stage: TrainingStage,
        prefix: str = "",
        data_labels: Optional[Sequence[str]] = None,
    ):
        logger = get_logger()
        if stage == TrainingStage.train and not args.training_args.eval_training:
            return

        if data_labels is not None:
            num_labels_in_model = (
                model.config.num_labels if hasattr(model.config, "num_labels") else None
            )
            assert num_labels_in_model == len(data_labels), (
                f"Number of labels in model ({num_labels_in_model}) "
                f"does not match number of labels in data ({len(data_labels)})"
            )

        # logger.info(
        #     f"Initializing metrics for stage={stage} with config: {pretty_print_dict(args.training_args.metric_args)}"
        # )
        metrics = MetricsFactory.initialize_stage_metrics(
            metric_args=args.training_args.metric_args,
            model_task=args.model_args.model_task,
            labels=data_labels,
        )

        if metrics[stage] is not None:
            logger = get_logger()
            for k, metric in metrics[stage].items():
                if metric is not None:
                    logger.info(f"Attaching metric {k} for stage={stage}")
                    metric().attach(
                        engine,
                        f"{stage}/{k}" if prefix == "" else f"{prefix}/{stage}/{k}",
                    )

    @classmethod
    def configure_lr_schedulers(
        cls,
        args: FusionArguments,
        training_engine: Engine,
        train_dataloader: DataLoader,
        training_sch_manager: FusionSchedulersManager,
        validation_engine: Engine = None,
    ):
        # setup learning rate schedulers as required in the arguments
        if training_sch_manager.lr_schedulers is None:
            return

        from ignite.engine import Events
        from ignite.handlers import (
            LRScheduler,
            ParamScheduler,
            ReduceLROnPlateauScheduler,
        )
        from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR

        logger = get_logger()
        for k, inner_sch in training_sch_manager.lr_schedulers.items():
            if inner_sch is None:
                continue

            warmup_duration = cls._get_warmup_steps(args, train_dataloader)
            if warmup_duration > 0:
                logger.info(
                    f"Initialized lr scheduler {inner_sch.__class__.__name__} with warmup. "
                )
                logger.info(f"Warmup ratio = {args.training_args.warmup_ratio}. ")
                logger.info(
                    f"Number of warmup steps = {warmup_duration}. This corresponds to optimizer updates, "
                    "not total batches in epoch and therefore its scaled by grad "
                    f"acummulation steps = ${args.training_args.gradient_accumulation_steps}."
                )

                if isinstance(inner_sch, (StepLR, MultiStepLR)):
                    logger.info(
                        "Warmup updates are triggered per optimizer steps whereas the scheduler updates are triggered per epoch."
                    )
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=warmup_duration,
                    )

                    # we want warmup on optimizer update steps and step_lr on epochs, so we create two events first for steps
                    # and then for epochs
                    # Trigger scheduler on iteration_started events before reaching warmup_duration
                    combined_events = OptimizerEvents.OPTIMIZER_STEP_CALLED(
                        event_filter=lambda _, __: training_engine.state.optimizer_step_called
                        <= warmup_duration
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events |= Events.EPOCH_STARTED(
                        event_filter=lambda _, __: training_engine.state.epoch
                        > 1
                        + warmup_duration
                        / cls._get_steps_per_epoch(args, train_dataloader)
                    )

                    training_engine.add_event_handler(combined_events, sch)

                    # update scheduler in dict
                    training_sch_manager.lr_schedulers[k] = sch
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    logger.info(
                        "Warmup updates are triggered per optimizer steps whereas the scheduler updates are triggered per validation step."
                    )
                    # we want warmup on steps and step_lr on epochs, so we create two events first for steps
                    # and then for epochs
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=warmup_duration,
                    )
                    training_engine.add_event_handler(
                        OptimizerEvents.OPTIMIZER_STEP_CALLED(
                            event_filter=lambda _, __: training_engine.state.optimizer_step_called
                            <= warmup_duration
                        ),
                        sch.schedulers[0],
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events = Events.COMPLETED | Events.COMPLETED(
                        event_filter=lambda _, __: training_engine.state.epoch
                        > 1
                        + warmup_duration
                        / cls._get_steps_per_epoch(args, train_dataloader)
                    )

                    validation_engine.add_event_handler(combined_events, inner_sch)

                    # update scheduler in dict
                    training_sch_manager.lr_schedulers[k] = sch
                else:
                    logger.info(
                        "Both warmup updates and the scheduler updates are triggered per optimizer step."
                    )
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=warmup_duration,
                    )
                    training_engine.add_event_handler(
                        OptimizerEvents.OPTIMIZER_STEP_CALLED, sch
                    )

                    # update scheduler in dict
                    training_sch_manager.lr_schedulers[k] = sch
            else:
                if not isinstance(inner_sch, ParamScheduler):
                    # convert scheduler to ignite scheduler
                    sch = LRScheduler(inner_sch)
                else:
                    sch = inner_sch

                # update scheduler in dict
                if isinstance(inner_sch, (StepLR, MultiStepLR, ExponentialLR)):
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__} with warmup. Scheduler updates are triggered per epoch. "
                    )
                    training_engine.add_event_handler(Events.EPOCH_STARTED, sch)
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__} with warmup. Scheduler updates are triggered per validation step. "
                    )
                    # inner_sch.trainer = training_engine
                    validation_engine.add_event_handler(Events.COMPLETED, sch)
                else:
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__} with warmup. Scheduler updates are triggered per optimizer step. "
                    )
                    training_engine.add_event_handler(
                        OptimizerEvents.OPTIMIZER_STEP_CALLED, sch
                    )
                training_sch_manager.lr_schedulers[k] = sch

    @classmethod
    def configure_wd_schedulers(
        cls,
        args: FusionArguments,
        training_engine: Engine,
        opt_manager: FusionOptimizerManager,
        training_sch_manager: FusionSchedulersManager,
        validation_engine: Engine = None,
    ):
        # setup learning rate schedulers as required in the arguments
        if training_sch_manager.wd_schedulers is None or all(
            sch is None for sch in training_sch_manager.wd_schedulers.values()
        ):
            return
        from ignite.engine import Events

        # handle weight decay
        def update_weight_decays():
            for key, opt in opt_manager.optimizers.items():
                opt_wd_schs = training_sch_manager.wd_schedulers[key]
                if opt_wd_schs.d is None:
                    continue
                for pg_idx, pg in enumerate(opt.param_groups):
                    group_name = pg["name"]
                    if group_name in opt_wd_schs.d:
                        pg["weight_decay"] = opt_wd_schs.d[group_name].step()

        training_engine.add_event_handler(
            Events.ITERATION_STARTED, update_weight_decays
        )

    @classmethod
    def configure_model_checkpoints(
        cls,
        args: FusionArguments,
        training_engine: Engine,
        model: FusionModel,
        opt_manager: FusionOptimizerManager,
        training_sch_manager: FusionSchedulersManager,
        output_dir: str,
        validation_engine: Optional[Engine] = None,
        do_val: bool = True,
        checkpoint_state_dict_extras: dict = {},
        checkpoint_class=Checkpoint,
    ):
        # setup checkpoint saving if required
        if args.training_args.enable_checkpointing:

            checkpoint_state_dict = {
                "training_engine": training_engine,
                **checkpoint_state_dict_extras,
            }
            checkpoint_state_dict = {
                **checkpoint_state_dict,
                **model.get_checkpoint_state_dict(),
            }

            # add optimizers and lr/wd scheduler states to checkpoint_state_dict
            checkpoint_state_dict = {
                **checkpoint_state_dict,
                **opt_manager.get_checkpoint_state_dict(),
                **training_sch_manager.get_checkpoint_state_dict(),
            }

            # if only to save weights, remove all other keys
            if args.training_args.model_checkpoint_config.save_weights_only:
                for k in list(checkpoint_state_dict.keys()):
                    if k not in ["training_engine", "model"]:
                        checkpoint_state_dict.pop(k)

            model_checkpoint_config = args.training_args.model_checkpoint_config
            checkpoint_dir = Path(output_dir) / model_checkpoint_config.dir
            save_handler = DiskSaver(
                checkpoint_dir,
                require_empty=False,
            )

            if model_checkpoint_config.save_per_epoch:

                checkpoint_handler = checkpoint_class(
                    checkpoint_state_dict,
                    cast(Union[Callable, BaseSaveHandler], save_handler),
                    filename_prefix=model_checkpoint_config.name_prefix,
                    global_step_transform=lambda *_: training_engine.state.epoch,
                    n_saved=model_checkpoint_config.n_saved,
                    include_self=True,
                )
                training_engine.add_event_handler(
                    Events.EPOCH_COMPLETED(
                        every=model_checkpoint_config.save_every_iters
                    ),
                    checkpoint_handler,
                )
            else:
                checkpoint_handler = checkpoint_class(
                    checkpoint_state_dict,
                    cast(Union[Callable, BaseSaveHandler], save_handler),
                    filename_prefix=model_checkpoint_config.name_prefix,
                    n_saved=model_checkpoint_config.n_saved,
                    include_self=True,
                )
                training_engine.add_event_handler(
                    Events.ITERATION_COMPLETED(
                        every=model_checkpoint_config.save_every_iters
                    ),
                    checkpoint_handler,
                )

        if args.training_args.resume_from_checkpoint:
            import torch

            from torchfusion.core.training.utilities.general import (
                find_resume_checkpoint,
            )

            logger = get_logger()

            resume_checkpoint_path = find_resume_checkpoint(
                args.training_args.resume_checkpoint_file,
                checkpoint_dir,
                args.training_args.load_best_checkpoint_resume,
            )
            if resume_checkpoint_path is not None:
                resume_checkpoint = torch.load(
                    resume_checkpoint_path, map_location="cpu"
                )
                for k in list(checkpoint_state_dict.keys()):
                    if k not in list(resume_checkpoint.keys()):
                        logger.warning(
                            f"Object {k} not found in the resume checkpoint_state_dict."
                        )
                        del checkpoint_state_dict[k]

                load_state_dict = {**checkpoint_state_dict}
                if args.training_args.model_checkpoint_config.load_weights_only:
                    for k in list(checkpoint_state_dict.keys()):
                        if k not in ["model"]:
                            load_state_dict.pop(k)

                checkpoint_class.load_objects(
                    to_load=load_state_dict,
                    checkpoint=resume_checkpoint,
                    strict=False,
                )

        if (
            validation_engine is not None
            and do_val
            and model_checkpoint_config.monitored_metric is not None
        ):
            from ignite.contrib.handlers import global_step_from_engine

            best_model_saver = checkpoint_class(
                checkpoint_state_dict,
                save_handler=DiskSaver(
                    checkpoint_dir,
                    require_empty=False,
                ),
                filename_prefix="best",
                # filename_pattern="{filename_prefix}_{name}.{ext}",
                n_saved=model_checkpoint_config.n_best_saved,
                global_step_transform=global_step_from_engine(training_engine),
                score_name=model_checkpoint_config.monitored_metric.replace("/", "-"),
                score_function=checkpoint_class.get_default_score_fn(
                    model_checkpoint_config.monitored_metric,
                    -1 if model_checkpoint_config.mode == "min" else 1.0,
                ),
                include_self=True,
            )
            validation_engine.add_event_handler(
                Events.COMPLETED,
                best_model_saver,
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

        # add loss as a running average metric
        for i, n in enumerate(args.training_args.outputs_to_metric):
            RunningAverage(
                output_transform=partial(output_transform, index=i, name=n),
                epoch_bound=True,
            ).attach(engine, f"{stage}/{n}")

    @classmethod
    def configure_progress_bars(
        cls,
        args: FusionArguments,
        model: FusionModel,
        engine: Engine,
        stage=TrainingStage.train,
        opt_manager: FusionOptimizerManager = None,
    ):
        from ignite.engine import Events

        from torchfusion.core.training.utilities.progress_bar import FusionProgressBar

        # redirect tqdm output to logger
        tqdm_to_logger = TqdmToLogger(get_logger())
        if stage == TrainingStage.train:
            if opt_manager is None:
                raise ValueError("opt_manager is required for TrainingStage=Train.")

            FusionProgressBar(
                persist=True,
                desc="Training",
                file=tqdm_to_logger,
            ).attach(
                engine,
                metric_names="all",
                optimizers=opt_manager.optimizers,
                optimizer_params=["lr"],
                event_name=Events.ITERATION_COMPLETED(
                    every=args.training_args.logging_steps,
                ),
            )

            @engine.on(Events.EPOCH_COMPLETED)
            def progress_on_epoch_completed(engine: Engine) -> None:
                import copy

                # print all metrics including lr and momentum
                metrics = copy.deepcopy(engine.state.metrics)
                # if opt_manager.optimizers is not None:
                #     for k, opt in opt_manager.optimizers.items():
                #         for param_name in ["lr"]:
                #             min_param = 10.0
                #             max_param = 0.0
                #             for pg in opt.param_groups:
                #                 min_param = min(min_param, pg[param_name])
                #                 max_param = max(max_param, pg[param_name])
                #             if (max_param - min_param) < 1e-6:
                #                 param = f"opt/{k}/{param_name}"
                #                 metrics[param] = float(max_param)
                #             else:
                #                 min_param_name = f"opt/{k}/min/{param_name}"
                #                 max_param_name = f"opt/{k}/max/{param_name}"
                #                 metrics[min_param_name] = float(min_param)
                #                 metrics[max_param_name] = float(max_param)

                if isinstance(model.ema_handler, dict):
                    for key in model.ema_handler.keys():
                        if hasattr(engine.state, f"{key}_ema_momentum"):
                            metrics[f"ema/{key}_mom"] = getattr(
                                engine.state, f"{key}_ema_momentum"
                            )
                else:
                    if hasattr(engine.state, f"ema_momentum"):
                        metrics["ema/mom"] = engine.state.ema_momentum

                log_training_metrics(
                    get_logger(),
                    engine.state.epoch,
                    engine.state.times["EPOCH_COMPLETED"],
                    TrainingStage.train,
                    metrics,
                )

        elif stage == TrainingStage.validation:
            FusionProgressBar(
                desc="Validation",
                persist=True,
                file=tqdm_to_logger,
            ).attach(
                engine,
                event_name=Events.ITERATION_COMPLETED(
                    every=args.training_args.logging_steps
                ),
            )
        elif stage == TrainingStage.test:
            FusionProgressBar(
                desc="Testing",
                persist=True,
                file=tqdm_to_logger,
            ).attach(
                engine,
                event_name=Events.ITERATION_COMPLETED(
                    every=args.training_args.logging_steps
                ),
            )

    @classmethod
    def attach_validator(
        cls,
        args: FusionArguments,
        model: FusionModel,
        training_engine: Engine,
        validation_engine: Engine,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        from ignite.engine import Events

        logger = get_logger()

        if (
            val_dataloader is None
        ):  # we need this for tasks that require validation run for generating stuff
            data = torch.arange(0, args.data_loader_args.per_device_eval_batch_size)
            val_dataloader = DataLoader(
                data,
                batch_size=args.data_loader_args.per_device_eval_batch_size,
            )

        def validate(engine):
            # prepare model for validation
            model.update_ema_for_stage(stage=TrainingStage.validation)

            epoch = training_engine.state.epoch
            state = validation_engine.run(val_dataloader, max_epochs=1)
            log_eval_metrics(
                logger,
                epoch,
                state.times["COMPLETED"],
                TrainingStage.validation,
                state.metrics,
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
    def attach_visualizer(
        cls,
        model: FusionModel,
        args: FusionArguments,
        training_engine: Engine,
        visualization_engine: Engine,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        from ignite.engine import Events

        if (
            val_dataloader is None
        ):  # we need this for tasks that require validation run for generating stuff
            data = torch.arange(0, args.data_loader_args.per_device_eval_batch_size)
            val_dataloader = DataLoader(
                data,
                batch_size=args.data_loader_args.per_device_eval_batch_size,
            )

        @torch.no_grad()
        def visualize(engine):
            # prepare model for validation
            model.update_ema_for_stage(stage=TrainingStage.visualization)

            visualization_engine.run(
                val_dataloader, max_epochs=1, epoch_length=1
            )  # only run for one batch

            # prepare model for training again
            model.update_ema_for_stage(stage=TrainingStage.train)

        if args.training_args.visualize_every_n_epochs >= 1:
            cond = Events.EPOCH_COMPLETED(
                every=args.training_args.visualize_every_n_epochs
            )
            cond = cond | Events.COMPLETED
            if args.training_args.visualize_on_start:
                cond = cond | Events.STARTED
            training_engine.add_event_handler(cond, visualize)
        else:
            steps_per_epoch = cls._get_steps_per_epoch(args, train_dataloader)
            cond = Events.ITERATION_COMPLETED(
                every=int(args.training_args.visualize_every_n_epochs * steps_per_epoch)
            )
            cond = cond | Events.COMPLETED
            if args.training_args.visualize_on_start:
                cond = cond | Events.STARTED
            training_engine.add_event_handler(
                cond,
                visualize,
            )

    @classmethod
    def configure_training_engine(
        cls,
        args: FusionArguments,
        training_engine: Engine,
        model: FusionModel,
        opt_manager: FusionOptimizerManager,
        training_sch_manager: FusionSchedulersManager,
        output_dir: str,
        validation_engine: Optional[Engine] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        do_val: bool = True,
        checkpoint_state_dict_extras: dict = {},
        checkpoint_class=Checkpoint,
        data_labels: Optional[Sequence[str]] = None,
    ) -> None:
        import ignite.distributed as idist

        # configure training engine
        cls.configure_train_sampler(
            args=args,
            training_engine=training_engine,
            train_dataloader=train_dataloader,
        )
        cls.configure_nan_callback(args=args, training_engine=training_engine)
        cls.configure_cuda_cache_callback(args=args, training_engine=training_engine)
        cls.configure_gpu_stats_callback(args=args, training_engine=training_engine)
        cls.configure_model_ema_callback(
            args=args, training_engine=training_engine, model=model
        )
        cls.configure_metrics(
            args=args,
            engine=training_engine,
            model=model,
            stage=TrainingStage.train,
            data_labels=data_labels,
        )
        cls.configure_wd_schedulers(
            args=args,
            training_engine=training_engine,
            opt_manager=opt_manager,
            training_sch_manager=training_sch_manager,
        )
        if validation_engine is None:
            cls.configure_lr_schedulers(
                args=args,
                training_engine=training_engine,
                train_dataloader=train_dataloader,
                training_sch_manager=training_sch_manager,
            )
        cls.configure_running_avg_logging(
            args=args, engine=training_engine, stage=TrainingStage.train
        )
        if idist.get_rank() == 0:
            cls.configure_progress_bars(
                args=args,
                model=model,
                engine=training_engine,
                opt_manager=opt_manager,
                stage=TrainingStage.train,
            )

        # configure validation engine
        if validation_engine is not None:
            cls.configure_metrics(
                args=args,
                engine=validation_engine,
                model=model,
                stage=TrainingStage.validation,
                data_labels=data_labels,
            )
            cls.configure_lr_schedulers(
                args=args,
                training_engine=training_engine,
                train_dataloader=train_dataloader,
                training_sch_manager=training_sch_manager,
                validation_engine=validation_engine,
            )
            cls.configure_progress_bars(
                args=args,
                model=model,
                engine=validation_engine,
                opt_manager=opt_manager,
                stage=TrainingStage.validation,
            )
            cls.configure_running_avg_logging(
                args=args, engine=validation_engine, stage=TrainingStage.validation
            )
            cls.configure_early_stopping_callback(
                args=args,
                training_engine=training_engine,
                validation_engine=validation_engine,
            )
            cls.attach_validator(
                args=args,
                model=model,
                training_engine=training_engine,
                validation_engine=validation_engine,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
            )

        cls.configure_model_checkpoints(
            args=args,
            training_engine=training_engine,
            model=model,
            opt_manager=opt_manager,
            training_sch_manager=training_sch_manager,
            output_dir=output_dir,
            validation_engine=validation_engine,
            do_val=do_val,
            checkpoint_state_dict_extras=checkpoint_state_dict_extras,
            checkpoint_class=checkpoint_class,
        )

        if idist.get_rank() == 0 and tb_logger is not None:
            # configure tensorboard
            cls.configure_tb_logger(
                args=args,
                training_engine=training_engine,
                validation_engine=validation_engine,
                model=model,
                opt_manager=opt_manager,
                tb_logger=tb_logger,
            )

    @classmethod
    def configure_test_engine(
        cls,
        args: FusionArguments,
        test_engine: Engine,
        model: FusionModel,
        output_dir: str,
        tb_logger: Optional[TensorboardLogger] = None,
        checkpoint_type: str = "last",
        load_checkpoint: bool = True,
        data_labels: Optional[Sequence[str]] = None,
    ) -> None:
        from pathlib import Path

        import ignite.distributed as idist
        import torch
        from ignite.engine import Events
        from ignite.handlers import Checkpoint

        from torchfusion.core.training.utilities.general import find_test_checkpoint

        # configure model checkpoint_state_dict
        model_checkpoint_config = args.training_args.model_checkpoint_config
        checkpoint_dir = Path(output_dir) / model_checkpoint_config.dir
        if load_checkpoint:
            if checkpoint_type in ["last", "ema"]:
                checkpoint = find_test_checkpoint(
                    args.training_args.test_checkpoint_file,
                    checkpoint_dir,
                    load_best=False,
                )
            if checkpoint_type == "best":
                checkpoint = find_test_checkpoint(
                    args.training_args.test_checkpoint_file,
                    checkpoint_dir,
                    load_best=True,
                )
            if checkpoint is not None:
                checkpoint_state_dict = model.get_checkpoint_state_dict()
                test_checkpoint = torch.load(checkpoint, map_location="cpu")
                Checkpoint.load_objects(
                    to_load=checkpoint_state_dict, checkpoint=test_checkpoint
                )

        if checkpoint_type == "ema":
            model.activate_ema = True
        else:
            model.activate_ema = False

        # configure test engine
        cls.configure_metrics(
            args=args,
            engine=test_engine,
            model=model,
            stage=TrainingStage.test,
            prefix=checkpoint_type,
            data_labels=data_labels,
        )
        if idist.get_rank() == 0:
            cls.configure_progress_bars(
                args=args, model=model, engine=test_engine, stage=TrainingStage.test
            )
        cls.configure_running_avg_logging(
            args=args, engine=test_engine, stage=TrainingStage.test
        )

        if idist.get_rank() == 0:
            logger = get_logger()

            def log_test_metrics(engine):
                state = engine.state
                log_eval_metrics(
                    logger,
                    state.epoch,
                    state.times["COMPLETED"],
                    TrainingStage.test,
                    state.metrics,
                )

            test_engine.add_event_handler(
                Events.EPOCH_COMPLETED,
                log_test_metrics,
            )

        if idist.get_rank() == 0 and tb_logger is not None:
            # configure tensorboard
            cls.configure_test_tb_logger(
                args=args, test_engine=test_engine, model=model, tb_logger=tb_logger
            )

    @classmethod
    def configure_prediction_engine(
        cls,
        args: FusionArguments,
        prediction_engine: Engine,
        model: FusionModel,
        tb_logger: Optional[TensorboardLogger] = None,
        data_labels: Optional[Sequence[str]] = None,
    ) -> None:
        pass

        import ignite.distributed as idist
        from ignite.engine import Events

        # configure test engine
        cls.configure_metrics(
            args=args,
            engine=prediction_engine,
            model=model,
            stage=TrainingStage.predict,
            data_labels=data_labels,
        )
        if idist.get_rank() == 0:
            cls.configure_progress_bars(
                args=args,
                model=model,
                engine=prediction_engine,
                stage=TrainingStage.test,
            )

        if idist.get_rank() == 0:
            logger = get_logger()

            def log_test_metrics(engine):
                state = engine.state
                log_eval_metrics(
                    logger,
                    state.epoch,
                    state.times["COMPLETED"],
                    TrainingStage.test,
                    state.metrics,
                )

            prediction_engine.add_event_handler(
                Events.EPOCH_COMPLETED,
                log_test_metrics,
            )

        if idist.get_rank() == 0 and tb_logger is not None:
            # configure tensorboard
            cls.configure_test_tb_logger(
                args=args,
                test_engine=prediction_engine,
                model=model,
                tb_logger=tb_logger,
            )

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
        # setup training engine
        training_engine = cls.initialize_training_engine(
            args=args,
            model=model,
            opt_manager=opt_manager,
            device=device,
            tb_logger=tb_logger,
        )

        # register events
        training_engine.register_events(
            *OptimizerEvents,
            event_to_attr={
                OptimizerEvents.OPTIMIZER_STEP_CALLED: "optimizer_step_called"
            },
        )

        validation_engine = None
        if do_val:
            # setup validation engine
            validation_engine = cls.initialize_validation_engine(
                args=args,
                model=model,
                training_engine=training_engine,
                output_dir=output_dir,
                device=device,
                tb_logger=tb_logger,
            )

        # configure training and validation engines
        cls.configure_training_engine(
            args=args,
            training_engine=training_engine,
            model=model,
            opt_manager=opt_manager,
            training_sch_manager=training_sch_manager,
            output_dir=output_dir,
            tb_logger=tb_logger,
            train_dataloader=train_dataloader,
            validation_engine=validation_engine,
            val_dataloader=val_dataloader,
            do_val=do_val,
            checkpoint_state_dict_extras=checkpoint_state_dict_extras,
            data_labels=data_labels,
        )

        # visualization_engine is just another evaluation engine
        visualization_engine = cls.initialize_visualization_engine(
            args=args,
            model=model,
            training_engine=training_engine,
            output_dir=output_dir,
            device=device,
            tb_logger=tb_logger,
        )

        cls.attach_visualizer(
            args=args,
            model=model,
            training_engine=training_engine,
            visualization_engine=visualization_engine,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

        return training_engine, validation_engine

    @classmethod
    def setup_test_engine(
        cls,
        args,
        model,
        test_dataloader,
        output_dir,
        tb_logger,
        device,
        checkpoint_type: str = "last",
        data_labels: Optional[Sequence[str]] = None,
    ):
        # setup training engine
        test_engine = cls.initialize_test_engine(
            args=args, model=model, device=device, tb_logger=tb_logger
        )

        # configure training and validation engines
        cls.configure_test_engine(
            args=args,
            test_engine=test_engine,
            model=model,
            output_dir=output_dir,
            tb_logger=tb_logger,
            checkpoint_type=checkpoint_type,
            data_labels=data_labels,
        )

        return test_engine

    @classmethod
    def setup_prediction_engine(
        cls,
        args,
        model,
        test_dataloader,
        output_dir,
        tb_logger,
        device,
        checkpoint_type: str = "last",
        keys_to_device: list = [],
        data_labels: Optional[Sequence[str]] = None,
    ):
        # setup training engine
        prediction_engine = cls.initialize_prediction_engine(
            args=args,
            model=model,
            device=device,
            tb_logger=tb_logger,
            keys_to_device=keys_to_device,
        )

        # configure training and validation engines
        cls.configure_prediction_engine(
            args=args,
            prediction_engine=prediction_engine,
            model=model,
            tb_logger=tb_logger,
            data_labels=data_labels,
        )

        return prediction_engine
