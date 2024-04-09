from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch.utils.data import DataLoader

from torchfusion.core.args.args import FusionArguments
from torchfusion.core.constants import DataKeys
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.functionality.default import (
    DefaultTrainingFunctionality,
    OptimizerEvents,
)
from torchfusion.core.training.fusion_opt_manager import FusionOptimizerManager
from torchfusion.core.training.utilities.constants import GANStage, TrainingStage
from torchfusion.utilities.logging import get_logger


class GANTrainingFunctionality(DefaultTrainingFunctionality):
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

        # make sure that optimizers are set up for generator and discriminator
        assert "gen" in opt_manager.optimizers.keys()
        assert "disc" in opt_manager.optimizers.keys()
        model_params = model.torch_model.get_param_groups()
        assert "gen" in model_params.keys()
        assert "disc" in model_params.keys()

        if args.training_args.with_amp:
            try:
                from torch.cuda.amp import autocast
            except ImportError:
                raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

            if scaler is None:
                from torch.cuda.amp.grad_scaler import GradScaler

                # scaler = dict(gen=GradScaler(enabled=True), disc=GradScaler(enabled=True))
                scaler = GradScaler(enabled=True)

        def toggle_opt(opt):
            for param in model.torch_model.parameters():
                param.requires_grad = False

            for group in opt.param_groups:
                for param in group["params"]:
                    param.requires_grad = True

        def update_step(engine, batch, key, gan_stage):
            opt = opt_manager.optimizers[key]

            # only update the parameters associated  with this optimizer
            toggle_opt(opt)

            if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
                opt.zero_grad()

            # get auto encoder output with loss
            outputs = model.torch_model.training_step(
                engine=engine, batch=batch, tb_logger=tb_logger, gan_stage=gan_stage
            )

            # make sure we get a dict from the model
            assert isinstance(outputs, dict), "Model must return an instance of dict."

            # get loss
            loss = outputs[DataKeys.LOSS]

            # accumulate loss if required
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            # perform optimizer update for correct gradient accumulation step
            if engine.state.iteration % gradient_accumulation_steps == 0:
                # perform gradient clipping if needed
                if args.training_args.enable_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.torch_model.get_param_groups()[key],
                        args.training_args.max_grad_norm,
                    )

                opt.step()

            return outputs

        def update_step_with_amp(engine, batch, key, gan_stage):
            opt = opt_manager.optimizers[key]

            # only update the parameters associated  with this optimizer
            toggle_opt(opt)

            if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
                opt.zero_grad()

            with autocast(enabled=True):
                # get auto encoder output with loss
                outputs = model.training_step(
                    engine=engine, batch=batch, tb_logger=tb_logger, gan_stage=gan_stage
                )

                # make sure we get a dict from the model
                assert isinstance(
                    outputs, dict
                ), "Model must return an instance of dict."

                # get loss
                loss = outputs[DataKeys.LOSS]

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
                        scaler.unscale_(opt)

                        torch.nn.utils.clip_grad_norm_(
                            model.torch_model.get_param_groups()[key],
                            args.training_args.max_grad_norm,
                        )

                    scaler.step(opt)
            else:
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

                    opt.step()

            return outputs

        def training_step(
            engine: Engine, batch: Sequence[torch.Tensor]
        ) -> Union[Any, Tuple[torch.Tensor]]:
            """
            Define the model training update step
            """

            from ignite.utils import convert_tensor

            # setup model for training
            model.torch_model.train()

            # put batch to device
            batch = convert_tensor(batch, device=device, non_blocking=non_blocking)

            if args.training_args.with_amp:
                # update the autoencoder stage
                gen_output = update_step_with_amp(
                    engine, batch, key="gen", gan_stage=GANStage.train_gen
                )

                # update the discriminator stage
                disc_output = update_step_with_amp(
                    engine, batch, key="disc", gan_stage=GANStage.train_disc
                )

                # perform optimizer update for correct gradient accumulation step
                if engine.state.iteration % gradient_accumulation_steps == 0:
                    if scaler:
                        # scaler update should be called only once. See https://pytorch.org/docs/stable/amp.html
                        scaler.update()
            else:
                # update the autoencoder stage
                gen_output = update_step(
                    engine, batch, key="gen", gan_stage=GANStage.train_gen
                )

                # update the discriminator stage
                disc_output = update_step(
                    engine, batch, key="disc", gan_stage=GANStage.train_disc
                )

            return {**gen_output, **disc_output}

        return Engine(training_step)

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

        logger = get_logger()

        # add loss as a running average metric
        for i, n in enumerate(args.training_args.outputs_to_metric):
            logger.info(f"Attaching metric {n} to {stage} engine.")
            RunningAverage(
                alpha=0.98,
                output_transform=partial(output_transform, index=i, name=n),
                epoch_bound=True,
            ).attach(engine, f"{stage}/{n}")

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
                        stage=TrainingStage.visualization,
                    )

        return Engine(visualization_step)

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
            data = torch.arange(
                0, args.data_args.data_loader_args.per_device_eval_batch_size
            )
            val_dataloader = DataLoader(
                data,
                batch_size=args.data_args.data_loader_args.per_device_eval_batch_size,
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
