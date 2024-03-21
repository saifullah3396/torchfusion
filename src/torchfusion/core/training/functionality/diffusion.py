from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple, Union

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch.utils.data import DataLoader

from torchfusion.core.args.args import FusionArguments
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.functionality.default import DefaultTrainingFunctionality
from torchfusion.core.training.utilities.constants import TrainingStage

if TYPE_CHECKING:
    import torch

    from torchfusion.core.models.fusion_model import FusionModel


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


class DiffusionTrainingFunctionality(DefaultTrainingFunctionality):
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
            model.prepare_model_for_run(stage=TrainingStage.visualization)

            visualization_engine.run(
                val_dataloader, max_epochs=1, epoch_length=1
            )  # only run for one batch

            # prepare model for training again
            model.prepare_model_for_run(stage=TrainingStage.train)

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
        (
            training_engine,
            validation_engine,
        ) = DefaultTrainingFunctionality.setup_training_engine(
            args=args,
            model=model,
            opt_manager=opt_manager,
            training_sch_manager=training_sch_manager,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            output_dir=output_dir,
            tb_logger=tb_logger,
            device=device,
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
