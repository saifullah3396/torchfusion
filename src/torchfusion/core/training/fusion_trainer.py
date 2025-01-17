from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING, Optional, Type

import ignite.distributed as idist
import torch
from omegaconf import DictConfig, OmegaConf
from torchfusion.core.args.args import FusionArguments
from torchfusion.core.data.datasets.dataset_metadata import FusionDatasetMetaData
from torchfusion.core.data.factory.batch_sampler import BatchSamplerFactory
from torchfusion.core.data.utilities.loaders import load_datamodule_from_args
from torchfusion.core.data.utilities.transforms import load_transforms_from_config
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.tasks import ModelTasks
from torchfusion.core.training.functionality.default import DefaultTrainingFunctionality
from torchfusion.core.training.functionality.detection import (
    ObjectDetectionTrainingFunctionality,
)
from torchfusion.core.training.functionality.diffusion import (
    DiffusionTrainingFunctionality,
)
from torchfusion.core.training.functionality.gan import GANTrainingFunctionality
from torchfusion.core.training.fusion_opt_manager import FusionOptimizerManager
from torchfusion.core.training.fusion_sch_manager import FusionSchedulersManager
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.general import (
    initialize_torch,
    print_tf_from_loader,
    setup_logging,
)
from torchfusion.core.utilities.dataclasses.dacite_wrapper import from_dict
from torchfusion.core.utilities.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torchfusion.core.models.fusion_model import FusionModel

logger = get_logger(__name__)


class FusionTrainer:
    def __init__(self, args: FusionArguments, hydra_config: DictConfig) -> None:
        self._args = args
        self._hydra_config = hydra_config
        self._output_dir = None
        self._tb_logger = None
        self._trainer_functionality = None
        self._model = None
        self._opt_manager = None
        self._training_sch_manager = None
        self._datamodule = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    @property
    def optimizers(self):
        return self._opt_manager.optimizers

    @property
    def lr_schedulers(self):
        return self._training_sch_manager.lr_schedulers

    @property
    def wd_schedulers(self):
        return self._training_sch_manager.wd_schedulers

    @property
    def batches_per_epch(self):
        return self._trainer_functionality._get_batches_per_epoch(
            self._args, self._train_dataloader
        )

    @property
    def steps_per_epoch(self):
        return self._trainer_functionality._get_steps_per_epoch(
            self._args, self._train_dataloader
        )

    @property
    def total_training_steps(self):
        return self._trainer_functionality._get_total_training_steps(
            self._args, self._train_dataloader
        )

    @property
    def warmup_steps(self):
        return self._trainer_functionality._get_warmup_steps(
            self._args, self._train_dataloader
        )

    def _setup_transforms(self):
        preprocess_transforms = load_transforms_from_config(
            self._args.train_preprocess_augs,
            self._args.eval_preprocess_augs,
        )
        realtime_transforms = load_transforms_from_config(
            self._args.train_realtime_augs,
            self._args.eval_realtime_augs,
        )

        return preprocess_transforms, realtime_transforms

    def _setup_model(
        self,
        summarize: bool = False,
        setup_for_train: bool = True,
        checkpoint: Optional[str] = None,
        strict: Optional[bool] = None,
        dataset_metadata: Optional[FusionDatasetMetaData] = None,
    ) -> FusionModel:
        """
        Initializes the model for training.
        """
        from torchfusion.core.models.factory import ModelFactory

        logger.info("Setting up model...")

        # setup model
        model = ModelFactory.create_fusion_model(
            self._args,
            checkpoint=checkpoint,
            strict=strict,
            dataset_metadata=dataset_metadata,
            tb_logger=self._tb_logger,
        )

        model.setup_model(setup_for_train=setup_for_train)

        # generate model summary
        if summarize:
            model.summarize_model()

        return model

    def _setup_training(self, setup_tb_logger=True):
        # initialize training
        initialize_torch(
            self._args,
            seed=self._args.general_args.seed,
            deterministic=self._args.general_args.deterministic,
        )

        # initialize torch device (cpu or gpu)
        self._device = idist.device()

        # get device rank
        self._rank = idist.get_rank()

        # initialize logging directory and tensorboard logger
        self._output_dir, self._tb_logger = setup_logging(
            output_dir=self._hydra_config.runtime.output_dir,
            setup_tb_logger=setup_tb_logger,
        )

    def _setup_trainer_functionality(self):
        if self._args.model_args is None:
            return None

        if self._args.model_args.model_task == ModelTasks.gan:
            return GANTrainingFunctionality
        elif self._args.model_args.model_task == ModelTasks.diffusion:
            return DiffusionTrainingFunctionality
        elif self._args.model_args.model_task == ModelTasks.object_detection:
            return ObjectDetectionTrainingFunctionality
        else:
            return DefaultTrainingFunctionality

    def _setup_training_engine(self):
        # setup training engine
        (
            training_engine,
            validation_engine,
            visualization_engine,
        ) = self._trainer_functionality.setup_training_engine(
            args=self._args,
            model=self._model,
            opt_manager=self._opt_manager,
            training_sch_manager=self._training_sch_manager,
            train_dataloader=self._train_dataloader,
            val_dataloader=self._val_dataloader,
            output_dir=self._output_dir,
            tb_logger=self._tb_logger,
            device=self._device,
            do_val=self._args.general_args.do_val,
            data_labels=self._datamodule.get_dataset_metadata().get_labels(),
        )
        training_engine.logger = logger
        # training_engine.logger.propagate = False

        if validation_engine is not None:
            validation_engine.logger = logger
            # validation_engine.logger.propagate = False

        if visualization_engine is not None:
            visualization_engine.logger = logger
            # visualization_engine.logger.propagate = False

        logger.info(f"Configured Training Engine:")
        logger.info(f"\tTotal steps per epoch = {self.batches_per_epch}")
        logger.info(
            f"\tGradient accumulation per device = {self._args.training_args.gradient_accumulation_steps}"
        )
        logger.info(
            f"\tTotal optimizer update steps over (scaled by grad accumulation steps) = {self.steps_per_epoch}"
        )
        logger.info(
            f"\tTotal optimizer update over complete training cycle (scaled by grad accumulation steps) = {self.total_training_steps}"
        )
        logger.info(f"\tTotal warmup steps = {self.warmup_steps}")
        logger.info(f"\tMax epochs = {self._args.training_args.max_epochs}")
        return training_engine, validation_engine

    def _setup_test_engine(self, checkpoint_type: str = "last"):
        # setup training engine
        test_engine, checkpoint_found = self._trainer_functionality.setup_test_engine(
            args=self._args,
            model=self._model,
            test_dataloader=self._test_dataloader,
            output_dir=self._output_dir,
            tb_logger=self._tb_logger,
            device=self._device,
            checkpoint_type=checkpoint_type,
            data_labels=self._datamodule.get_dataset_metadata().get_labels(),
        )
        test_engine.logger = logger

        return test_engine, checkpoint_found

    def _setup_opt_manager(self):
        opt_manager = FusionOptimizerManager(self._args, self._model, self)
        opt_manager.setup()
        return opt_manager

    def _setup_training_sch_manager(self):
        training_sch_manager = FusionSchedulersManager(
            self._args, self._opt_manager, self
        )
        training_sch_manager.setup()
        return training_sch_manager

    def _setup_collate_fns(self):
        # now assign collate fns
        # see if the collator function requires a tokenizer or not
        collator_kwargs = {}
        if (
            "tokenizer"
            in list(  # get all key word arguments and see if tokenizer is required
                inspect.signature(self._model.get_data_collators).parameters.keys()
            )
        ):
            tokenizer = self._datamodule.get_tokenizer_if_available()
            collator_kwargs = {"tokenizer": tokenizer}

        return self._model.get_data_collators(**collator_kwargs)

    def train(self, local_rank=0):
        """
        Initializes the training of a model given dataset, and their configurations.
        """

        assert (
            self._args.model_args is not None
        ), "Model args must be provided for a training run."

        if self._args.training_args.test_run:
            logger.warning(
                "This is a test run. This will run one training and evaluation batch and terminate."
            )

        # setup training
        self._setup_training()

        # setup base training functionality
        self._trainer_functionality = self._setup_trainer_functionality()

        # setup datamodule
        self._datamodule = load_datamodule_from_args(
            args=self._args, stage=TrainingStage.train, rank=self._rank
        )

        # setup batch sampler if needed
        batch_sampler_wrapper = BatchSamplerFactory.create(
            self._args.data_loader_args.train_batch_sampler.name,
            **self._args.data_loader_args.train_batch_sampler.kwargs,
        )

        # setup model
        self._model = self._setup_model(
            summarize=True,
            setup_for_train=True,
            dataset_metadata=self._datamodule.get_dataset_metadata(),
        )

        # set collate fns
        self._datamodule._collate_fns = self._setup_collate_fns()

        # setup dataloaders
        self._train_dataloader = self._datamodule.train_dataloader(
            self._args.data_loader_args.per_device_train_batch_size,
            dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_loader_args.pin_memory,
            shuffle_data=self._args.data_loader_args.shuffle_data,
            dataloader_drop_last=self._args.data_loader_args.dataloader_drop_last,
            batch_sampler_wrapper=batch_sampler_wrapper,
        )
        self._val_dataloader = self._datamodule.val_dataloader(
            self._args.data_loader_args.per_device_eval_batch_size,
            dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_loader_args.pin_memory,
        )

        # initialize optimizer manager
        self._opt_manager = self._setup_opt_manager()

        # initialize training schedulers manager
        self._training_sch_manager = self._setup_training_sch_manager()

        # setup training engine
        self._training_engine, self._validation_engine = self._setup_training_engine()

        resume_epoch = self._training_engine.state.epoch
        if (
            self._training_engine._is_done(self._training_engine.state)
            and resume_epoch >= self._args.training_args.max_epochs
        ):  # if we are resuming from last checkpoint and training is already finished
            logger.info(
                "Training has already been finished! Either increase the number of "
                f"epochs (current={self._args.training_args.max_epochs}) >= {resume_epoch} "
                "OR reset the training from start."
            )
            return

        if self._args.training_args.test_run:
            from ignite.engine import Events

            def terminate_on_iteration_complete(
                engine,
            ):  # this is necessary for fldp to work with correct privacy accounting
                logger.info("Terminating training engine as test_run=True")
                engine.terminate()

            self._training_engine.add_event_handler(
                Events.ITERATION_COMPLETED, terminate_on_iteration_complete
            )

        logger.debug("Final sanity check... Training transforms:")
        print_tf_from_loader(self._train_dataloader, stage=TrainingStage.train)

        logger.debug("Final sanity check... Validation transforms:")
        print_tf_from_loader(self._val_dataloader, stage=TrainingStage.validation)

        # run training
        if self._args.model_args.model_task == ModelTasks.object_detection:
            from detectron2.utils.events import EventStorage, TensorboardXWriter
            from ignite.engine import Events

            writers = [
                # It may not always print what you want to see, since it prints "common" metrics only.
                TensorboardXWriter(self._hydra_config.runtime.output_dir),
            ]

            def event_writers_update():
                for writer in writers:
                    writer.write()

            def event_writers_close(self):
                for writer in writers:
                    writer.write()
                    writer.close()

            with EventStorage(start_iter=0) as self.storage:

                self._training_engine.add_event_handler(
                    Events.ITERATION_COMPLETED(
                        every=self._args.training_args.logging_steps
                    ),
                    event_writers_update,
                )
                self._training_engine.add_event_handler(
                    Events.COMPLETED, event_writers_close
                )

                def set_storage_iteration(
                    engine,
                ):
                    self.storage.iter = engine.state.iteration

                self._training_engine.add_event_handler(
                    Events.ITERATION_STARTED, set_storage_iteration
                )

                self._training_engine.run(
                    self._train_dataloader,
                    max_epochs=self._args.training_args.max_epochs,
                )
        else:
            self._training_engine.run(
                self._train_dataloader, max_epochs=self._args.training_args.max_epochs
            )

        if self._rank == 0:
            # close tb logger
            self._tb_logger.close()

        return (
            self._training_engine.state,
            self._validation_engine.state if self._validation_engine else None,
        )

    def test(self, local_rank=0):
        """
        Initializes the training of a model given dataset, and their configurations.
        """

        # setup training
        self._setup_training()

        # setup base training functionality
        self._trainer_functionality = self._setup_trainer_functionality()

        # setup dataloaders
        if self._args.data_loader_args.use_val_set_for_test:
            # setup datamodule (since we need validation dataset, we load the complete datamodule here)
            # setting training stage since validation set is to be used which is loaded in the train stage
            self._datamodule = load_datamodule_from_args(
                args=self._args, stage=TrainingStage.train, rank=self._rank
            )

            # setup model
            self._model = self._setup_model(
                summarize=True,
                setup_for_train=False,
                dataset_metadata=self._datamodule.get_dataset_metadata(),
            )

            # set collate fns
            self._datamodule._collate_fns = self._setup_collate_fns()

            # create dataloader
            self._test_dataloader = self._datamodule.val_dataloader(
                self._args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_loader_args.pin_memory,
            )
        else:
            # setup datamodule (we only load the test dataset here)
            self._datamodule = load_datamodule_from_args(
                args=self._args, stage=TrainingStage.test, rank=self._rank
            )

            # setup model
            self._model = self._setup_model(
                summarize=True,
                setup_for_train=False,
                dataset_metadata=self._datamodule.get_dataset_metadata(),
            )

            # set collate fns
            self._datamodule._collate_fns = self._setup_collate_fns()

            # setup dataloaders
            self._test_dataloader = self._datamodule.test_dataloader(
                self._args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_loader_args.pin_memory,
            )

        # print transforms before training run just for sanity check
        print_tf_from_loader(self._test_dataloader, stage=TrainingStage.test)

        # setup test engines for different types of model checkpoints
        output_states = {}
        for checkpoint_type in ["last", "best", "ema"]:
            if (
                checkpoint_type == "ema"
                and not self._args.training_args.model_ema_args.enabled
            ):
                continue

            # setup training engine
            self._test_engine, checkpoint_found = self._setup_test_engine(
                checkpoint_type
            )

            if checkpoint_type == "best" and not checkpoint_found:
                continue

            # run tests
            self._test_engine.run(self._test_dataloader)

            # save test engine state
            output_states[checkpoint_type] = self._test_engine.state

        # close tb logger
        self._tb_logger.close()

        # return state of the engine
        return output_states

    @classmethod
    def train_parallel(
        cls, local_rank: int, args: FusionArguments, hydra_config: DictConfig
    ):
        # initialize logger
        get_logger(hydra_config=hydra_config)

        cls.run_diagnostic(local_rank, args, hydra_config)
        return cls(args, hydra_config).train(local_rank)

    @classmethod
    def run_diagnostic(
        cls, local_rank: int, args: FusionArguments, hydra_config: DictConfig
    ):
        prefix = f"{local_rank}) "
        print(f"{prefix}Rank={idist.get_rank()}")
        print(f"{prefix}torch version: {torch.version.__version__}")
        print(f"{prefix}torch git version: {torch.version.git_version}")

        if torch.cuda.is_available():
            print(f"{prefix}torch version cuda: {torch.version.cuda}")
            print(f"{prefix}number of cuda devices: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"{prefix}\t- device {i}: {torch.cuda.get_device_properties(i)}")
        else:
            print("{prefix}no cuda available")

        if "SLURM_JOBID" in os.environ:
            for k in [
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_NTASKS",
                "SLURM_JOB_NODELIST",
                "MASTER_ADDR",
                "MASTER_PORT",
            ]:
                print(f"{k}: {os.environ[k]}")

    @classmethod
    def run_train(
        cls,
        cfg: DictConfig,
        hydra_config: DictConfig,
        data_class: Type[FusionArguments] = FusionArguments,
    ):
        # initialize general configuration for script
        cfg = OmegaConf.to_object(cfg)
        args = from_dict(data_class=data_class, data=cfg["args"])
        if args.general_args.n_devices > 1:
            if args.general_args.do_train:
                # setup logging
                logger = get_logger(hydra_config=hydra_config)
                logger.info("Starting torchfusion training script with arguments:")
                logger.info(args)

                try:
                    import ignite.distributed as idist

                    # we run the torch distributed environment with spawn if we have all the gpus on the same script
                    # such as when we set --gpus-per-task=N
                    if "SLURM_JOBID" in os.environ:
                        ntasks = int(os.environ["SLURM_NTASKS"])
                    else:
                        ntasks = 1
                    if ntasks == 1:
                        if "SLURM_JOB_ID" in os.environ:
                            port = (
                                int(os.environ["SLURM_JOB_ID"]) + 10007
                            ) % 16384 + 49152
                        else:
                            port = 27015
                        logger.info(f"Starting distributed training on port: [{port}]")
                        with idist.Parallel(
                            backend=args.general_args.backend,
                            nproc_per_node=args.general_args.n_devices,
                            master_port=port,
                        ) as parallel:
                            return parallel.run(cls.train_parallel, args, hydra_config)
                    elif ntasks == int(args.general_args.n_devices):
                        with idist.Parallel(
                            backend=args.general_args.backend
                        ) as parallel:
                            return parallel.run(cls.train_parallel, args, hydra_config)
                    else:
                        raise ValueError(
                            f"Your slurm tasks do not match the number of required devices [{ntasks}!={args.general_args.n_devices}]."
                        )
                except KeyboardInterrupt:
                    logging.info("Received ctrl-c interrupt. Stopping training...")
                except Exception as e:
                    logging.exception(e)
                finally:
                    return None, None
            return None, None
        else:
            if args.general_args.do_train:
                # setup logging
                logger = get_logger(hydra_config=hydra_config)
                logger.info("Starting torchfusion training script with arguments:")
                logger.info(args)

                try:
                    return cls(args, hydra_config).train()
                except KeyboardInterrupt:
                    logging.info("Received ctrl-c interrupt. Stopping training...")
                except Exception as e:
                    logging.exception(e)
                finally:
                    return None, None
            return None, None

    @classmethod
    def run_test(
        cls,
        cfg: DictConfig,
        hydra_config: DictConfig,
        data_class: Type[FusionArguments] = FusionArguments,
    ):
        # initialize general configuration for script
        cfg = OmegaConf.to_object(cfg)
        args = from_dict(data_class=data_class, data=cfg["args"])
        if args.general_args.do_test:
            # setup logging
            logger = get_logger("init")
            logger.info("Starting torchfusion testing script with arguments:")
            logger.info(args)

            try:
                return cls(args, hydra_config).test()
            except Exception as e:
                logging.exception(e)
            finally:
                return None
