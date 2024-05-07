from __future__ import annotations

import logging
import os
from typing import Type

import ignite.distributed as idist
import torch
from omegaconf import DictConfig, OmegaConf
from torchfusion.core.analyzer.tasks.factory import AnalyzerTaskFactory
from torchfusion.core.args.args import FusionArguments
from torchfusion.core.utilities.dataclasses.dacite_wrapper import from_dict
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


class FusionAnalyzer:
    def __init__(self, args, hydra_config) -> None:
        self._args = args
        self._hydra_config = hydra_config

    def analyze(self, local_rank=0):
        """
        Initializes the training of a model given dataset, and their configurations.
        """

        if self._args.analyzer_args is None:
            raise ValueError("No analyzer arguments found in the config.")

        # setup task
        analyzer_tasks = AnalyzerTaskFactory.create(self._args, self._hydra_config)

        # run task on datamodule and models
        for task_name, task in analyzer_tasks.items():
            logger.info(f"Running task: {task_name}")

            # initialize task
            task.setup(task_name)

            # run task
            task.run()

            # task cleanup
            task.cleanup()

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
    def analyze_parallel(
        cls, local_rank: int, args: FusionArguments, hydra_config: DictConfig
    ):
        cls.run_diagnostic(local_rank, args, hydra_config)
        return cls(args, hydra_config).analyze(local_rank)

    @classmethod
    def run(
        cls,
        cfg: DictConfig,
        hydra_config: DictConfig,
        data_class: Type[FusionArguments] = FusionArguments,
    ):
        # setup logging
        logger = get_logger(hydra_config=hydra_config)
        logger.info("Starting torchfusion training script")

        # initialize general configuration for script
        cfg = OmegaConf.to_object(cfg)
        args = from_dict(data_class=data_class, data=cfg["args"])
        if args.general_args.n_devices > 1:
            try:
                import ignite.distributed as idist

                # we run the torch distributed environment with spawn if we have all the gpus on the same script
                # such as when we set --gpus-per-task=N
                if "SLURM_JOBID" in os.environ:
                    ntasks = int(os.environ["SLURM_NTASKS"])
                else:
                    ntasks = 1
                if ntasks == 1:
                    port = (int(os.environ["SLURM_JOB_ID"]) + 10007) % 16384 + 49152
                    logger.info(f"Starting distributed training on port: [{port}]")
                    with idist.Parallel(
                        backend=args.general_args.backend,
                        nproc_per_node=args.general_args.n_devices,
                        master_port=port,
                    ) as parallel:
                        return parallel.run(cls.analyze_parallel, args, hydra_config)
                elif ntasks == int(args.general_args.n_devices):
                    with idist.Parallel(backend=args.general_args.backend) as parallel:
                        return parallel.run(cls.analyze_parallel, args, hydra_config)
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
        else:
            try:
                return cls(args, hydra_config).analyze()
            except KeyboardInterrupt:
                logging.info("Received ctrl-c interrupt. Stopping training...")
            except Exception as e:
                logging.exception(e)
            finally:
                return None, None
