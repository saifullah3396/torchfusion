from __future__ import annotations

import logging
import os
from typing import Mapping, Optional, TextIO

import coloredlogs
import ignite
import ignite.distributed as idist
import torch
from omegaconf import DictConfig

DEFAULT_LOGGER_NAME = "torchfusion"
DEFAULT_LEVEL_STYLES = {
    "critical": {"bold": True, "color": "red"},
    "debug": {"color": "green"},
    "error": {"color": "red"},
    "info": {"color": "cyan"},
    "notice": {"color": "magenta"},
    "spam": {"color": "green", "faint": True},
    "success": {"bold": True, "color": "green"},
    "verbose": {"color": "blue"},
    "warning": {"color": "yellow"},
}


def log_basic_info(args: Mapping, log_args=False, quiet=True):
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    if not quiet:
        logger.info(f"PyTorch version: {torch.__version__}{idist.get_rank()}")
        logger.info(f"Ignite version: {ignite.__version__}{idist.get_rank()}")

        if torch.cuda.is_available():
            # explicitly import cudnn as
            # torch.backends.cudnn can not be pickled with hvd spawning procs
            from torch.backends import cudnn

            logger.info(
                f"GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
            )
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDNN version: {cudnn.version()}")
        else:
            logger.info(f"Device: {idist.device()}")

        if idist.get_world_size() > 1:
            logger.info("Distributed setting:")
            logger.info(f"backend: {idist.backend()}")
            logger.info(f"world size: {idist.get_world_size()}")

        # print args
        logger.info("Initializing training script with the following arguments:")
        if idist.get_rank() == 0 and log_args:
            print(args)


class LoggingHandler:
    hydra_config: DictConfig = None

    @staticmethod
    def setup_logger(
        name: Optional[str] = "root",
        level: int = logging.INFO,
        stream: Optional[TextIO] = None,
        format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
        filepath: Optional[str] = None,
        reset: bool = False,
        hydra_config: DictConfig = None,
    ) -> logging.Logger:
        if hydra_config is not None:
            LoggingHandler.hydra_config = hydra_config

        if "LOG_LEVEL" in os.environ:
            level = getattr(logging, os.environ["LOG_LEVEL"].upper(), None)

        # check if the logger already exists
        existing = name is None or name in logging.root.manager.loggerDict

        # if existing, get the logger otherwise create a new one
        logger = logging.getLogger(name)

        distributed_rank = idist.get_rank()

        # Remove previous handlers
        if distributed_rank > 0 or reset:
            if logger.hasHandlers():
                for h in list(logger.handlers):
                    logger.removeHandler(h)

        if distributed_rank > 0:
            # Add null handler to avoid multiple parallel messages
            logger.addHandler(logging.NullHandler())

        # Keep the existing configuration if not reset
        if existing and not reset:
            return logger

        if distributed_rank == 0:
            logger.setLevel(level)

            formatter = logging.Formatter(format)

            ch = logging.StreamHandler(stream=stream)
            ch.setLevel(level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            coloredlogs.install(
                logger=logger,
                level=level,
                level_styles=DEFAULT_LEVEL_STYLES,
            )

            if filepath is not None:
                fh = logging.FileHandler(filepath)
                fh.setLevel(level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)

            try:
                # this isdefault hydra config path for logging
                filepath = f"{LoggingHandler.hydra_config.runtime.output_dir}/{LoggingHandler.hydra_config.job.name}.log"
                fh = logging.FileHandler(filepath)
                fh.setLevel(level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as exc:
                pass

        # don't propagate to ancestors
        # the problem here is to attach handlers to loggers
        # should we provide a default configuration less open ?
        if name is not None:
            logger.propagate = False

        return logger


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
    hydra_config: DictConfig = None,
    reset: bool = False,
) -> logging.Logger:
    if hydra_config is not None:
        reset = True
    return LoggingHandler.setup_logger(
        name, hydra_config=hydra_config, level=logging.INFO, reset=reset
    )
