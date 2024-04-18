from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import ignite.distributed as idist
from torch.utils.data import Subset

from torchfusion.core.args.args import FusionArguments
from torchfusion.core.data.data_augmentations.general import DictTransform
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.utilities.logging import get_logger

if TYPE_CHECKING:
    from ignite.contrib.handlers.base_logger import BaseLogger


def is_dist_avail_and_initialized():
    import torch.distributed as dist

    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def empty_cuda_cache(_) -> None:
    import torch

    torch.cuda.empty_cache()
    import gc

    gc.collect()


def reset_random_seeds(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def concatenate_list_dict_to_dict(list_dict):
    import torch

    output = {}
    for d in list_dict:
        for k, v in d.items():
            if k not in output:
                output[k] = []
            output[k].append(v)
    output = {
        k: torch.cat(v) if len(v[0].shape) > 0 else torch.tensor(v)
        for k, v in output.items()
    }
    return output


def initialize_torch(args: FusionArguments, seed: int = 0, deterministic: bool = False):
    pass

    import os

    import ignite.distributed as idist
    import torch

    from torchfusion.utilities.logging import log_basic_info

    logger = get_logger()

    # log basic information
    log_basic_info(args)

    # initialize seed
    rank = idist.get_rank()

    seed = seed + rank
    logger.info(f"Global seed set to {seed}")
    reset_random_seeds(seed)

    # set seedon environment variable
    os.environ["DEFAULT_SEED"] = str(seed)

    # ensure that all operations are deterministic on GPU (if used) for reproducibility
    if deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def generate_output_dir(
    root_output_dir: str,
    model_task: str,
    dataset_name: str,
    model_name: str,
    experiment_name: str,
    overwrite_output_dir: bool = False,
    logging_dir_suffix: str = "",
):
    """
    Sets up the output dir for an experiment based on the arguments.
    """
    from pathlib import Path

    import ignite.distributed as idist

    logger = get_logger()

    # generate root output dir = output_dir / model_task / model_name
    output_dir = Path(root_output_dir) / model_task / dataset_name / Path(model_name)

    # create a unique directory for each experiment
    if logging_dir_suffix != "":
        experiment = f"{experiment_name}/{logging_dir_suffix}"
    else:
        experiment = f"{experiment_name}"

    # append experiment name to output dir
    output_dir = output_dir / experiment

    # overwrite the experiment if required
    if overwrite_output_dir and idist.get_rank() == 0:
        import shutil

        logger.info("Overwriting output directory.")
        shutil.rmtree(output_dir, ignore_errors=True)

    # generate directories
    if not output_dir.exists() and idist.get_rank() == 0:
        output_dir.mkdir(parents=True)

    return output_dir


def setup_logging(
    output_dir: str,
    setup_tb_logger=False,
) -> Tuple[str, BaseLogger]:
    import ignite.distributed as idist

    rank = idist.get_rank()
    logger = get_logger()

    # get the root logging directory
    logger.info(f"Setting output directory: {output_dir}")

    # Define a Tensorboard logger
    tb_logger = None
    if rank == 0 and setup_tb_logger:
        from torchfusion.core.training.utilities.tb_logger import (
            FusionTensorboardLogger,
        )

        tb_logger = FusionTensorboardLogger(log_dir=output_dir)

    return output_dir, tb_logger


def find_checkpoint_file(
    filename, checkpoint_dir: str, load_best: bool = False, resume=True, quiet=False
):
    import glob
    import os
    from pathlib import Path

    if not checkpoint_dir.exists():
        return

    logger = get_logger()
    if filename is not None:
        if Path(filename).exists():
            return Path(filename)
        elif Path(checkpoint_dir / filename).exists():
            return Path(checkpoint_dir / filename)
        else:
            logger.warning(
                f"User provided checkpoint file filename={filename} not found."
            )

    list_checkpoints = glob.glob(str(checkpoint_dir) + "/*.pt")
    if len(list_checkpoints) > 0:
        if not load_best:
            list_checkpoints = [c for c in list_checkpoints if "best" not in c]
        else:
            list_checkpoints = [c for c in list_checkpoints if "best" in c]

        if len(list_checkpoints) > 0:
            latest_checkpoint = max(list_checkpoints, key=os.path.getctime)
            if resume:
                if not quiet:
                    logger.info(
                        f"Checkpoint detected, resuming training from {latest_checkpoint}. To avoid this behavior, change "
                        "the `general_args.output_dir` or add `general_args.overwrite_output_dir` to train from scratch."
                    )
            else:
                if not quiet:
                    logger.info(
                        f"Checkpoint detected, testing model using checkpoint {latest_checkpoint}."
                    )
            return latest_checkpoint


def find_resume_checkpoint(
    resume_checkpoint_file: str, checkpoint_dir: str, load_best: bool = False
):
    return find_checkpoint_file(
        filename=resume_checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_best=load_best,
        resume=True,
    )


def find_test_checkpoint(
    test_checkpoint_file: str, checkpoint_dir: str, load_best: bool = False
):
    return find_checkpoint_file(
        filename=test_checkpoint_file,
        checkpoint_dir=checkpoint_dir,
        load_best=load_best,
        resume=False,
    )


def print_transform(tf):
    logger = get_logger()
    if idist.get_rank() == 0:
        for idx, transform in enumerate(tf.transforms):
            if isinstance(transform, DictTransform):
                logger.info(f"{idx}, {transform.key}: {transform.transform}")
            else:
                logger.info(f"{idx}, {transform}")


def print_transforms(tf, title):
    logger = get_logger()
    for split in ["train", "validation", "test"]:
        if tf[split] is None or tf[split].transforms is None:
            continue
        logger.info(f"Defining [{split}] {title}:")
        print_transform(tf[split])


def print_tf_from_loader(dataloader, stage=TrainingStage.train):
    logger = get_logger()
    tf = (
        dataloader.dataset.dataset._transforms
        if isinstance(dataloader.dataset, Subset)
        else dataloader.dataset._transforms
    )

    if tf is not None:
        logger.info("Final sanity check... Validation transforms:")
        print_transform(tf)
