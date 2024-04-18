import os
import sys

os.environ["DATA_ROOT_DIR"] = "/home/ataraxia/Datasets/"
os.environ["TORCH_FUSION_CACHE_DIR"] = "/home/ataraxia/torchfusion/cache/"
os.environ["TORCH_FUSION_OUTPUT_DIR"] = "/home/ataraxia/torchfusion/output/torchfusion"

PATH_TO_SOURCE_DIR = os.path.abspath("./src/")
PATH_TO_CONFIGS_DIR = os.path.abspath("./cfg_new/")
sys.path.append(PATH_TO_SOURCE_DIR)

print("Adding source path: ", PATH_TO_SOURCE_DIR)
print("Using configs path: ", PATH_TO_CONFIGS_DIR)

from hydra import compose, initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from torchfusion.core.args.args import FusionArguments
from torchfusion.core.args.args_base import ClassInitializerArgs
from torchfusion.core.training.fusion_trainer import FusionTrainer
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.utilities.dataclasses.dacite_wrapper import from_dict


def initialize_hydra(config_path: str):
    # clear previous instances of hydra
    GlobalHydra.instance().clear()

    # initialize hydra
    initialize_config_dir(
        version_base=None, config_dir=f"{PATH_TO_CONFIGS_DIR}/{config_path}"
    )

    # compose a hydra config from the path given
    overrides = [
        "args/data_args=datasets/image_classification/tobacco3482",  # prepare cifar10
        "args/train_val_sampler=random_split",
    ]
    cfg = compose(
        config_name="prepare_image_datasets.yaml",
        overrides=overrides,
        return_hydra_config=True,
    )
    hydra_config = cfg["hydra"]
    hydra_config.runtime.output_dir = "./"
    cfg = OmegaConf.to_object(cfg["args"])

    # # in case of notebooks we remove the analyzer arguments as the notebook itself runs the analysis code
    args = from_dict(data_class=FusionArguments, data=cfg)

    # set tb logger to off
    args.training_args.log_to_tb = False

    # you can update args here
    # for example set train_val_sampler to None
    # args.train_val_sampler = ClassInitializerArgs(
    #     name="RandomSplitSampler", kwargs={"random_split_ratio": 0.9}
    # )
    # args.train_val_sampler = None

    # print the default config that we received
    # print("Using the following configuration: ")
    # print(args)

    return args, hydra_config


args, hydra_config = initialize_hydra("defaults")

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.data_augmentations.general import DictTransform
from torchfusion.core.data.utilities.containers import TransformsDict
from torchfusion.core.models.utilities.data_collators import BatchToTensorDataCollator
from torchfusion.utilities.logging import get_logger


def get_trainer(pretrained_checkpoint=None, transforms=None):
    trainer = FusionTrainer(args, hydra_config=hydra_config)

    # setup training
    trainer._setup_training(setup_tb_logger=False)

    # setup datamodule
    trainer._datamodule = trainer._setup_datamodule(
        stage=None, realtime_transforms=transforms
    )

    # setup dataloaders
    trainer._args.data_loader_args.per_device_train_batch_size = 32
    trainer._args.data_loader_args.per_device_eval_batch_size = 32
    trainer._args.data_loader_args.dataloader_num_workers = 1
    trainer._test_dataloader = trainer._datamodule.test_dataloader(
        trainer._args.data_loader_args.per_device_eval_batch_size,
        dataloader_num_workers=trainer._args.data_loader_args.dataloader_num_workers,
        pin_memory=trainer._args.data_loader_args.pin_memory,
    )

    return trainer


trainer = get_trainer(transforms=None)

# setup dataloaders
trainer._datamodule._collate_fns.test = BatchToTensorDataCollator(
    type_map={"image": torch.float32, "label": torch.long}
)
with torch.no_grad():
    for batch in trainer._test_dataloader:
        image = batch["image"]
        image = image.cuda()
        trainer._datamodule.show_batch(batch)
        break
