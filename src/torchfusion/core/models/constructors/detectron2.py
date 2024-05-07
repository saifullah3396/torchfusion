from __future__ import annotations

import copy
from dataclasses import dataclass

import timm
import torch
from detectron2.modeling import build_model
from torchfusion.core.models.constructors.base import ModelConstructor
from torchfusion.core.models.tasks import ModelTasks


def add_vit_config(cfg):
    from detectron2.config import CfgNode as CN

    _C = cfg

    _C.MODEL.VIT = CN()

    # CoaT model name.
    _C.MODEL.VIT.NAME = ""

    # Output features from CoaT backbone.
    _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]

    _C.MODEL.VIT.IMG_SIZE = [224, 224]

    _C.MODEL.VIT.POS_TYPE = "shared_rel"

    _C.MODEL.VIT.DROP_PATH = 0.0

    _C.MODEL.VIT.MODEL_KWARGS = "{}"


@dataclass
class Detectron2ModelConstructor(ModelConstructor):
    def __post_init__(self):
        assert self.model_task in [
            ModelTasks.object_detection,
        ], f"Task {self.model_task} not supported for Detectron2ModelConstructor."

    def _init_model(self, **kwargs) -> torch.Any:
        from detectron2.config import get_cfg

        assert (
            "cfg_path" in kwargs
        ), "cfg_path must be provided for detectron2 model initialization."

        # instantiate detectron2 config
        cfg = get_cfg()

        # add vit config
        if "add_vit_config" in kwargs and kwargs["add_vit_config"]:
            add_vit_config(cfg)

        cfg.merge_from_file(kwargs["cfg_path"])

        # instantiate model
        return build_model(cfg)
