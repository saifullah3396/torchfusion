""" PyTorch module that defines the base model for training/testing etc. """

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import ignite.distributed as idist
import torch
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from torchfusion.core.args.args import FusionArguments
from torchfusion.core.constants import DataKeys
from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.utilities.ddp_model_proxy import ModuleProxyWrapper
from torchfusion.core.models.utilities.general import batch_norm_to_group_norm
from torchfusion.core.training.args.ema import FusionEMAHandler
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.utilities.logging import get_logger


class FusionModel:
    @dataclass
    class Config(FusionModelConfig):
        pass

    def __init__(
        self,
        args: FusionArguments,
        tb_logger: Optional[TensorboardLogger] = None,
    ):
        super().__init__()

        # initialize arguments
        self._args = args
        self._tb_logger = tb_logger

        # ema parameters
        self._ema_handler = None
        self._ema_activated = False
        self._stored_parameters = None

        # initialize logger
        self._logger = get_logger()

    @property
    def ema_handler(self):
        return self._ema_handler

    @property
    def model_name(self):
        return self.model_args.model_directory_name

    @property
    def torch_model(self):
        return self._torch_model

    @property
    def model_args(self):
        return self._args.model_args

    @property
    def config(self) -> FusionModelConfig:
        return self.model_args.model_config

    @property
    def data_args(self):
        return self._args.data_args

    @property
    def training_args(self):
        return self._args.training_args

    @abstractmethod
    def _build_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
    ):
        pass

    @abstractmethod
    def _training_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _evaluation_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _predict_step(self, *args, **kwargs):
        pass

    def _visualization_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _model_forward(self):
        pass

    @abstractmethod
    def get_data_collators(self):
        pass

    def _init_weights(self):
        pass

    def get_param_groups(self):
        return {
            "default": list(self.torch_model.parameters()),
        }

    def build_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
    ):
        from torchfusion.core.models.factory import ModelFactory

        # models sometimes download pretrained checkpoints when initializing. Only download it on rank 0
        if idist.get_rank() > 0:  # stop all ranks > 0
            idist.barrier()

        # build the underlying nn model
        self._torch_model = self._build_model(checkpoint=checkpoint, strict=strict)
        assert self._torch_model is not None and isinstance(
            self._torch_model, nn.Module
        ), "Child class must return a torch nn.Module on self._build_model()"

        # wait for rank 0 to download checkpoints
        if idist.get_rank() == 0:
            idist.barrier()

    def training_step(self, *args, **kwargs) -> None:
        if self._args.training_args.test_run:
            self._logger.info("Following input is provided to the model for training:")
            for key, value in kwargs["batch"].items():
                if value.dtype == torch.float:
                    self._logger.info(
                        f"[{key}] shape={value.shape}, min={value.min()}, max={value.max()}, mean={value.mean()}, std={value.std()}"
                    )
                else:
                    self._logger.info(f"[{key}] shape={value.shape}")
                self._logger.info(f"[{key}] Sample input: {value[0]}")
        if "stage" in kwargs:
            kwargs.pop("stage")
        return self._training_step(*args, **kwargs)

    def evaluation_step(self, *args, **kwargs) -> None:
        return self._evaluation_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> None:
        if "stage" in kwargs:
            kwargs.pop("stage")
        return self._predict_step(*args, **kwargs)

    def visualization_step(self, *args, **kwargs) -> None:
        if "stage" in kwargs:
            kwargs.pop("stage")
        return self._visualization_step(*args, **kwargs)

    def setup_ema(self):
        if (
            self._args.training_args.model_ema_args.enabled
            and self._ema_handler is None
        ):  # if ema handler has not been setup already, set it up for the first time
            # EMA takes care of DDP so if model is DDP model here it will get taken care of so we move ModuleProxyWrapper after applying EMA.
            module = self._torch_model
            if isinstance(module, ModuleProxyWrapper):
                module = module.module

            self._ema_handler = FusionEMAHandler(
                module,
                momentum=self._args.training_args.model_ema_args.momentum,
                momentum_warmup=self._args.training_args.model_ema_args.momentum_warmup,
                warmup_iters=self._args.training_args.model_ema_args.warmup_iters,
                handle_buffers="update",
            )

    def update_ema_for_stage(self, stage: TrainingStage):
        if stage == TrainingStage.train:
            if self._ema_activated:
                self._logger.info("Resetting model weights to training weights")
                # deactivate ema
                if isinstance(self._ema_handler, dict):
                    for key, ema_handler in self._ema_handler.items():
                        ema_handler.swap_params()
                else:
                    self._ema_handler.swap_params()
                self._ema_activated = False
        else:
            # replace ema model for validation stage available
            if (
                self._ema_handler is not None
                and self._args.training_args.use_ema_for_val
            ):
                if not self._ema_activated:
                    self._logger.info(
                        "Updating model weights from EMA weights for evaluation"
                    )
                    if isinstance(self._ema_handler, dict):
                        for key, ema_handler in self._ema_handler.items():
                            ema_handler.swap_params()
                    else:
                        self._ema_handler.swap_params()
                    self._ema_activated = True

    def wrap_dist(self, module):
        if isinstance(module, (DistributedDataParallel, DataParallel)):
            module = ModuleProxyWrapper(module)
        return module

    def setup_model(self, setup_for_train):
        # replace batch norm with group norm if required
        if self._args.model_args.convert_bn_to_gn:
            batch_norm_to_group_norm(self._torch_model)

        if setup_for_train:
            self._init_weights()

        # put model to device
        self._torch_model = self.module_to_device(self._torch_model)

        # setup model ema
        if setup_for_train:
            self.setup_ema()

    def module_to_device(
        self,
        module: nn.Module,
        device="gpu",
    ) -> Union[nn.Module, FusionModel]:
        self._logger.info(f"Putting {type(module)} to {device}.")
        module_requires_grad = False
        for p in module.parameters():
            if p.requires_grad:
                module_requires_grad = True
                break

        device = idist.device() if device == "gpu" else device
        if module_requires_grad:
            module = idist.auto_model(
                module,
                sync_bn=(
                    False
                    if device == torch.device("cpu")
                    else self._args.training_args.sync_batchnorm
                ),
            )
            module = self.wrap_dist(module)
        else:
            module.to(device)

        return module

    def nn_modules(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, nn.Module)}

    def summarize_model(self):
        from torchinfo import summary

        logger = get_logger()

        # get all nn modules in self
        for k, v in self.nn_modules().items():
            logger.info(f"Model component [{k}]:")
            logger.info(summary(v, verbose=0, depth=2))

    def get_checkpoint_state_dict(self) -> None:
        """
        Called from checkpoint connector when saving checkpoints
        """

        # initialize checkpoint dict
        checkpoint_state_dict = {}

        # add model to checkpoint
        checkpoint_state_dict["model"] = self.torch_model

        # if ema model is available, save it
        if self._ema_handler is not None:
            if isinstance(self._ema_handler, dict):
                for key, ema_handler in self._ema_handler.items():
                    checkpoint_state_dict[f"ema_model_{key}"] = ema_handler.ema_model
                    checkpoint_state_dict[f"ema_momentum_scheduler_{key}"] = (
                        ema_handler.momentum_scheduler
                    )
            else:
                checkpoint_state_dict["ema_model"] = self._ema_handler.ema_model
                checkpoint_state_dict["ema_momentum_scheduler"] = (
                    self._ema_handler.momentum_scheduler
                )

        return checkpoint_state_dict
