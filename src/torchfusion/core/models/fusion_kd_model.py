""" PyTorch module that defines the base model for training/testing etc. """

from __future__ import annotations

from operator import is_
from typing import Any, Dict, Optional

import ignite.distributed as idist
import torch
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from torchfusion.core.args.args import FusionArguments, FusionKDArguments
from torchfusion.core.constants import DataKeys
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.utilities.ddp_model_proxy import ModuleProxyWrapper
from torchfusion.core.models.utilities.general import batch_norm_to_group_norm
from torchfusion.core.models.utilities.knowledge_distillation import (
    EnsembleKnowledgeTransferLoss,
    GaussianLoss,
    TemperatureScaledKLDivLoss,
)
from torchfusion.core.training.args.ema import FusionEMAHandler
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.utilities.logging import get_logger


class FusionKDModel(FusionModel):
    def __init__(
        self,
        args: FusionArguments,
        tb_logger: Optional[TensorboardLogger] = None,
        dataset_features: Optional[dict] = None,
    ):
        super().__init__()

        # initialize arguments
        self._args = args
        self._tb_logger = tb_logger
        self._dataset_features = dataset_features

        # ema parameters
        self._ema_handler = None
        self._ema_activated = False
        self._stored_parameters = None

        # build model
        if self._args.model_mode == "student":
            self._student_nn_model = self._build_model(is_student=True)
        elif self._args.model_mode == "teacher":
            self._teacher_nn_model = self._build_model(is_student=False)
        elif self._args.model_mode == "distillation":
            self._student_nn_model = self._build_model(is_student=True)
            self._teacher_nn_model = self._build_model(is_student=False)
            # set requires grad
            for param in self._teacher_nn_model.parameters():
                param.requires_grad = False
            self._teacher_nn_model.eval()

        # initialize logger
        self._logger = get_logger()

    def _build_model(self, is_student=True):
        from torchfusion.core.models.factory import ModelFactory

        # models sometimes download pretrained checkpoints when initializing. Only download it on rank 0
        if idist.get_rank() > 0:  # stop all ranks > 0
            idist.barrier()

        model_args = (
            self._args.student_model_args
            if is_student
            else self._args.teacher_model_args
        )
        if model_args.model_task in [
            "image_classification",
            "sequence_classification",
            "token_classification",
        ]:
            if DataKeys.LABEL not in self._dataset_features:
                raise ValueError(
                    "class_labels are required in dataset_features for image_classification tasks."
                )

        model_class = ModelFactory.get_fusion_nn_model_class(model_args)
        nn_model = model_class(
            model_args=model_args,
            data_args=self._args.data_args,
            training_args=self._args.training_args,
            dataset_features=self._dataset_features,
        )

        # build model
        nn_model.build_model()

        # wait for rank 0 to download checkpoints
        if idist.get_rank() == 0:
            idist.barrier()

        return nn_model

    def training_step(self, *args, **kwargs) -> None:
        # remove stage from kwargs
        if "stage" in kwargs:
            kwargs.pop("stage")

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
        if self._args.model_mode == "student":
            return self._student_nn_model.training_step(*args, **kwargs)
        elif self._args.model_mode == "teacher":
            return self._teacher_nn_model.training_step(*args, **kwargs)
        elif self._args.model_mode == "distillation":
            assert hasattr(
                self._student_nn_model, "kd_training_step"
            ), "kd_training_step should be implemented in the student model for distillation training"
            return self._student_nn_model.kd_training_step(
                *args, **kwargs, teacher=self._teacher_nn_model
            )

    def evaluation_step(self, *args, **kwargs) -> None:
        assert "is_student" in kwargs, "is_student should be provided in kwargs"
        if (
            is_student
        ):  # if we do forward pass on student model, we need to make sure that the model is in student mode
            assert self._args.model_mode in [
                "student",
                "distillation",
            ], f"train_mode should be student or distillation, but got {self._args.model_mode}"
        else:  # if we do forward pass on teacher model, we need to make sure that the model is in teacher mode
            assert self._args.model_mode in [
                "teacher",
                "distillation",
            ], f"train_mode should be teacher or distillation, but got {self._args.model_mode}"

        is_student = kwargs.pop("is_student")

        if is_student:
            return self._student_nn_model.evaluation_step(*args, **kwargs)
        else:
            return self._teacher_nn_model.evaluation_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> None:
        assert "is_student" in kwargs, "is_student should be provided in kwargs"
        if (
            is_student
        ):  # if we do forward pass on student model, we need to make sure that the model is in student mode
            assert self._args.model_mode in [
                "student",
                "distillation",
            ], f"train_mode should be student or distillation, but got {self._args.model_mode}"
        else:  # if we do forward pass on teacher model, we need to make sure that the model is in teacher mode
            assert self._args.model_mode in [
                "teacher",
                "distillation",
            ], f"train_mode should be teacher or distillation, but got {self._args.model_mode}"

        is_student = kwargs.pop("is_student")

        if "stage" in kwargs:
            kwargs.pop("stage")
        if is_student:
            return self._student_nn_model.predict_step(*args, **kwargs)
        else:
            return self._teacher_nn_model.predict_step(*args, **kwargs)

    def put_modules_to_device(self):
        if hasattr(self, "_student_nn_model"):
            self._student_nn_model = self.module_to_device(self._student_nn_model)
        if hasattr(self, "_teacher_nn_model"):
            self._teacher_nn_model = self.module_to_device(self._teacher_nn_model)

    def setup_model(self, setup_for_train):
        # replace batch norm with group norm if required
        if self._args.student_model_args.convert_bn_to_gn:
            batch_norm_to_group_norm(self._student_nn_model)
        if self._args.teacher_model_args.convert_bn_to_gn:
            batch_norm_to_group_norm(self._teacher_nn_model)

        if setup_for_train:
            # if required reinit some weights
            self._student_nn_model._reinit_weights()

        if hasattr(self, "_student_nn_model"):
            self._student_nn_model = self.module_to_device(self._student_nn_model)
        if hasattr(self, "_teacher_nn_model"):
            self._teacher_nn_model = self.module_to_device(self._teacher_nn_model)

        if setup_for_train:
            self.setup_ema()

    @property
    def ema_handler(self):
        return self._ema_handler

    @property
    def model_name(self):
        return self.torch_model.model_name

    @property
    def student(self):
        return self._student_nn_model

    @property
    def teacher(self):
        return self._teacher_nn_model

    @property
    def torch_model(self):
        # torch model is the student model if model_mode is student or distil, else teacher model
        if self._args.model_mode == "teacher":
            return self._teacher_nn_model
        elif self._args.model_mode == "student":
            return self._student_nn_model
        else:
            return self._student_nn_model

    def summarize_model(self):
        from torchinfo import summary

        logger = get_logger()

        if hasattr(self, "_teacher_nn_model"):
            logger.info("Teacher model:")
            logger.info(summary(self._teacher_nn_model, verbose=0, depth=4))

        if hasattr(self, "_student_nn_model"):
            logger.info("Student model:")
            logger.info(summary(self._student_nn_model, verbose=0, depth=4))

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

        # add additional information from the underlying model if needed
        checkpoint_state_dict = {
            **checkpoint_state_dict,
            **self.torch_model.get_checkpoint_state_dict(),
        }

        return checkpoint_state_dict
