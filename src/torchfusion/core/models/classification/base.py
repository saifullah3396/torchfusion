from abc import abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Optional

import torch
from datasets.features import Sequence
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.utilities.knowledge_distillation import (
    EnsembleKnowledgeTransferLoss,
    GaussianLoss,
    TemperatureScaledKLDivLoss,
)
from torchfusion.core.training.utilities.constants import TrainingStage


class FusionModelForClassification(FusionModel):
    _SUPPORTS_CUTMIX = False
    _SUPPORTS_KD = False
    _LABEL_KEY = DataKeys.LABEL

    @dataclass
    class Config(FusionModel.Config):
        pass

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._logger.info(
            "Initialized the model with data labels: {}".format(self.labels)
        )

    @property
    def labels(self):
        if isinstance(self._dataset_features[self._LABEL_KEY], Sequence):
            return self._dataset_features[self._LABEL_KEY].feature.names
        else:
            return self._dataset_features[self._LABEL_KEY].names

    @property
    def num_labels(self):
        return len(self.labels)

    @abstractmethod
    def _build_classification_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        model_constructor: Optional[dict] = None,
        model_constructor_args: Optional[dict] = None,
    ):
        pass

    def _build_model(self, checkpoint: Optional[str] = None, strict: bool = False):
        model = self._build_classification_model(
            checkpoint=checkpoint,
            strict=strict,
            model_constructor=self.config.model_constructor,
            model_constructor_args=self.config.model_constructor_args,
        )

        # initialize classification related stuff
        self.loss_fn_train = torch.nn.CrossEntropyLoss()
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

        # setup mixup function
        self.mixup_fn = None
        if self._SUPPORTS_CUTMIX and self.training_args.cutmixup_args is not None:
            self.mixup_fn = self.training_args.cutmixup_args.get_fn(
                num_classes=self.num_labels, smoothing=self.training_args.smoothing
            )

        # setup loss accordingly if mixup, or label smoothing is required
        if self.mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.loss_fn_train = SoftTargetCrossEntropy()
        elif self.training_args.smoothing > 0.0:
            self.loss_fn_train = LabelSmoothingCrossEntropy(
                smoothing=self.training_args.smoothing
            )
        else:
            self.loss_fn_train = torch.nn.CrossEntropyLoss()
        self.loss_fn_eval = torch.nn.CrossEntropyLoss()

        if self.model_args.kd_args is not None and self._SUPPORTS_KD:
            # get kd arguments
            kd_args = self.model_args.kd_args

            if kd_args.model_mode in ["teacher", "distillation"]:
                # initialize the teacher model using different model parameters
                self._teacher_model = self._build_classification_model(
                    checkpoint=checkpoint,
                    strict=strict,
                    model_constructor=self.config.model_constructor,
                    model_constructor_args=kd_args.teacher_model_constructor_args,
                )

                if (
                    kd_args.model_mode == "teacher"
                ):  # if we want to train the teacher we discard student model
                    return self._teacher_model
                elif (
                    kd_args.model_mode == "distillation"
                ):  # if we want to train the student model with transfer we keep both
                    # set requires grad false
                    for param in self._teacher_model.parameters():
                        param.requires_grad = False

                    self._teacher_model = self.module_to_device(self._teacher_model)
                    self._teacher_model.eval()

                    # build teacher training parameters
                    teacher_logit_criterion = TemperatureScaledKLDivLoss(
                        temperature=kd_args.temperature
                    )
                    teacher_feature_criterion = GaussianLoss()
                    self.loss_fn_train = EnsembleKnowledgeTransferLoss(
                        label_criterion=self.loss_fn_train,
                        teacher_logit_criterion=teacher_logit_criterion,
                        teacher_feature_criterion=teacher_feature_criterion,
                        teacher_logit_factor=kd_args.knowledge_distillation_factor,
                        teacher_feature_factor=kd_args.variational_information_distillation_factor,
                    )

                    return model
            elif (
                kd_args.model_mode == "student"
            ):  # if we want to train student model without transfer we keep only student model
                return model
            else:
                raise ValueError(
                    f"Invalid model mode for knowledge distillation: {kd_args.model_mode}"
                )

        return model

    def _training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        assert self._LABEL_KEY in batch, "Label must be passed for training"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        if (
            self.model_args.kd_args is not None
            and self._SUPPORTS_KD
            and self.model_args.kd_args.model_mode == "distillation"
        ):
            with torch.no_grad():
                if isinstance(input, dict):
                    teacher_logit, teacher_features = self._teacher_model(**input)
                else:
                    teacher_logit, teacher_features = self._teacher_model(input)

            logit, teacher_feature_preds = self._model_forward(input)
            loss = self.loss_fn_train(
                logit=logit,
                label=label,
                teacher_feature_preds=teacher_feature_preds,
                teacher_logit=teacher_logit,
                teacher_features=teacher_features,
            )
        else:
            # compute logits
            logits = self._model_forward(input)

            # compute loss
            loss = self.loss_fn_train(logits, label)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                self._LABEL_KEY: label,
            }
        else:
            return (loss, logits, label)

    def _evaluation_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> None:
        assert self._LABEL_KEY in batch, "Label must be passed for evaluation"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits = self._model_forward(input)

        # compute loss
        loss = self.loss_fn_eval(logits, label)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                self._LABEL_KEY: label,
            }
        else:
            return (loss, logits, label)

    def _predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits = self._model_forward(input)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOGITS: logits,
            }
        else:
            return (logits,)

    def _model_forward(self, input, return_logits=True):
        if isinstance(input, dict):
            # compute logits
            output = self.torch_model(**input)
        else:
            output = self.torch_model(input)

        # we assume first element is logits
        if isinstance(output, torch.Tensor):  # usually timm returns a tensor directly
            return output
        elif isinstance(output, tuple):  # usually timm returns a tensor directly
            if return_logits:
                return output[0]
            else:
                return output
        elif is_dataclass(output):  # usually huggingface returns a dataclass
            if return_logits:
                return getattr(output, "logits")
            else:
                return output

    @abstractmethod
    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        pass

    @abstractmethod
    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        pass

    @abstractmethod
    def get_data_collators(
        self,
        data_key_type_map: Optional[dict] = None,
    ) -> CollateFnDict:
        pass
