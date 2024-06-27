from dataclasses import dataclass
from typing import Optional

import torch
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.classification.base import FusionModelForClassification
from torchfusion.core.models.constructors.factory import ModelConstructorFactory
from torchfusion.core.models.constructors.fusion import FusionModelConstructor
from torchfusion.core.models.constructors.transformers import (
    TransformersModelConstructor,
)
from torchfusion.core.models.utilities.data_collators import (
    BatchToTensorDataCollator,
    SequenceTokenizerDataCollator,
)
from torchfusion.core.training.utilities.constants import TrainingStage


class FusionModelForTokenClassification(FusionModelForClassification):
    _SUPPORTS_CUTMIX = False
    _SUPPORTS_KD = False

    @dataclass
    class Config(FusionModelForClassification.Config):
        use_bbox: bool = True
        use_image: bool = True
        input_is_tokenized: bool = True
        input_stride: int = 0

    def _prepare_input(self, engine, batch, tb_logger, **kwargs):
        assert (
            DataKeys.TOKEN_IDS in batch
        ), "No tokenized input found in batch. Did you forget to tokenize the dataset?"

        inputs = dict(
            input_ids=batch[DataKeys.TOKEN_IDS],
            attention_mask=batch[DataKeys.ATTENTION_MASKS],
        )
        if self.config.use_image:
            inputs["pixel_values"] = batch[DataKeys.IMAGE]
        if self.config.use_bbox:
            inputs["bbox"] = batch[DataKeys.TOKEN_BBOXES]

        return inputs

    def _prepare_label(self, engine, batch, tb_logger, **kwargs):
        return batch[self._LABEL_KEY]

    def _build_classification_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
        model_constructor: Optional[dict] = None,
        model_constructor_args: Optional[dict] = None,
        num_labels: Optional[int] = None,
    ):
        model_constructor = ModelConstructorFactory.create(
            name=model_constructor,
            kwargs=model_constructor_args,
        )
        assert isinstance(
            model_constructor,
            (TransformersModelConstructor, FusionModelConstructor),
        ), (
            f"Model constructor must be of type TransformersModelConstructor. "
            f"Got {type(model_constructor)}"
        )
        return model_constructor(
            checkpoint=checkpoint, strict=strict, num_labels=num_labels
        )

    def _build_model(
        self,
        checkpoint: Optional[str] = None,
        strict: bool = False,
    ):
        return self._build_classification_model(
            checkpoint=checkpoint,
            strict=strict,
            model_constructor=self.config.model_constructor,
            model_constructor_args=self.config.model_constructor_args,
            num_labels=self.config.num_labels,
        )

    def _training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        assert self._LABEL_KEY in batch, "Label must be passed for training"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        hf_output = self._model_forward({**input, "labels": label})

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: hf_output.loss,
                DataKeys.LOGITS: hf_output.logits,
                self._LABEL_KEY: label,
            }
        else:
            return (hf_output.loss, hf_output.logits, label)

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
        hf_output = self._model_forward({**input, "labels": label})

        # here we check if the input is strided or not. With strided input tokenization, the first "number of stride"
        # tokens are to be ignored for evaluation as they will be repeated tokens from the previous part of the documnet
        # first we check if there are overflowing samples in the batch and if so for these tokens first N stride tokens
        # are to be ignored for evaluation
        for sample_idx, word_ids in enumerate(batch[DataKeys.WORD_IDS]):
            if word_ids[word_ids != -100].min() > 0:
                label[sample_idx][: self.config.input_stride] = -100

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: hf_output.loss,
                DataKeys.LOGITS: hf_output.logits,
                DataKeys.PRED: hf_output.logits.argmax(dim=-1),
                self._LABEL_KEY: label,
            }
        else:
            return (
                hf_output.loss,
                hf_output.logits,
                hf_output.logits.argmax(dim=-1),
                label,
            )

    def _predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        hf_output = self._model_forward(input)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOGITS: hf_output.logits,
                DataKeys.LOGITS: hf_output.logits,
            }
        else:
            return (
                hf_output.logits,
                hf_output.logits.argmax(dim=-1),
            )

    def _model_forward(self, input):
        if isinstance(input, dict):
            return self.torch_model(**input)
        else:
            return self.torch_model(input)

    def get_data_collators(
        self, data_key_type_map=None, tokenizer=None
    ) -> CollateFnDict:
        if self.config.input_is_tokenized:
            collate_fn_class = BatchToTensorDataCollator
        else:
            assert (
                tokenizer is not None
            ), "Tokenizer must be provided for token classification if input is not tokenzized"
            collate_fn_class = SequenceTokenizerDataCollator
        if data_key_type_map is None:
            data_key_type_map = {
                DataKeys.TOKEN_IDS: torch.long,
                DataKeys.TOKEN_TYPE_IDS: torch.long,
                DataKeys.ATTENTION_MASKS: torch.long,
                DataKeys.WORD_IDS: torch.long,
                self._LABEL_KEY: torch.long,
            }
        else:
            data_key_type_map[DataKeys.TOKEN_IDS] = torch.long
            data_key_type_map[DataKeys.TOKEN_TYPE_IDS] = torch.long
            data_key_type_map[DataKeys.ATTENTION_MASKS] = torch.long
            data_key_type_map[DataKeys.WORD_IDS] = torch.long
            data_key_type_map[self._LABEL_KEY] = torch.long

        if self.model_args.model_config.use_bbox:
            data_key_type_map[DataKeys.TOKEN_BBOXES] = torch.long

        if self.model_args.model_config.use_image:
            data_key_type_map[DataKeys.IMAGE] = torch.float

        if collate_fn_class == BatchToTensorDataCollator:
            collate_fn = collate_fn_class(
                data_key_type_map=data_key_type_map,  # make sure we only get what we want not any extra stuff from dataset
                allowed_keys=list(data_key_type_map.keys()),
            )
        elif collate_fn_class == SequenceTokenizerDataCollator:
            collate_fn = collate_fn_class(
                data_key_type_map=data_key_type_map,
                tokenizer=tokenizer,
                allowed_keys=list(data_key_type_map.keys()),
            )
        else:
            raise ValueError(f"Unknown collate function class {collate_fn_class}")

        # initialize the data collators for bert grid based word classification
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)
