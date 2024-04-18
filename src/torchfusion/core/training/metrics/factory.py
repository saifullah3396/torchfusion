from pathlib import Path

import ignite.distributed as idist
from ignite.metrics import Accuracy, Precision, Recall

from torchfusion.core.args.args_base import ClassInitializerArgs
from torchfusion.core.constants import DataKeys, MetricKeys
from torchfusion.core.data.utilities.containers import MetricsDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.models.tasks import ModelTasks
from torchfusion.core.training.metrics.seqeval import create_seqeval_metric
from torchfusion.core.training.utilities.general import pretty_print_dict
from torchfusion.utilities.logging import get_logger


def f1_score(output_transform):
    # use ignite arthematics of metrics to compute f1 score
    # unnecessary complication
    precision = Precision(average=False, output_transform=output_transform)
    recall = Recall(average=False, output_transform=output_transform)
    return (precision * recall * 2 / (precision + recall)).mean()


def get_ignite_metric_class(metric_name):
    if metric_name == MetricKeys.ACCURACY:
        return Accuracy
    elif metric_name == MetricKeys.PRECISION:
        return Precision
    elif metric_name == MetricKeys.RECALL:
        return Recall
    elif metric_name == MetricKeys.F1:
        return f1_score
    else:
        raise ValueError(f"Metric {metric_name} not supported")


class MetricsFactory:
    @staticmethod
    def initialize_stage_metrics(metric_args, model_task, labels=None):
        metrics = {}
        logger = get_logger()
        for metric_config in metric_args:
            assert isinstance(metric_config, ClassInitializerArgs), (
                f"Metric config must be of type {ClassInitializerArgs}. "
                f"Got {type(metric_config)}"
            )
            metrics[metric_config.name] = MetricsFactory.create_metric(
                metric_config.name,
                metric_config.kwargs,
                model_task=model_task,
                labels=labels,
            )

        return MetricsDict(
            train=metrics, validation=metrics, test=metrics, predict=metrics
        )

    @staticmethod
    def create_metric(metric_name: str, metric_kwargs: dict, model_task: str, labels):
        logger = get_logger()
        if model_task in [
            ModelTasks.image_classification,
            ModelTasks.sequence_classification,
        ]:
            if metric_name not in [
                MetricKeys.ACCURACY,
                MetricKeys.PRECISION,
                MetricKeys.RECALL,
                MetricKeys.F1,
            ]:
                raise ValueError(
                    f"Metric {metric_name} not supported for model task {model_task}"
                )

            assert (
                labels is not None
            ), "Labels required for classification tasks"  # is this really required here?

            def output_transform(output):
                assert (
                    DataKeys.LOGITS in output
                ), f"Logits not found in output: {output.keys()} required for metric {metric_name}"
                assert (
                    DataKeys.LABEL in output
                ), f"Logits not found in output: {output.keys()} required for metric {metric_name}"

                # ignite metrics accuracy, precision, recall, f1_score take logits as input instead of argmax'ed predictions
                return output[DataKeys.LOGITS], output[DataKeys.LABEL]

            return lambda: get_ignite_metric_class(metric_name)(
                output_transform=output_transform,
                **metric_kwargs,
            )

        elif model_task in [ModelTasks.token_classification]:
            if metric_name not in [
                MetricKeys.ACCURACY,
                MetricKeys.PRECISION,
                MetricKeys.RECALL,
                MetricKeys.F1,
            ]:
                raise ValueError(
                    f"Metric {metric_name} not supported for model task {model_task}"
                )

            assert labels is not None, "Labels required for token_classification tasks"

            def output_transform(output):
                assert (
                    DataKeys.LOGITS in output
                ), f"Logits not found in output: {output.keys()} required for metric {metric_name}"
                assert (
                    DataKeys.LABEL in output
                ), f"Logits not found in output: {output.keys()} required for metric {metric_name}"
                return output[DataKeys.LOGITS].argmax(dim=2), output[DataKeys.LABEL]

            return lambda: create_seqeval_metric(
                labels,
                fn=metric_name,
                output_transform=output_transform,
            )

        elif model_task in [ModelTasks.autoencoding, ModelTasks.gan]:
            if metric_name not in [MetricKeys.FID]:
                raise ValueError(
                    f"Metric {metric_name} not supported for model task {model_task}"
                )

            from pytorch_fid.inception import InceptionV3

            from torchfusion.core.training.metrics.fid_metric import (
                FID,
                WrapperInceptionV3,
            )

            def output_transform(output):
                # FID takes the output:
                #   train, test = output
                # train refers to the predictions, and test refers to the target dataset statistics

                # pred, real or fake, real,
                # Do not send image first!!
                return output[DataKeys.RECONS], output[DataKeys.IMAGE]

            # pytorch_fid model
            dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx])

            # wrapper model to pytorch_fid model
            wrapper_model = WrapperInceptionV3(model)
            wrapper_model.requires_grad_(False)
            wrapper_model.eval()

            # get fid path
            assert (
                "fid_stats" in metric_kwargs
            ), "fid stats path must be provided for the fid metric. You can compute it using args.data_args.compute_dataset_statistics=True"

            fid_stats = Path(metric_kwargs["fid_stats"])
            assert fid_stats.exists(), f"fid stats file {fid_stats} does not exist"
            logger.info(f"Using fid statistics from fid stats file: {fid_stats}")

            # here we only need validation
            return lambda: FID(
                num_features=dims,
                feature_extractor=wrapper_model,
                output_transform=output_transform,
                ckpt_path=fid_stats,
                device=idist.get_rank(),
            )
