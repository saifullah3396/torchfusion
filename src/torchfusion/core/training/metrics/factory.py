from ignite.metrics import Accuracy, Precision, Recall

from torchfusion.core.args.args_base import ClassInitializerArgs
from torchfusion.core.constants import DataKeys, MetricKeys
from torchfusion.core.data.utilities.containers import MetricsDict
from torchfusion.core.models.fusion_model import FusionModel
from torchfusion.core.training.metrics.seqeval import create_seqeval_metric


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
        for metric_config in metric_args:
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
        if model_task in ["image_classification", "sequence_classification"]:
            if metric_name not in [
                MetricKeys.ACCURACY,
                MetricKeys.PRECISION,
                MetricKeys.RECALL,
                MetricKeys.F1,
            ]:
                raise ValueError(
                    f"Metric {metric_name} not supported for model task {model_task}"
                )

            assert labels is not None, "Labels required for classification tasks"

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
                # labels=labels,
                **metric_kwargs,
            )

        elif model_task in ["token_classification"]:
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
