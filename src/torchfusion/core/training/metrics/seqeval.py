from typing import Callable

from ignite.metrics import EpochMetric
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def modify_labels(class_labels, preds, targets):
    # convert tensor to list
    preds = preds.detach().cpu().tolist()
    preds = [
        [
            class_labels[pred_label]
            for (pred_label, target_label) in zip(prediction, target)
            if target_label != -100
        ]
        for prediction, target in zip(preds, targets)
    ]

    # convert tensor to list
    targets = targets.detach().cpu().tolist()

    # remove -100 labels from targets
    targets = [
        [class_labels[target_label] for target_label in target if target_label != -100]
        for target in targets
    ]

    return preds, targets


def create_seqeval_metric(
    class_labels, fn="accuracy", output_transform: Callable = None
):
    if fn == "accuracy":

        def wrap(preds, targets):  # reversed targets and preds
            preds, targets = modify_labels(class_labels, preds, targets)

            return accuracy_score(targets, preds)

        return EpochMetric(wrap, output_transform=output_transform)

    if fn == "f1":

        def wrap(preds, targets):  # reversed targets and preds
            preds, targets = modify_labels(class_labels, preds, targets)
            return f1_score(targets, preds, scheme="IOB2")

        return EpochMetric(wrap, output_transform=output_transform)

    if fn == "precision":

        def wrap(preds, targets):  # reversed targets and preds
            preds, targets = modify_labels(class_labels, preds, targets)

            return precision_score(targets, preds, scheme="IOB2")

        return EpochMetric(wrap, output_transform=output_transform)

    if fn == "recall":

        def wrap(preds, targets):  # reversed targets and preds
            preds, targets = modify_labels(class_labels, preds, targets)

            return recall_score(targets, preds, scheme="IOB2")

        return EpochMetric(wrap, output_transform=output_transform)

    if fn == "classification_report":

        def wrap(preds, targets):  # reversed targets and preds
            preds, targets = modify_labels(class_labels, preds, targets)

            return classification_report(targets, preds, scheme="IOB2")

        return EpochMetric(wrap, output_transform=output_transform)
