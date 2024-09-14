"""Utility functions for ML."""
import logging
import math
import warnings
from typing import Dict
from typing import List

import numpy as np
from datasets import load_metric
from sklearn.exceptions import UndefinedMetricWarning
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def compute_ner_metrics(
    eval_pred: EvalPrediction, label_list: List
) -> Dict[str, float]:
    """
    Compute NER metrics.

    Parameters
    ----------
    eval_pred: (logits, labels) tuple.
    label_list: list of NER labels.

    Returns
    -------
    dict
    A dictionary with the following keys:
    - accuracy
    - f1
    - precision
    - recall
    - {NER_type}_f1
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    metric = load_metric("seqeval")

    true_predictions = [
        [
            label_list[pred]
            for (pred, lab) in zip(prediction, label)
            if lab != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [
            label_list[lab]
            for (pred, lab) in zip(prediction, label)
            if lab != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]
    # seqeval.compute does not allow selecting average=macro
    # the default is micro
    # ref: https://github.com/huggingface/datasets
    # /blob/master/metrics/seqeval/seqeval.py#L147
    results = metric.compute(
        predictions=true_predictions,
        references=true_labels,
        zero_division=0  # add by Yongjian to avoid zero-devision error
    )
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for key in results.keys():
        if key not in flattened_results:
            flattened_results[key + "_f1"] = results[key]["f1"]
    return flattened_results


def compute_classification_metrics(
    eval_pred: EvalPrediction,
) -> Dict[str, float]:
    """
    Compute classification metrics for the given predictions.

    The compute function needs to receive a tuple (with logits and labels)
    and has to return a dictionary with string keys
    (the name of the metric) and float values. It will be called at
    the end of each evaluation phase on the whole arrays of
    predictions/labels.

    This function uses prediction evaluation code from HuggingFace.

    Parameters
    ----------
    eval_pred: (logits, labels) tuple.

    Returns
    -------
    dict
    A dictionary with the following keys:
    - accuracy
    - f1
    - precision
    - recall
    """
    # Huggingface load_metrics for f1 and precision
    # haven't implemented the zero_division param. So muting the warnings
    # until they do so.
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    metric_prec = load_metric("precision")
    metric_rec = load_metric("recall")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_val = metric_acc.compute(
        predictions=predictions, references=labels
    )["accuracy"]

    f1_val = metric_f1.compute(
        predictions=predictions, references=labels, average="macro"
    )["f1"]

    prec_val = metric_prec.compute(
        predictions=predictions, references=labels, average="macro"
    )["precision"]

    recall_val = metric_rec.compute(
        predictions=predictions, references=labels, average="macro"
    )["recall"]

    return {
        "accuracy": accuracy_val,
        "f1": f1_val,
        "precision": prec_val,
        "recall": recall_val,
    }


def compute_lm_perplexity(eval_results: Dict[str, float]) -> float:
    """
    Compute perplexity of a language model.

    Parameters
    ----------
    eval_results: evaluation result dictionary return by `trainer.evaluate()`

    Returns
    -------
    Perplexity of the language model.
    """
    perplexity = math.exp(eval_results["eval_loss"])
    return perplexity
