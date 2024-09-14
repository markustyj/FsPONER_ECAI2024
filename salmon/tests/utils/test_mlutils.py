"""Tests for mlutils."""
import numpy as np
import pytest
from transformers import EvalPrediction

from salmon.utils.mlutils import compute_classification_metrics
from salmon.utils.mlutils import compute_lm_perplexity


@pytest.mark.parametrize(
    "predictions, labels, expected_accuracy",
    [
        # perfect accuracy on a two class prediction
        (
            np.array([[1, 0], [1, 0], [0, 1], [0, 1]]),
            np.array([0, 0, 1, 1]),
            1.0,
        ),
        # one error on a two class prediction
        (
            np.array([[1, 0], [1, 0], [0, 1], [0, 1]]),
            np.array([0, 0, 1, 0]),
            0.75,
        ),
    ],
)
def test_compute_classification_metrics(predictions, labels, expected_accuracy):
    """Test the classification metrics."""
    eval_predictions = EvalPrediction(predictions, labels)

    metrics = compute_classification_metrics(eval_predictions)

    assert metrics["accuracy"] == expected_accuracy


@pytest.mark.parametrize(
    "evaluation_results, expected_perplexity",
    [
        # perplexity of 0 loss
        (
            {"eval_loss": 0},
            1.0,
        ),
        # perplexity of an int loss
        (
            {"eval_loss": 10},
            22026.465794806718,
        ),
        # perplexity of an less than 1 loss
        (
            {"eval_loss": 0.8},
            2.225540928492468,
        ),
        # perplexity of an greater than 1 float loss
        (
            {"eval_loss": 1.8},
            6.0496474644129465,
        ),
    ],
)
def test_compute_lm_perplexity(evaluation_results, expected_perplexity):
    """Test perplexity computation."""
    perplexity = compute_lm_perplexity(evaluation_results)
    assert perplexity == expected_perplexity
