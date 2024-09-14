"""Downstream task tests."""
import logging
import warnings

import pytest

from salmon.downstream.ner_finetune import transformer_finetune_ner_downstream
from salmon.downstream.text_classification_finetune import (
    transformer_finetune_textcat_downstream,
)

logger = logging.getLogger(__name__)


@pytest.mark.training
def test_transformer_finetune_textcat_downstream(
    tmpdir,
    text_classification_training_file,
    text_classification_test_file,
    text_classification_val_file,
    tiny_base_lm,
    base_roberta_tokenizer,
):
    """Text classification fine-tuning test."""
    # Remove this filter when there's a fix from huggingface on this:
    # https://stackoverflow.com/q/71113363/2263844
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    model_name = "test_tc"
    logs_dir = "logs_dir"
    tmpdir_str = str(tmpdir)
    tmp_output_dir = f"{tmpdir_str}/{model_name}"
    tmp_output_log_dir = f"{tmpdir_str}/{logs_dir}"

    eval_before, eval_after = transformer_finetune_textcat_downstream(
        train_file=text_classification_training_file,
        val_file=text_classification_val_file,
        test_file=text_classification_test_file,
        base_model=tiny_base_lm,
        output_model=tmp_output_dir,
        logging_dir=tmp_output_log_dir,
        tokenizer_model=base_roberta_tokenizer,
        num_train_epochs=1,
        training_data_fraction=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        evaluate_before_training=True,
    )
    msg = "Evaluation before fine-tuning:"
    logger.info(msg)
    logger.info(eval_before)

    msg = "Evaluation after fine-tuning:"
    logger.info(msg)
    logger.info(eval_after)

    assert (
        abs(eval_before["eval_accuracy"] - eval_after["eval_accuracy"])
        < eval_before["eval_accuracy"] / 2
    )


@pytest.mark.training
def test_transformer_finetune_ner_downstream(
    tmpdir,
    ner_training_file,
    ner_val_file,
    ner_test_file,
    tiny_base_lm,
    base_roberta_tokenizer,
):
    """NER fine-tuning test."""
    # Remove this filter when there's a fix from huggingface on this:
    # https://stackoverflow.com/q/71113363/2263844
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    model_name = "test_ner"
    logs_dir = "logs_dir_ner"
    tmpdir_str = str(tmpdir)
    tmp_output_dir = f"{tmpdir_str}/{model_name}"
    tmp_output_log_dir = f"{tmpdir_str}/{logs_dir}"

    eval_before, eval_after = transformer_finetune_ner_downstream(
        train_file=ner_training_file,
        val_file=ner_val_file,
        test_file=ner_test_file,
        base_model=tiny_base_lm,
        output_model=tmp_output_dir,
        logging_dir=tmp_output_log_dir,
        tokenizer_model=base_roberta_tokenizer,
        num_train_epochs=4,
        training_data_fraction=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        evaluate_before_training=True,
        tokenizer_add_prefix_space=True,
    )
    msg = "Evaluation before fine-tuning:"
    logger.info(msg)
    logger.info(eval_before)

    msg = "Evaluation after fine-tuning:"
    logger.info(msg)
    logger.info(eval_after)

    assert (
        abs(
            eval_before["eval_overall_accuracy"]
            - eval_after["eval_overall_accuracy"]
        )
        < eval_before["eval_overall_accuracy"] / 2
    )
