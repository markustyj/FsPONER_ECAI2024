"""MLM pre-training tests."""
import logging
import warnings

import pytest
from datasets import Dataset
from datasets import DatasetDict
from datasets import disable_caching
from datasets import load_dataset
from transformers import AutoTokenizer

from salmon.pre_training.pre_train_masked_lm import group_block_texts
from salmon.pre_training.pre_train_masked_lm import (
    transformer_masked_lm_pretrain,
)
from salmon.utils.preprocessing import drop_features_except

logger = logging.getLogger(__name__)


@pytest.mark.training
def test_pre_train_masked_lm(
    tmpdir, lm_training_file, lm_test_file, tiny_base_lm, base_roberta_tokenizer
):
    """MLM pre-training Test case."""
    # Remove this filter when there's a fix from huggingface on this:
    # https://stackoverflow.com/q/71113363/2263844
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    model_name = "test_mlm"
    logs_dir = "logs_dir"
    tmpdir_str = str(tmpdir)
    tmp_output_dir = f"{tmpdir_str}/{model_name}"
    tmp_output_log_dir = f"{tmpdir_str}/{logs_dir}"

    perplexity_before, perplexity_after = transformer_masked_lm_pretrain(
        train_file=lm_training_file,
        test_file=lm_test_file,
        base_model=tiny_base_lm,
        output_model=tmp_output_dir,
        logging_dir=tmp_output_log_dir,
        tokenizer_model=base_roberta_tokenizer,
        grouping_block_size=256,
        grouping_batch_size=512,
        num_train_epochs=1,
    )
    msg = f"Tempdir: {tmpdir_str}"
    logger.info(msg)
    msg = f"Perplexity before: {perplexity_before}"
    logger.info(msg)
    msg = f"Perplexity after: {perplexity_after}"
    logger.info(msg)
    # perplexity improvement or deterioration after domain adaptation
    # is less than 50% of the perplexity before domain adaptation
    assert abs(perplexity_before - perplexity_after) < perplexity_before / 2


@pytest.mark.parametrize(
    "examples, grp_size, expected_num_grps, expected_labels",
    [
        # base case
        (
            {"input_ids": [[2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1]]},
            2,
            2,
            [[2, 3], [4, 5]],
        ),
        # trailing element which gets dropped
        (
            {
                "input_ids": [[2, 3, 4, 5, 6]],
                "attention_mask": [[1, 1, 1, 1, 1]],
            },
            2,
            2,
            [[2, 3], [4, 5]],
        ),
        # alternative group size
        (
            {"input_ids": [[2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1]]},
            4,
            1,
            [[2, 3, 4, 5]],
        ),
        # multiple documents
        (
            {
                "input_ids": [[2, 3, 4, 5], [6, 7, 8, 9]],
                "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            },
            4,
            2,
            [[2, 3, 4, 5], [6, 7, 8, 9]],
        ),
        # multiple documents get concatenated, extra elements are dropped
        (
            {
                "input_ids": [[2, 3, 4, 5], [6, 7, 8, 9]],
                "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            },
            6,
            1,
            [[2, 3, 4, 5, 6, 7]],
        ),
        (
            {
                "input_ids": [[2, 3, 4, 5]],
                "attention_mask": [[1, 1, 1, 1]],
                "additional_numeric_field": [[6, 7, 8, 9]],
            },
            2,
            2,
            [[2, 3], [4, 5]],
        ),
        (
            {
                "input_ids": [[2, 3, 4, 5]],
                "attention_mask": [[1, 1, 1, 1]],
                "additional_alphanumeric_field": [["a", "b", "c", "d"]],
            },
            2,
            2,
            [[2, 3], [4, 5]],
        ),
    ],
)
def test_group_block_texts(
    examples, grp_size, expected_num_grps, expected_labels
):
    """Check that tokenised data is grouped correctly."""
    result = group_block_texts(examples, grouping_block_size=grp_size)

    assert result["input_ids"] == result["labels"]
    assert len(result["input_ids"]) == len(result["attention_mask"])
    assert len(result["input_ids"]) == expected_num_grps
    assert len(result["input_ids"][0]) == grp_size
    assert result["labels"] == expected_labels


def test_group_block_texts_via_datasets(
    lm_training_file, lm_test_file, base_roberta_tokenizer
):
    """Test grouping texts."""
    disable_caching()

    datasets_for_test = load_dataset(
        "json",
        data_files={
            "train": lm_training_file,
            "test": lm_test_file,
        },
    )

    datasets_for_test_text_only = drop_features_except(
        datasets_for_test, ["text"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_roberta_tokenizer, use_fast=True
    )

    tokenized_datasets = datasets_for_test_text_only.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )

    lm_datasets = tokenized_datasets.map(
        lambda x: group_block_texts(x, grouping_block_size=128),
        batched=True,
        batch_size=512,
        num_proc=4,
    )
    assert len(lm_datasets["train"]) == 270
    assert len(lm_datasets["test"]) == 344


@pytest.mark.parametrize(
    "input_dict, retain_fields, expected_error",
    [
        ({"a": [1, 2, 3], "b": [4, 5, 6]}, ["a"], None),
        ({"a": [1, 2, 3]}, ["a"], None),
        ({"a": [1, 2, 3], "b": [4, 5, 6]}, ["c"], ValueError),
        ({"a": [1, 2, 3], "b": [4, 5, 6]}, [], ValueError),
    ],
)
def test_drop_features_except(input_dict, retain_fields, expected_error):
    """Check that fields are correctly retained."""
    dataset = DatasetDict(train=Dataset.from_dict(input_dict))

    if expected_error:
        with pytest.raises(ValueError):
            drop_features_except(dataset, retain_fields=retain_fields)

    else:
        dataset2 = drop_features_except(dataset, retain_fields=retain_fields)
        assert dataset2["train"].column_names == retain_fields
