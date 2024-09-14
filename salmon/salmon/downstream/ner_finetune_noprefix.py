"""Fine-tuning a pretrained language model for NER ."""
import json
import logging
import pandas as pd
from functools import cmp_to_key
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

from datasets import Dataset
from datasets import load_dataset
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from salmon.utils.mlutils import compute_ner_metrics

# pylint: disable=too-many-arguments, too-many-locals

logger = logging.getLogger(__name__)


def transformer_finetune_ner_downstream(
    #total_file: str,
    train_file: str,
    val_file: str,
    test_file: str,
    base_model: str,
    output_model: str,
    tokenizer_model: str,
    training_data_fraction: float,
    logging_dir: str,
    evaluation_strategy: Union[str, IntervalStrategy] = "epoch",
    save_strategy: Union[str, IntervalStrategy] = "epoch",
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_train_epochs: int = 4,
    save_total_limit: int = 2,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "loss",
    greater_is_better: bool = False,
    evaluate_before_training: bool = False,
    evaluation_metrics_output: str = None,
    tokenizer_add_prefix_space: bool = False,
    device_map: str = "cuda:3"
) -> Tuple[Dict, Dict]:
    """
    Fine tune a language model for the downstream task of NER.

    Parameters
    ----------
    train_file: Full training data file in JSON Lines format.
    val_file: Full validation data file in JSON Lines format.
    test_file: Full test data file in JSON Lines format.
    base_model: Base Transformer model that would be fine-tuned. It can be a
      path to a model or name of a out-of-the-box
    HuggingFace model.
    output_model: Path where the output model will be saved.
    tokenizer_model: Tokenizer model. Must be a HuggingFace Tokenizer model.
    training_data_fraction: What fraction of the full training data that
      should be used for fine-tuning.
    logging_dir: Path for the logging directory.
    evaluation_strategy:
            The evaluation strategy to adopt during training. Possible values
            are:

                * `"no"`: No evaluation is done during training.
                * `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                * `"epoch"`: Evaluation is done at the end of each epoch.
    save_strategy:
            The checkpoint save strategy to adopt during training. Possible
            values are:

                * `"no"`: No save is done during training.
                * `"epoch"`: Save is done at the end of each epoch.
                * `"steps"`: Save is done every `save_steps`.
    train_batch_size:
        Per device training batch size.
    eval_batch_size:
        Per device evaluation batch size.
    num_train_epochs:
            Total number of training epochs to perform (if not an integer, will
            perform the decimal part percents of the last epoch before stopping
            training).
    save_total_limit:
            If a value is passed, will limit the total amount of checkpoints.
            Deletes the older checkpoints in `output_dir`.
    load_best_model_at_end:
         Whether or not to load the best model found during training at
         the end of training. Defaults to False.
    metric_for_best_model:
        Use in conjunction with load_best_model_at_end to specify the metric
        to use to compare two different models. Must be the name of a metric
        returned by the evaluation with or without the prefix "eval_".
        Will default to "loss" if unspecified
        and load_best_model_at_end=True (to use the evaluation loss).

        If you set this value, greater_is_better will default to True.
        Don’t forget to set it to False if your metric is better when lower.
    greater_is_better:
        Use in conjunction with load_best_model_at_end and
        metric_for_best_model to specify if better models
        should have a greater metric or not. Will default to False.
    evaluate_before_training:
        Evaluating the untrained model.
    evaluation_metrics_output:
        File name where the evaluation metrics would be written.
    tokenizer_add_prefix_space:
        Required to be set to True to run RoBERTa tokenizers.
    """
    # Load the JSONL training, validation, and test data into a dataset
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": val_file,
            "test": test_file,
        },
    )
    
    # Compute labels and their categorical encodings from the full
    # training data. The label encoding is computed from the full data
    # because we want to avoid unseen labels in the test and validation
    # data when a fraction of training data is selected or when some labels
    # are note available in the training data. Data pre-processing should
    # already take care of this, but this is just a safety measure.
    label_list, label_encoding_dict = create_label_list(raw_datasets["train"],raw_datasets["validation"],raw_datasets["test"])
    #label_list, label_encoding_dict = create_label_list(raw_datasets["train"])

    # Load the tokenizer. note the add_prefix_space param. This is required
    # for RoBERTa models.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model,
        use_fast=True,
        add_prefix_space=tokenizer_add_prefix_space,
    #    device_map = device_map
    )

    # Re-tokenize the input tokens. This will align the input tokens to
    # BERT tokens (BPE/WordPiece -- whatever is given as the tokenizer).
    def tokenize_function(examples):
        return tokenize_and_align_labels(
            examples, tokenizer, label_encoding_dict
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, num_proc=4
    )

    # Separate out the test, validation, and training sets into different
    # variables for convenience. Select the fraction of training data that
    # that should be used.
    full_test_dataset = tokenized_datasets["test"]
    full_validation_dataset = tokenized_datasets["validation"]

    train_data_size = int(
        tokenized_datasets.shape["train"][0] * training_data_fraction
    )

    full_train_dataset = (
        tokenized_datasets["train"]
        .shuffle(seed=42)
        .select(range(train_data_size))
    )

    logger.info("Training data size: %s", train_data_size)
    logger.info(
        "Training data original size: %s", tokenized_datasets.shape["train"][0]
    )
    # Fine-tuning
    # -----------
    num_labels = len(label_list)
    model_name = output_model.split("/")[-1]

    # Initiate the token classification auto model with the computed
    # number of labels.
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, 
        num_labels=num_labels,
    #    device_map = device_map,
    )

    # Initiate training arguments.
    training_args = TrainingArguments(
        model_name,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        logging_dir=logging_dir,
        evaluation_strategy=evaluation_strategy,
        num_train_epochs=num_train_epochs,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
    )

    # Data collator pads the data correctly
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Initiate the training with validation data and training data.
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=full_validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=full_train_dataset,
        compute_metrics=lambda x: compute_ner_metrics(x, label_list),
    )

    # Train and store the model
    trainer.train()
    trainer.save_model(output_model)

    eval_before = {}
    if evaluate_before_training:
        logger.info("Evaluating the untrained model...")
        # Evaluate only (without training) if evaluate_before_training is set
        eval_before = evaluate_only(
            data_collator,
            full_test_dataset,
            model,
            tokenizer,
            training_args,
            label_list,
        )

    # Train and store the model
    trainer.train()
    trainer.save_model(output_model)

    # Load the stored model.
    trained_model = AutoModelForTokenClassification.from_pretrained(
        output_model, 
        num_labels=num_labels,
    #    device_map = device_map
    )

    logger.info("Evaluating the fine-tuned model...")
    # Run the final evaluation with the validation data.
    eval_after = evaluate_only(
        data_collator,
        full_test_dataset,
        trained_model,
        tokenizer,
        training_args,
        label_list,
    )

    # Write the results into a file.
    if evaluation_metrics_output is not None:
        with open(evaluation_metrics_output, "w", encoding="utf-8") as fout:
            json.dump(eval_after, fout, indent=4)

    return eval_before, eval_after


def evaluate_only(
    data_collator: DataCollatorForTokenClassification,
    test_dataset: Any,
    model: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
    label_list: list,
) -> Dict:
    """
    Evaluate only using the HuggingFace Training API.

    Parameters
    ----------
    data_collator: data collator.
    test_dataset: test dataset.
    model: HuggingFace model.
    tokenizer: HuggingFace tokenizer.
    training_args: training arguments.
    label_list: list of NER labels.

    Returns
    -------
    Evaluation result dictionary.

    """
    evaluator = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_ner_metrics(x, label_list),
    )
    eval_before = evaluator.evaluate()
    return eval_before


def tokenize_and_align_labels(
    examples: Any, tokenizer: Any, label_encoding_dict: Dict
) -> Any:
    """
    Align tokens to BPE/WordPiece tokens and adjust the labels accordingly.

    Parameters
    ----------
    examples: a sample/batch of samples from a dataset.
    tokenizer: a HuggingFace tokenizer.
    label_encoding_dict: a dictionary that contains the labels as keys and their
    encoded integer id as values.

    Returns
    -------
    Tokenized and adjusted sample(s).
    """
    label_all_tokens = True
    tokenized_inputs = tokenizer(
        list(examples["tokens"]), truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == "0":
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(
                    label_encoding_dict[label[word_idx]]
                    if label_all_tokens
                    else -100
                )
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compare_labels(item1: str, item2: str) -> int:
    """
    Compare function to sort labels so that their indices can be used as IDs.

    Parameters
    ----------
    item1: label name string.
    item2: label name string.

    Returns
    -------
    -1 or 1 or 0 according to the order.
    """
    # pylint: disable=too-many-return-statements
    if len(item1) == 1 and len(item2) == 1:
        if item1 > item2:
            return -1
        if item1 < item2:
            return 1
        return 0

    if len(item1) == 1:
        return -1
    if len(item2) == 1:
        return 1

    if item1 > item2:
        return -1
    if item1 < item2:
        return 1
    return 0
    

def create_label_list(dataset_train: Dataset, dataset_val: Dataset, dataset_test: Dataset) -> Tuple:
    """
    Create a list of label from the dataset and return an encoding Dictionary.

    Parameters
    ----------
    dataset: HuggingFace dataset.

    Returns
    -------
    A tuple of a list of labels, and a dictionary of label encodings.
    """
    labels_set = set()

    dataset_train.map(lambda x: labels_set.update(x["ner_tags"]))
    dataset_val.map(lambda x: labels_set.update(x["ner_tags"]))
    dataset_test.map(lambda x: labels_set.update(x["ner_tags"]))

    label_list = sorted(list(labels_set), key=cmp_to_key(compare_labels))
    label_encoding_dict = {}
    for i, label in enumerate(label_list):
        label_encoding_dict[label] = i
    return label_list, label_encoding_dict
