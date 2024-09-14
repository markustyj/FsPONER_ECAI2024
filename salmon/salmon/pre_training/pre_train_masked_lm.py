"""Pre-training a MaskedLM."""
import itertools
import logging
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

from datasets import load_dataset
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from salmon.utils.mlutils import compute_lm_perplexity
from salmon.utils.preprocessing import drop_features_except

# pylint: disable=too-many-arguments,too-many-locals

logger = logging.getLogger(__name__)


def transformer_masked_lm_pretrain(
    train_file,
    test_file,
    base_model: str,
    output_model: str,
    logging_dir: str,
    tokenizer_model: str,
    evaluation_strategy: Union[str, IntervalStrategy] = "epoch",
    save_strategy: Union[str, IntervalStrategy] = "epoch",
    mlm_probability: float = 0.15,
    tokenizer_batched: bool = True,
    tokenizer_num_proc: int = 4,
    tokenizer_use_fast: bool = True,
    grouping_batched: bool = True,
    grouping_block_size: int = 128,
    grouping_batch_size: int = 1000,
    grouping_num_proc: int = 4,
    num_train_epochs: int = 4,
    save_total_limit: int = 2,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
) -> Tuple[float, float]:
    """
    Pretrain a base transformer masked language model.

    Parameters
    ----------
    train_file: training data file in JSON Lines format.
    test_file: test data file in JSON Lines format.
    base_model: base Transformer model that would be fine-tuned. It can be a
        path to a model or name of a out-of-the-box
    output_model: path where the output model will be saved.
    logging_dir: path for the logging directory.
    tokenizer_model: tokenizer model. Must be a HuggingFace Tokenizer model.
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
    mlm_probability: MLM probability.
    tokenizer_batched: flag for activating/deactivating batched tokenization.
    tokenizer_num_proc: number of processes to be used for tokenization.
    tokenizer_use_fast: flag for activating/deactivating fast tokenization.
    grouping_batched: flag for activating/deactivating batched for grouping
        examples.
    grouping_block_size: block size/chunk size for grouping.
    grouping_batch_size: batch size for grouping.
    grouping_num_proc: number of processes to be used for grouping.
    num_train_epochs:
            Total number of training epochs to perform (if not an integer, will
            perform the decimal part percents of the last epoch before stopping
            training).
    save_total_limit:
            If a value is passed, will limit the total amount of checkpoints.
            Deletes the older checkpoints in `output_dir`.
    train_batch_size:
        Per device training batch size.
    eval_batch_size:
        Per device evaluation batch size.

    Returns
    -------
    A tuple containing (perplexity before domain adaptation, perplexity after
        domain adaptation)
    """
    datasets = load_dataset(
        "json", data_files={"train": train_file, "test": test_file}
    )

    # only the text field is required for domain adaptation
    datasets = drop_features_except(datasets, ["text"])

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model, use_fast=tokenizer_use_fast
    )

    tokenized_datasets = datasets.map(
        lambda x: tokenizer(x["text"]),
        batched=tokenizer_batched,
        num_proc=tokenizer_num_proc,
        remove_columns=["text"],
    )

    # We concatenate all our texts together then split the result in small
    # chunks of a certain `block_size`. To do this, we use the `map` method
    # again, with the option `batched=True`. This option lets us change
    # the number of examples in the datasets by returning a different number
    # of examples than we got. This way, we can create our new samples from
    # a batch of examples.

    # First, we grab the maximum length our model was pretrained with.
    # This might be too big to fit in the GPU RAM,
    # so here we take a bit less at just 128 (i.e. default
    # grouping_block_size).

    lm_datasets = tokenized_datasets.map(
        lambda x: group_block_texts(x, grouping_block_size=grouping_block_size),
        batched=grouping_batched,
        batch_size=grouping_batch_size,
        num_proc=grouping_num_proc,
    )
    model_name = output_model.split("/")[-1]

    model = AutoModelForMaskedLM.from_pretrained(base_model)
    training_args = TrainingArguments(
        model_name,
        evaluation_strategy=evaluation_strategy,
        num_train_epochs=num_train_epochs,
        logging_dir=logging_dir,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
    )

    # We use a special `data_collator` which is a function
    # that is responsible for taking the samples and batching them in tensors.
    # Here we want to do random-masking. We could do it as a pre-processing
    # step (like the tokenization) but then the tokens would always be masked
    # the same way at each epoch. By doing this step inside the `data_collator`,
    # we ensure this random masking is done in a new way each time
    # we go over the data.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=mlm_probability
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )

    eval_results_before = trainer.evaluate()
    perplexity_before = compute_lm_perplexity(eval_results_before)

    trainer.train()

    eval_results_after = trainer.evaluate()
    perplexity_after = compute_lm_perplexity(eval_results_after)
    msg = f"Perplexity before domain adaptation: {perplexity_before:.2f}"
    logger.info(msg)
    msg = f"Perplexity after domain adaptation: {perplexity_after:.2f}"
    logger.info(msg)
    # trainer.save_metrics(split="eval", metrics=eval_results)
    trainer.save_model(output_model)
    return perplexity_before, perplexity_after


def group_block_texts(
    examples: Dict, grouping_block_size: int
) -> Union[Dict, Any]:
    """
    Preprocessing function that groups texts.

    This function concatenates the tokenised documents and splits them into
    new groups of length `grouping_block_size`. The input is a dictionary
    containing fields `input_ids` and `attention_mask`, each of which is a
    list of lists of integer indices. The total number of elements should be
    equal.

    The result is a dict with fields `input_ids`, `attention_mask` and
    `labels`. Each is a list of list, where each sublist is of length
    grouping_block_size.

    Parameters
    ----------
    examples: ungrouped examples from the dataset.
    grouping_block_size: block size/chunk size for grouping.

    Returns
    -------
    Dict:
        The grouped examples
    """
    # Flatten each list
    concatenated_examples = {
        k: list(itertools.chain.from_iterable(examples[k]))
        for k in examples.keys()
    }

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model
    # supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // grouping_block_size) * grouping_block_size
    # Split by chunks of max_len.
    result = {
        k: [
            t[i : i + grouping_block_size]
            for i in range(0, total_length, grouping_block_size)
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
