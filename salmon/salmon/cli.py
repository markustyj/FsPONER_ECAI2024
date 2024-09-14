"""The command line interface for salmon."""
import importlib
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from datasets import load_dataset

from salmon import __version__ as salmon_version
from salmon.downstream.ner_finetune import transformer_finetune_ner_downstream
from salmon.downstream.text_classification_finetune import (
    transformer_finetune_textcat_downstream,
)
from salmon.models import compute_embedding
from salmon.pre_training.pre_train_masked_lm import (
    transformer_masked_lm_pretrain,
)
from salmon.storage.model_storage import AIStoreModelStorage
from salmon.storage.model_storage import S3ModelStorage
from salmon.storage.storage_config import StorageConfig
from salmon.storage.storage_factory import BaseModelStorage

# pylint: disable=too-many-arguments


app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)


class ExternalModelStoreIdentifier(str, Enum):
    """The name of an officially supported storage backend."""

    S3 = "s3"
    AISTORE = "aistore"
    AZUREML = "azureml"

    def check_import(self):
        """
        Check that the respective dependency can be imported.

        The name of module must follow the convention "salmon-<name>", where
        <name> is the name of the model storage backend, ie s3, aistore or
        azureml. The corresponding libraries are hosted in the salmon Gitlab
        group: https://code.siemens.com/salmon/

        Returns
        -------
        bool:
            True if the module can be imported. Returns false and issues
            an error message if the module could not be imported.

        Raises
        ------
        ImportError
        """
        library_name = f"salmon-{self.value}"
        logger.debug("Checking import of %s", library_name)

        try:
            importlib.import_module(library_name)
        except ImportError as _ie:
            print(
                f"Failed to import {library_name}, so the storage "
                f"backend {self.value} is not available."
            )
            print(
                f"Please specify a different storage backend or install"
                f" salmon with the necessary dependency:\n   "
                f"pip install salmon[{self.value}]\n"
            )
            return False

        return True

    def get_storage_connection(self) -> BaseModelStorage:
        """
        Create the appropriate model storage object.

        This function will fail with an import error if the matching
        library has not yet been imported, e.g. salmon-aistore. Ensure
        that `self.check_import()` has been called.

        Returns
        -------
        BaseModelStorage
            A connection to the appropriate model registry

        Raises
        ------
        NotImplementedError
            If the storage backend is not supported yet
        """
        # TODO: BaseModelStorage should be moved into salmon-commons,
        #       since it is required by any plugin which implements the
        #       model storage operations.
        storage_connection: Optional[BaseModelStorage] = None

        print(
            f"Creating connection to model registry for registry "
            f"type {self.value}"
        )

        if self == ExternalModelStoreIdentifier.S3:
            # TODO: this implementation could be moved into salmon-s3 later
            storage_connection = S3ModelStorage()
        elif self == ExternalModelStoreIdentifier.AISTORE:
            # TODO: this implementation needs to move to salmon-aistore
            storage_connection = AIStoreModelStorage()
        elif self == ExternalModelStoreIdentifier.AZUREML:
            # TODO: We leave this error here as a placeholder until
            #       the AzureML registry is implemented
            raise NotImplementedError(
                "The storage backend for AzureML is not"
                " yet supported by the CLI"
            )

        return storage_connection


class EmbeddingAggregationMethod(str, Enum):
    """The method for aggregating the last hidden state of the model."""

    CLS = "cls"


class EvaluationSaveStrategy(str, Enum):
    """The range of possible values for evaluation_strategy."""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EmbeddingOutputFormat(str, Enum):
    """The range of possible output formats for document embeddings."""

    JSON = "json"
    NUMPY = "numpy"

    def write(self, data, output_path: Path):
        """Write the given data in the appropriate format to the given file."""
        print(f"Writing data to {output_path} in format {self.value}")
        if self == EmbeddingOutputFormat.NUMPY:
            with open(output_path, "wb") as f:
                np.save(f, data)
        elif self == EmbeddingOutputFormat.JSON:
            raise NotImplementedError()


@app.command(help="Print the version of the Salmon library", name="version")
def cli_print_version():
    """Print the library version."""
    print(salmon_version)


@app.command(help="Fine tune a model for downstream NLP task", name="finetune")
def cli_finetune(
    textcat: bool = typer.Option(
        False,
        help="Text classification fine-tuning",
    ),
    ner: bool = typer.Option(
        False,
        help="Named Entity Recognition (NER) fine-tuning",
    ),
    train_file: Path = typer.Option(
        ...,
        help="Training data file in json lines format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    val_file: Path = typer.Option(
        ...,
        help="Validation data file in json lines format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    test_file: Path = typer.Option(
        ...,
        help="Test data file in json lines format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    training_data_fraction: float = typer.Option(
        default=1.0,
        min=0.0,
        max=1.0,
        help="Fraction of the training data that should be used",
    ),
    num_train_epochs: int = typer.Option(
        default=4, min=1, help="Number of training epochs"
    ),
    base_model: str = typer.Option(
        ...,
        help="Path of the base transformer model or the name in huggingface"
        " model hub",
    ),
    output_model: str = typer.Option(
        ..., help="Path to which the trained model should be saved"
    ),
    logging_dir: Path = typer.Option(
        "./logs",
        help="Path to the log directory",
    ),
    tokenizer_model: str = typer.Option(
        ...,
        help="Name of the tokenizer model",
    ),
    evaluation_strategy: EvaluationSaveStrategy = typer.Option(
        "epoch",
        help="The evaluation strategy to adopt during training",
    ),
    save_strategy: EvaluationSaveStrategy = typer.Option(
        "epoch",
        help="The checkpoint save strategy to adopt during training",
    ),
    save_total_limit: int = typer.Option(
        default=2,
        help="Limit the total amount of checkpoints",
    ),
    train_batch_size: int = typer.Option(
        default=4,
        help="Per device training batch size",
    ),
    eval_batch_size: int = typer.Option(
        default=4,
        help="Per device evaluation batch size",
    ),
    evaluation_metrics_output: Path = typer.Option(
        None,
        help="File name where the evaluation metrics would be written",
    ),
    load_best_model_at_end: bool = typer.Option(
        False,
        help="Whether or not to save the best model found during training",
    ),
    metric_for_best_model: str = typer.Option(
        "eval_loss",
        help="Name of the evaluation metric for load_best_model_at_end",
    ),
    greater_is_better: bool = typer.Option(
        False,
        help="Use with load_best_model_at_end and "
        "metric_for_best_model to specify if better models "
        "should have a greater metric or not",
    ),
    tokenizer_add_prefix_space: bool = typer.Option(
        False,
        help="Whether or not to add an initial space to the input. "
        "Required to be set to True to run RoBERTa tokenizers.",
    ),
) -> None:
    """Fine tune a model for downstream NLP task."""
    logger.info("Training data file %s", train_file)
    logger.info("Test data file %s", test_file)

    if textcat is False and ner is False:
        print("--textcat or --ner flag needs to be set.")
        sys.exit(1)
    elif textcat is True and ner is True:
        print("Only one of --textcat or --ner flag needs to be set.")
        sys.exit(1)

    evaluation_metrics_output = (
        None
        if evaluation_metrics_output is None
        else str(evaluation_metrics_output)
    )

    if textcat:
        transformer_finetune_textcat_downstream(
            train_file=str(train_file),
            val_file=str(val_file),
            test_file=str(test_file),
            base_model=base_model,
            output_model=output_model,
            training_data_fraction=training_data_fraction,
            logging_dir=str(logging_dir),
            tokenizer_model=tokenizer_model,
            evaluation_strategy=evaluation_strategy.value,
            save_strategy=save_strategy.value,
            num_train_epochs=num_train_epochs,
            save_total_limit=save_total_limit,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            evaluation_metrics_output=evaluation_metrics_output,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
        )
    elif ner:
        transformer_finetune_ner_downstream(
            train_file=str(train_file),
            val_file=str(val_file),
            test_file=str(test_file),
            base_model=base_model,
            output_model=output_model,
            training_data_fraction=training_data_fraction,
            logging_dir=str(logging_dir),
            tokenizer_model=tokenizer_model,
            evaluation_strategy=evaluation_strategy.value,
            save_strategy=save_strategy.value,
            num_train_epochs=num_train_epochs,
            save_total_limit=save_total_limit,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            evaluation_metrics_output=evaluation_metrics_output,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            tokenizer_add_prefix_space=tokenizer_add_prefix_space,
        )


@app.command(
    help="Perform domain adaptation for a given model and corpus", name="adapt"
)
def cli_domain_adaptation(
    train_file: Path = typer.Option(
        ...,
        help="Training data file in json lines format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    test_file: Path = typer.Option(
        ...,
        help="Test data file in json lines format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    mlm_probability: float = typer.Option(
        default=0.15,
        min=0.0,
        max=1.0,
        help="Masking probability for the MLM pre-training task",
    ),
    num_train_epochs: int = typer.Option(
        default=4, min=1, help="Number of training epochs"
    ),
    base_model: str = typer.Option(
        ...,
        help="Path of the base transformer model or the name in huggingface"
        " model hub",
    ),
    output_model: str = typer.Option(
        ..., help="Path to which the trained model should be saved"
    ),
    logging_dir: Path = typer.Option(
        "./logs",
        help="Path to the log directory",
    ),
    tokenizer_model: str = typer.Option(
        ...,
        help="Name of the tokenizer model",
    ),
    evaluation_strategy: EvaluationSaveStrategy = typer.Option(
        "epoch",
        help="The evaluation strategy to adopt during training",
    ),
    save_strategy: EvaluationSaveStrategy = typer.Option(
        "epoch",
        help="The checkpoint save strategy to adopt during training",
    ),
    save_total_limit: int = typer.Option(
        default=2,
        help="Limit the total amount of checkpoints",
    ),
    grouping_block_size: int = typer.Option(
        default=128,
        help="Block size/chunk size for grouping training samples",
    ),
    grouping_batch_size: int = typer.Option(
        default=1000,
        help="Batch size for grouping training samples",
    ),
    grouping_num_proc: int = typer.Option(
        default=4,
        help="Number of processes for grouping training samples",
    ),
    tokenizer_num_proc: int = typer.Option(
        default=4,
        help="Number of processes for tokenizing training samples",
    ),
    train_batch_size: int = typer.Option(
        default=4,
        help="Per device training batch size",
    ),
    eval_batch_size: int = typer.Option(
        default=4,
        help="Per device evaluation batch size",
    ),
) -> None:
    """Perform domain adaptation for a given model and corpus."""
    _, _ = transformer_masked_lm_pretrain(
        train_file=str(train_file),
        test_file=str(test_file),
        base_model=base_model,
        output_model=output_model,
        logging_dir=str(logging_dir),
        tokenizer_model=tokenizer_model,
        evaluation_strategy=evaluation_strategy.value,
        save_strategy=save_strategy.value,
        num_train_epochs=num_train_epochs,
        save_total_limit=save_total_limit,
        mlm_probability=mlm_probability,
        grouping_block_size=grouping_block_size,
        grouping_batch_size=grouping_batch_size,
        grouping_num_proc=grouping_num_proc,
        tokenizer_num_proc=tokenizer_num_proc,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )


@app.command(
    help="Compute document embeddings with a given model and corpus",
    name="embed",
)
def cli_compute_embedding(
    docs_file: Path = typer.Option(
        ...,
        help="Documents data file in json lines format",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    model: str = typer.Option(
        ...,
        help="Path of the transformer model or the name in huggingface"
        " model hub",
    ),
    tokenizer_model: str = typer.Option(
        ...,
        help="Name of the tokenizer model",
    ),
    max_length: int = typer.Option(
        default=512,
        min=1,
        help="Length in tokens to which the documents will be padded or "
        "truncated",
    ),
    aggregation: EmbeddingAggregationMethod = typer.Option(
        "cls",
        help="The method for aggregating the last hidden state of the "
        "model into a single document embedding",
    ),
    output_path: Path = typer.Option(
        ...,
        help="File path to write embeddings to",
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
    output_format: EmbeddingOutputFormat = typer.Option(
        "numpy",
        help="The output format for the embeddings. Numpy files will "
        "be pickled.",
    ),
) -> None:
    """Compute document embeddings with a given model and corpus."""
    documents_dataset = load_dataset("json", data_files=str(docs_file))
    documents = documents_dataset["train"]["text"]
    logging.info("Loaded %s documents from %s", len(documents), docs_file)

    embeddings = compute_embedding(
        model,
        tokenizer_model,
        documents,
        aggregation=aggregation,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    logging.info(
        "Computed embeddings matrix of shape %s", str(embeddings.shape)
    )
    output_format.write(embeddings, output_path)


@app.command(help="Pull a model from the Salmon registry", name="pull")
def cli_pull_model(
    store: ExternalModelStoreIdentifier = typer.Option(
        "aistore",
        help="The storage backend from which the model should be pulled",
    ),
    model_name: str = typer.Option(..., help="Name of the model"),
) -> None:
    """Pull a model from the Salmon registry."""
    if not store.check_import():
        print("Exiting...")
        sys.exit(-2)

    storage_connection = store.get_storage_connection()

    # TODO: from here on the functions from the interface of
    #       BaseModelStorage should be used to pull the model
    storage_config = StorageConfig(model_name=model_name)
    storage_connection.download(storage_config)


@app.command(help="Publish a model to the Salmon registry", name="publish")
def cli_publish_model_to_registry(
    store: ExternalModelStoreIdentifier = typer.Option(
        "aistore",
        help="The storage backend to which the model should be published",
    ),
    model_name: str = typer.Option(..., help="Name of the model"),
) -> None:
    """Publish a model to the Salmon registry."""
    if not store.check_import():
        print("Exiting...")
        sys.exit(-2)

    _storage_connection = store.get_storage_connection()
    print(f"Publishing model: {model_name}")

    raise NotImplementedError("This function is not yet supported by the CLI")


@app.command(
    help="Get a list of the models in the Salmon registry", name="list"
)
def cli_list_models_in_registry(
    store: ExternalModelStoreIdentifier = typer.Option(
        "aistore",
        help="The storage backend from which the list of model should be read",
    )
) -> None:
    """Get a list of the models in the Salmon registry."""
    if not store.check_import():
        print("Exiting...")
        sys.exit(-2)

    _storage_connection = store.get_storage_connection()

    raise NotImplementedError("This function is not yet supported by the CLI")


def main():
    """Run the cli app."""
    logging.basicConfig(level=logging.INFO)
    app()


if __name__ == "__main__":
    main()
