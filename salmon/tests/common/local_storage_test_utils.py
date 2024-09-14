"""Utilities for local storage tests."""
import os
import shutil
import tarfile

from salmon.storage.storage_config import StorageConfig
from salmon.storage.storage_config import get_local_model_cache_directory


def init_config(
    model_name,
    model_card_path,
    local_model_repository_directory,
    local_working_directory,
) -> StorageConfig:
    """
    Test utility function to initiate an StorageConfig instance.

    Parameters
    ----------
    model_name: model name.
    model_card_path: model card path.
    local_model_repository_directory: local model repository path.
    local_working_directory: local working directory path.

    Returns
    -------
    An instance of StorageConfig.
    """
    local_model_path = local_working_directory + f"/{model_name}"

    # extract the test model into the working directory
    test_model_file = f"tests/models/{model_name}.tar.gz"
    if not os.path.exists(local_model_path):
        with tarfile.open(test_model_file, "r:gz") as tar:
            tar.extractall(path=local_working_directory)
    config = StorageConfig(
        local_model_repository_directory=local_model_repository_directory,
        local_model_path=local_model_path,
        model_name=model_name,
        model_card_path=model_card_path,
    )
    return config


def put_model_into_cache(model_name: str) -> None:
    """
    Test utility function to manually put a model into the cache.

    Parameters
    ----------
    model_name: name of the model.

    """
    local_model_file = f"tests/models/{model_name}.tar.gz"
    shutil.copy(local_model_file, get_local_model_cache_directory())


def put_model_into_local_repository(
    model_name: str, local_model_repository_directory: str
) -> None:
    """
    Test utility function to manually put a model into the local repository.

    Parameters
    ----------
    model_name: name of the model.
    local_model_repository_directory: the directory that should be used
    as the local repository.
    """
    local_model_file = f"tests/models/{model_name}.tar.gz"
    shutil.copy(local_model_file, local_model_repository_directory)


def extract_model_locally(
    local_model_path, local_working_directory, model_name
):
    """Extract a model locally for some test cases."""
    # extract the test model into the working directory
    test_model_file = f"tests/models/{model_name}.tar.gz"
    if not os.path.exists(local_model_path):
        with tarfile.open(test_model_file, "r:gz") as tar:
            tar.extractall(path=local_working_directory)
