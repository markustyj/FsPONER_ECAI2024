"""Module for the storage configuration related code."""

import os
import pathlib
from dataclasses import dataclass
from dataclasses import field


@dataclass
class StorageConfig:
    """
    Configuration arguments for the storage operations.

    Parameters
    ----------
        local_model_repository_directory:
            The directory of the local file system that should be used as the
            repository directory where the model will be stored.
        local_model_download_directory:
            The directory where the user will download the model.
        model_name:
            The name of the model.
        local_model_path:
            The path of the model that would be uploaded to the repository.
        model_card_path:
            The path of the model card file which specify
            the metadata of the model to be uploaded.
        s3_model_repository_bucket:
             The s3 bucket name that should be used as the
            repository where the model will be stored.
    """

    local_model_repository_directory: str = field(
        default="",
        metadata={
            "help": "The local directory which acts as the salmon repository."
        },
    )
    local_model_download_directory: str = field(
        default="",
        metadata={
            "help": "The local directory where the "
            "HuggingFace model should be downloaded."
        },
    )
    model_name: str = field(
        default="", metadata={"help": "The name of he model."}
    )

    local_model_path: str = field(
        default="", metadata={"help": "Path of the local model."}
    )
    model_card_path: str = field(default="", metadata="Model card file path.")

    s3_model_repository_bucket: str = field(
        default="",
        metadata={
            "help": "The s3 bucket name which acts as the salmon repository."
        },
    )


def get_local_home_directory() -> str:
    """
    Get the local home directory.

    This is used to cache and temporarily
    extract archived files.

    Returns
    -------
    Path of the home directory.
    """
    directory = str(pathlib.Path.home())
    return directory


def get_local_cache_directory() -> str:
    """
    Get the path of the cache directory.

    Returns
    -------
    Returns the path of the cache directory.
    """
    directory = get_local_home_directory() + "/.salmon/cache"
    if not os.path.exists(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_local_model_cache_directory() -> str:
    """
    Get path of the local model cache.

    Models are extracted and cached in the '~/.cache/salmon/models' directory.

    Returns
    -------
    Returns path of the local model cache.
    """
    directory = get_local_cache_directory() + "/models"
    if not os.path.exists(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return directory
