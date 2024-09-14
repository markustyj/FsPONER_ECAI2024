"""Implementation of the storage factories."""

import logging
import os.path
import shutil
import tarfile
import tempfile
from typing import Any

import boto3
from transformers import AutoModel
from transformers import AutoTokenizer

from salmon.errors.exceptions import CacheError
from salmon.errors.exceptions import NotFoundInRepository
from salmon.errors.exceptions import SalmonConfigError
from salmon.metadata.model_metadata import SalmonModelCard
from salmon.storage.storage_config import StorageConfig
from salmon.storage.storage_config import get_local_model_cache_directory

logger = logging.getLogger(__name__)


class BaseModelStorage:
    """Base storage factory class."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the ModelStorage.

        Parameters
        ----------
        args: args
        kwargs: kwargs
        """
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.download()` or "
            f"`{self.__class__.__name__}.upload()` methods."
        )

    @classmethod
    def from_pretrained(cls, config: StorageConfig) -> Any:
        """
        Load a HuggingFace model from the repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.

        Returns
        -------
        An instance of HuggingFace model. The type of the model is automatically
        allocated depending on what type of model is loaded.
        """
        raise NotImplementedError("Method not implemented.")

    @classmethod
    def from_pretrained_tokenizer(cls, config: StorageConfig) -> Any:
        """
        Load a HuggingFace tokenizer from the repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.

        Returns
        -------
        An instance of HuggingFace model. The type of the tokenizer is
        automatically allocated depending on what type of model is loaded.
        """
        raise NotImplementedError("Method not implemented.")

    @classmethod
    def upload(cls, config: StorageConfig) -> None:
        """
        Upload a HuggingFace model to the repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.
        """
        raise NotImplementedError("Method not implemented.")

    @classmethod
    def download(cls, config: StorageConfig) -> None:
        """
        Download a model from the repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.
        """
        raise NotImplementedError("Method not implemented.")


def is_in_cache(config: StorageConfig) -> bool:
    """
    Check if a model is in the cache.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    True or False
    """
    model_name = config.model_name
    cached_file = get_local_model_cache_directory() + f"/{model_name}.tar.gz"
    if os.path.exists(cached_file):
        return True
    return False


def load_model_from_cache(config: StorageConfig) -> Any:
    """
    Load a model from the salmon cache.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    A HuggingFace model.
    """
    extracted_model_path = get_cached_model_path(config)
    model = AutoModel.from_pretrained(extracted_model_path)
    return model


def load_tokenizer_from_cache(config: StorageConfig) -> Any:
    """
    Load a tokenizer from the salmon cache.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    A HuggingFace tokenizer.
    """
    extracted_model_path = get_cached_model_path(config)
    model = AutoTokenizer.from_pretrained(extracted_model_path)
    return model


def get_cached_model_path(config: StorageConfig) -> str:
    """
    Get the path of a cached model.

    It extracts the model from .tar.gz into the cache.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    Returns the path of the model in the cache.
    """
    model_name = config.model_name
    cached_file = get_local_model_cache_directory() + f"/{model_name}.tar.gz"
    if not is_in_cache(config):
        raise CacheError(
            f"Model {model_name} is not found in "
            f"the cache {get_local_model_cache_directory()}."
        )
    # un-tar the archive
    extract_path = get_local_model_cache_directory()
    extracted_model_path = extract_path + f"/{model_name}"
    if not os.path.exists(extracted_model_path):
        with tarfile.open(cached_file, "r:gz") as tar:
            tar.extractall(path=extract_path)
    return extracted_model_path


def load_model_from_local_repository(config: StorageConfig) -> Any:
    """
    Load model from the local repository.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    A HuggingFace model.

    """
    cache_from_local_repository(config)
    model = load_model_from_cache(config)
    return model


def load_tokenizer_from_local_repository(config: StorageConfig) -> Any:
    """
    Load a tokenizer from the local repository.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    A HuggingFace tokenizer.
    """
    cache_from_local_repository(config)
    tokenizer = load_tokenizer_from_cache(config)
    return tokenizer


def cache_from_local_repository(config: StorageConfig) -> None:
    """
    Put a model from the local repository to the cache.

    Parameters
    ----------
    config: specifies storage configuration.
    """
    model_name = config.model_name
    file_in_repo = (
        config.local_model_repository_directory + f"/{model_name}.tar.gz"
    )
    if not os.path.exists(file_in_repo):
        raise NotFoundInRepository(
            f"Model {model_name} not found in "
            f"the repository {config.local_model_repository_directory}."
        )
    shutil.copy(file_in_repo, get_local_model_cache_directory())


def load_model_card(config: StorageConfig) -> SalmonModelCard:
    """
    Load a model card.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    An instance of SalmonModelCard
    """
    mode_card = SalmonModelCard.load_modelcard(config.model_card_path)
    return mode_card


def upload_to_local_repository(
    config: StorageConfig, model_card: SalmonModelCard
) -> None:
    """
    Upload a model to local repository.

    Parameters
    ----------
    config: specifies storage configuration.
    model_card: instance of ModelCard.
    """
    model_name = config.model_name
    extracted_model_name = config.local_model_path.split("/")[-1]
    if model_name:
        if model_name != extracted_model_name:
            raise SalmonConfigError(
                f"Model name (StorageConfig.model_name)"
                f" different from the model directory "
                f"name {config.local_model_path}"
            )
    else:
        model_name = extracted_model_name
        config.model_name = extracted_model_name

    tar_filename_in_repo = (
        config.local_model_repository_directory + f"/{model_name}.tar.gz"
    )
    model_card_filename_in_repo = (
        config.local_model_repository_directory + f"/{model_name}.md"
    )
    if os.path.exists(tar_filename_in_repo):
        os.remove(tar_filename_in_repo)
        msg = (
            f"{tar_filename_in_repo} file is already in the repository "
            f"and hence the existing file is deleted."
        )
        logger.warning(msg)

    if os.path.exists(model_card_filename_in_repo):
        os.remove(model_card_filename_in_repo)
        msg = (
            f"{model_card_filename_in_repo} file is already in "
            f"the repository and hence the existing file is deleted."
        )

        logger.warning(msg)

    # tar the directory, remember the -C option which is the arcname param here
    with tarfile.open(tar_filename_in_repo, "w|gz") as archive:
        archive.add(config.local_model_path, arcname=extracted_model_name)
    # serialize the model card into the repository directory
    model_card.serialize(model_card_filename_in_repo)


def download_from_cache(config: StorageConfig) -> None:
    """
    Download a model/tokenizer from cache.

    Parameters
    ----------
    config: specifies storage configuration.
    """
    extracted_model_path = get_cached_model_path(config)
    target_dir = (
        config.local_model_download_directory + f"/{config.model_name}/"
    )
    shutil.copytree(extracted_model_path, target_dir)


def download_from_local_repository(config: StorageConfig) -> None:
    """
    Download a model/tokenizer from the local repository.

    Parameters
    ----------
    config: specifies storage configuration.
    """
    cache_from_local_repository(config)
    download_from_cache(config)


def cache_from_s3_repository(config: StorageConfig) -> None:
    """
    Put a model from the s3 repository to the cache.

    Parameters
    ----------
    config: specifies storage configuration.
    """
    model_name = config.model_name
    file_in_repo = f"{model_name}.tar.gz"
    download_file = f"{get_local_model_cache_directory()}/{model_name}.tar.gz"

    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(config.s3_model_repository_bucket)
    objs = list(bucket.objects.filter(Prefix=file_in_repo))

    if len(objs) > 0:
        msg = (
            f"Caching model {model_name} from s3 "
            f"bucket {config.s3_model_repository_bucket} to cache location"
            f" {download_file}"
        )
        logger.info(msg)
        bucket.download_file(file_in_repo, download_file)
    else:
        raise NotFoundInRepository(
            f"Model {model_name} not found in "
            f"the repository {config.s3_model_repository_bucket}."
        )


def load_model_from_s3_repository(config: StorageConfig) -> Any:
    """
    Load model from the s3 repository.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    A HuggingFace model.

    """
    cache_from_s3_repository(config)
    model = load_model_from_cache(config)
    return model


def upload_to_s3_repository(
    config: StorageConfig, model_card: SalmonModelCard
) -> None:
    """
    Upload a model to S3 repository.

    Parameters
    ----------
    config: specifies storage configuration.
    model_card: instance of ModelCard.
    """
    model_name = config.model_name
    extracted_model_name = config.local_model_path.split("/")[-1]
    if model_name:
        if model_name != extracted_model_name:
            raise SalmonConfigError(
                f"Model name (StorageConfig.model_name)"
                f" different from the model directory "
                f"name {config.local_model_path}"
            )
    else:
        model_name = extracted_model_name
        config.model_name = extracted_model_name

    # the context manager will automatically delete the tmp directory after
    # everything is done.
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar_filename_in_repo = f"{model_name}.tar.gz"
        model_card_filename_in_repo = f"{model_name}.md"

        s3_resource = boto3.resource("s3")
        bucket = s3_resource.Bucket(config.s3_model_repository_bucket)
        objs = list(bucket.objects.filter(Prefix=tar_filename_in_repo))

        if len(objs) > 0:

            s3_resource.Object(
                config.s3_model_repository_bucket, tar_filename_in_repo
            ).delete()
            msg = (
                f"{tar_filename_in_repo} file is already in the repository "
                f"and hence the existing file is deleted."
            )
            logger.warning(msg)

        objs = list(bucket.objects.filter(Prefix=model_card_filename_in_repo))

        if len(objs) > 0:
            s3_resource.Object(
                config.s3_model_repository_bucket, model_card_filename_in_repo
            ).delete()
            msg = (
                f"{model_card_filename_in_repo} file is already in "
                f"the repository and hence the existing file is deleted."
            )

            logger.warning(msg)

        # tar the directory,
        # remember the -C option which is the arcname param here
        tmp_tar_file = f"{str(tmpdirname)}/{tar_filename_in_repo}"
        tmp_model_card_file = f"{str(tmpdirname)}/{model_card_filename_in_repo}"
        with tarfile.open(tmp_tar_file, "w|gz") as archive:
            archive.add(config.local_model_path, arcname=extracted_model_name)
        bucket.upload_file(tmp_tar_file, tar_filename_in_repo)
        # serialize the model card into the tmp directory
        model_card.serialize(tmp_model_card_file)
        bucket.upload_file(tmp_model_card_file, model_card_filename_in_repo)


def download_from_s3_repository(config: StorageConfig) -> None:
    """
    Download a model/tokenizer from the s3 repository.

    Parameters
    ----------
    config: specifies storage configuration.
    """
    cache_from_s3_repository(config)
    download_from_cache(config)


def load_tokenizer_from_s3_repository(config: StorageConfig) -> Any:
    """
    Load a tokenizer from the s3 repository.

    Parameters
    ----------
    config: specifies storage configuration.

    Returns
    -------
    A HuggingFace tokenizer.
    """
    cache_from_s3_repository(config)
    tokenizer = load_tokenizer_from_cache(config)
    return tokenizer
