"""Module for the implementation of the storage backend operations."""

import logging
from typing import Any

from salmon.storage.storage_config import StorageConfig
from salmon.storage.storage_factory import BaseModelStorage
from salmon.storage.storage_factory import download_from_cache
from salmon.storage.storage_factory import download_from_local_repository
from salmon.storage.storage_factory import download_from_s3_repository
from salmon.storage.storage_factory import is_in_cache
from salmon.storage.storage_factory import load_model_card
from salmon.storage.storage_factory import load_model_from_cache
from salmon.storage.storage_factory import load_model_from_local_repository
from salmon.storage.storage_factory import load_model_from_s3_repository
from salmon.storage.storage_factory import load_tokenizer_from_cache
from salmon.storage.storage_factory import load_tokenizer_from_local_repository
from salmon.storage.storage_factory import load_tokenizer_from_s3_repository
from salmon.storage.storage_factory import upload_to_local_repository
from salmon.storage.storage_factory import upload_to_s3_repository

logger = logging.getLogger(__name__)


class LocalModelStorage(BaseModelStorage):
    """Factory class for local file system backend."""

    @classmethod
    def from_pretrained(cls, config: StorageConfig) -> Any:
        """
        Load a HuggingFace model from the repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations. The
        StorageConfig.local_model_repository_directory and
        StorageConfig.model_name must be set for this method to work.

        Returns
        -------
        An instance of HuggingFace model. The type of the model is automatically
        allocated depending on what type of model is loaded.
        """
        if is_in_cache(config):
            model = load_model_from_cache(config)
        else:
            model = load_model_from_local_repository(config)
        return model

    @classmethod
    def from_pretrained_tokenizer(cls, config: StorageConfig) -> Any:
        """
        Load a HuggingFace tokenizer from the repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations. The
        StorageConfig.local_model_repository_directory and
        StorageConfig.model_name must be set for this method to work.

        Returns
        -------
        An instance of HuggingFace model. The type of the tokenizer is
        automatically allocated depending on what type of model is loaded.
        """
        if is_in_cache(config):
            tokenizer = load_tokenizer_from_cache(config)
        else:
            tokenizer = load_tokenizer_from_local_repository(config)
        return tokenizer

    @classmethod
    def upload(cls, config: StorageConfig) -> None:
        """
        Upload a HuggingFace model to the local repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations. The
        StorageConfig.local_model_repository_directory,
        StorageConfig.model_name, StorageConfig.local_model_path, and
        StorageConfig.mode_card_path must be set for this method to work.
        """
        model_card = load_model_card(config)
        upload_to_local_repository(config, model_card)
        logger.info("Model upload finished!")

    @classmethod
    def download(cls, config: StorageConfig) -> None:
        """
        Download a model from the local repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations. The
        StorageConfig.local_model_repository_directory,
        StorageConfig.model_name, and
        StorageConfig.local_model_download_directory
        must be set for this method to work.
        """
        if is_in_cache(config):
            download_from_cache(config)
        else:
            download_from_local_repository(config)


class S3ModelStorage(BaseModelStorage):
    """Factory class for S3 storage backend, not implemented yet."""

    @classmethod
    def from_pretrained(cls, config: StorageConfig) -> Any:
        """
        Load a HuggingFace model from the S3 repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.

        Returns
        -------
        An instance of HuggingFace model. The type of the model is automatically
        allocated depending on what type of model is loaded.
        """
        if is_in_cache(config):
            model = load_model_from_cache(config)
        else:
            model = load_model_from_s3_repository(config)
        return model

    @classmethod
    def from_pretrained_tokenizer(cls, config: StorageConfig) -> Any:
        """
        Load a HuggingFace tokenizer from the S3 repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.

        Returns
        -------
        An instance of HuggingFace model. The type of the tokenizer is
        automatically allocated depending on what type of model is loaded.
        """
        if is_in_cache(config):
            tokenizer = load_tokenizer_from_cache(config)
        else:
            tokenizer = load_tokenizer_from_s3_repository(config)
        return tokenizer

    @classmethod
    def upload(cls, config: StorageConfig) -> None:
        """
        Upload a HuggingFace model to the S3 repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.
        """
        model_card = load_model_card(config)
        upload_to_s3_repository(config, model_card)
        logger.info("Model upload to S3 repository finished!")

    @classmethod
    def download(cls, config: StorageConfig) -> None:
        """
        Download a model from the S3 repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.
        """
        if is_in_cache(config):
            download_from_cache(config)
        else:
            download_from_s3_repository(config)


class AIStoreModelStorage(BaseModelStorage):
    """Factory class for AI-Store storage backend, not implemented yet."""

    @classmethod
    def from_pretrained(cls, config: StorageConfig) -> Any:
        """
        Load a HuggingFace model from the AI-Store repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.

        Returns
        -------
        An instance of HuggingFace model. The type of the model is automatically
        allocated depending on what type of model is loaded.
        """
        raise NotImplementedError("AIStore storage layer is not implemented.")

    @classmethod
    def from_pretrained_tokenizer(cls, config: StorageConfig) -> Any:
        """
        Load a HuggingFace tokenizer from the AI-Store repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.

        Returns
        -------
        An instance of HuggingFace model. The type of the tokenizer is
        automatically allocated depending on what type of model is loaded.
        """
        raise NotImplementedError("AIStore storage layer is not implemented.")

    @classmethod
    def upload(cls, config: StorageConfig) -> None:
        """
        Upload a HuggingFace model to the AI-Store repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.
        """
        raise NotImplementedError("AIStore storage layer is not implemented.")

    @classmethod
    def download(cls, config: StorageConfig) -> None:
        """
        Download a model from the AI-Store repository.

        Parameters
        ----------
        config: instance of `StorageConfig` to specify configurations.
        """
        raise NotImplementedError("AIStore storage layer is not implemented.")
