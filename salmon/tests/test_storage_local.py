"""Test cases for storage."""
import filecmp
import logging
import os
import pathlib

from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import RobertaTokenizerFast

from salmon.metadata.model_metadata import SalmonModelCard
from salmon.storage.model_storage import LocalModelStorage
from salmon.storage.storage_config import StorageConfig
from salmon.storage.storage_config import get_local_cache_directory
from salmon.storage.storage_config import get_local_home_directory
from salmon.storage.storage_config import get_local_model_cache_directory
from salmon.storage.storage_factory import cache_from_local_repository
from salmon.storage.storage_factory import download_from_cache
from salmon.storage.storage_factory import download_from_local_repository
from salmon.storage.storage_factory import is_in_cache
from salmon.storage.storage_factory import load_model_card
from salmon.storage.storage_factory import load_model_from_cache
from salmon.storage.storage_factory import load_model_from_local_repository
from salmon.storage.storage_factory import load_tokenizer_from_cache
from salmon.storage.storage_factory import load_tokenizer_from_local_repository
from salmon.storage.storage_factory import upload_to_local_repository
from tests.common.local_storage_test_utils import extract_model_locally
from tests.common.local_storage_test_utils import init_config
from tests.common.local_storage_test_utils import put_model_into_cache
from tests.common.local_storage_test_utils import (
    put_model_into_local_repository,
)

logger = logging.getLogger(__name__)


def test_is_in_cache():
    """Tests if the `is_in_cache()` method works."""
    model_name = "test_model"
    file_name = get_local_model_cache_directory() + "/" + model_name + ".tar.gz"
    # pylint: disable=unspecified-encoding
    with open(file_name, "a+") as f:
        f.write("test model file")
    config = StorageConfig(model_name=model_name)
    contains = is_in_cache(config)
    os.remove(file_name)
    does_not_contain = is_in_cache(config)

    assert contains and not does_not_contain


def test_load_model_from_cache(tiny_base_lm_local):
    """Tests if `load_model_from_cache` loads a model correctly."""
    model_name = tiny_base_lm_local
    put_model_into_cache(model_name)

    config = StorageConfig(
        model_name=model_name,
    )
    model = load_model_from_cache(config)
    assert isinstance(model, RobertaModel)


def test_load_tokenizer_from_cache(tiny_base_lm_local):
    """Tests if `load_tokenizer_from_cache` loads a tokenizer correctly."""
    model_name = tiny_base_lm_local
    put_model_into_cache(model_name)

    config = StorageConfig(
        model_name=model_name,
    )
    tokenizer = load_tokenizer_from_cache(config)
    assert isinstance(tokenizer, RobertaTokenizerFast)


def test_load_model_from_local_repository(
    tiny_base_lm_local, local_test_repository
):
    """Tests if `load_model_from_local_repository` loads a model correctly."""
    model_name = tiny_base_lm_local

    local_model_repository_directory = local_test_repository

    put_model_into_local_repository(
        model_name, local_model_repository_directory
    )

    config = StorageConfig(
        local_model_repository_directory=local_model_repository_directory,
        model_name=model_name,
    )
    model = load_model_from_local_repository(config)
    assert isinstance(model, RobertaModel)


def test_load_tokenizer_from_local_repository(
    tiny_base_lm_local, local_test_repository
):
    """Tests if the tokenizer can be loaded from a local repo correctly."""
    model_name = tiny_base_lm_local

    local_model_repository_directory = local_test_repository
    put_model_into_local_repository(
        model_name, local_model_repository_directory
    )

    config = StorageConfig(
        local_model_repository_directory=local_model_repository_directory,
        model_name=model_name,
    )
    tokenizer = load_tokenizer_from_local_repository(config)
    assert isinstance(tokenizer, RobertaTokenizerFast)


def test_cache_model_from_local(tiny_base_lm_local, local_test_repository):
    """Tests if `cache_from_local_repository` caches a model correctly."""
    model_name = tiny_base_lm_local

    local_model_repository_directory = local_test_repository
    put_model_into_local_repository(
        model_name, local_model_repository_directory
    )

    config = StorageConfig(
        local_model_repository_directory=local_model_repository_directory,
        model_name=model_name,
    )
    cache_from_local_repository(config)

    local_model_repository_file = (
        local_model_repository_directory + f"/{model_name}.tar.gz"
    )
    cached_file = get_local_model_cache_directory() + f"/{model_name}.tar.gz"

    assert filecmp.cmp(local_model_repository_file, cached_file)


def test_load_model_card(tiny_base_lm_local, mc_file, local_test_repository):
    """Tests if `load_model_card` loads a model card correctly."""
    model_name = tiny_base_lm_local

    local_model_repository_directory = local_test_repository

    config = StorageConfig(
        local_model_repository_directory=local_model_repository_directory,
        model_name=model_name,
        model_card_path=mc_file,
    )

    model_card = load_model_card(config)
    assert isinstance(model_card, SalmonModelCard)


def test_upload_to_local_repository(
    tiny_base_lm_local, mc_file, local_test_repository, local_test_working_dir
):
    """Tests if `upload_to_local_repository` uploads correctly."""
    model_name = tiny_base_lm_local

    local_model_repository_directory = local_test_repository
    local_working_directory = local_test_working_dir
    local_model_path = local_working_directory + f"/{model_name}"

    extract_model_locally(local_model_path, local_working_directory, model_name)

    config = StorageConfig(
        local_model_repository_directory=local_model_repository_directory,
        local_model_path=local_model_path,
        model_name=model_name,
    )

    mode_card = SalmonModelCard.load_modelcard(mc_file)

    upload_to_local_repository(config, mode_card)
    model_repo_file = local_model_repository_directory + f"/{model_name}.tar.gz"
    model_card_repo_file = (
        local_model_repository_directory + f"/{model_name}.md"
    )
    assert os.path.exists(model_repo_file) and os.path.exists(
        model_card_repo_file
    )


def test_download_from_cache(tiny_base_lm_local, local_test_download_dir):
    """Tests if `download_from_cache` downloads correctly."""
    model_name = tiny_base_lm_local
    put_model_into_cache(model_name)
    local_download_directory = local_test_download_dir

    config = StorageConfig(
        model_name=model_name,
        local_model_download_directory=local_download_directory,
    )
    download_from_cache(config)
    target_dir = (
        config.local_model_download_directory + f"/{config.model_name}/"
    )
    assert os.path.exists(target_dir) and os.path.isdir(target_dir)


def test_download_from_local_repository(
    tiny_base_lm_local, local_test_repository, local_test_download_dir
):
    """Tests if `download_from_local_repository` downloads correctly."""
    model_name = tiny_base_lm_local

    local_model_repository_directory = local_test_repository

    local_working_directory = local_test_download_dir

    put_model_into_local_repository(
        model_name, local_model_repository_directory
    )

    config = StorageConfig(
        local_model_repository_directory=local_model_repository_directory,
        local_model_download_directory=local_working_directory,
        model_name=model_name,
    )

    download_from_local_repository(config)
    target_dir = (
        config.local_model_download_directory + f"/{config.model_name}/"
    )
    assert os.path.exists(target_dir) and os.path.isdir(target_dir)


def test_local_from_pretrained(
    tiny_base_lm_local, mc_file, local_test_repository, local_test_working_dir
):
    """Tests if local storage model load works."""
    config = init_config(
        model_name=tiny_base_lm_local,
        model_card_path=mc_file,
        local_model_repository_directory=local_test_repository,
        local_working_directory=local_test_working_dir,
    )

    # upload a model first
    LocalModelStorage.upload(config)
    # download the model and check if they are the same
    model = LocalModelStorage.from_pretrained(config)

    local_model = AutoModel.from_pretrained(config.local_model_path)

    assert type(model) is type(local_model)


def test_local_from_pretrained_tokenizer(
    tiny_base_lm_local, mc_file, local_test_repository, local_test_working_dir
):
    """Tests if the local storage tokenizer load works."""
    config = init_config(
        model_name=tiny_base_lm_local,
        model_card_path=mc_file,
        local_model_repository_directory=local_test_repository,
        local_working_directory=local_test_working_dir,
    )
    # upload a model first
    LocalModelStorage.upload(config)
    # download the model and check if they are the same
    tokenizer = LocalModelStorage.from_pretrained_tokenizer(config)

    local_tokenizer = AutoTokenizer.from_pretrained(config.local_model_path)

    assert type(tokenizer) is type(local_tokenizer)


def test_local_download(
    tiny_base_lm_local,
    mc_file,
    local_test_repository,
    local_test_working_dir,
    local_test_download_dir,
):
    """Tests of the local storage upload works."""
    config = init_config(
        model_name=tiny_base_lm_local,
        model_card_path=mc_file,
        local_model_repository_directory=local_test_repository,
        local_working_directory=local_test_working_dir,
    )

    config.local_model_download_directory = local_test_download_dir

    # upload a model first
    LocalModelStorage.upload(config)

    # download the model and check if they are the same
    LocalModelStorage.download(config)

    downloaded_model = LocalModelStorage.from_pretrained(config)

    local_model = AutoModel.from_pretrained(
        config.local_model_download_directory + f"/{config.model_name}"
    )

    assert type(downloaded_model) is type(local_model)


def test_local_upload(
    tiny_base_lm_local,
    mc_file,
    local_test_repository,
    local_test_working_dir,
):
    """Tests if the local storage download works."""
    config = init_config(
        model_name=tiny_base_lm_local,
        model_card_path=mc_file,
        local_model_repository_directory=local_test_repository,
        local_working_directory=local_test_working_dir,
    )
    # upload a model first
    LocalModelStorage.upload(config)
    model_repo_file = (
        config.local_model_repository_directory + f"/{config.model_name}.tar.gz"
    )

    model_card_repo_file = (
        config.local_model_repository_directory + f"/{config.model_name}.md"
    )

    # check if the model is uploaded in the repository
    assert os.path.exists(model_repo_file) and os.path.exists(
        model_card_repo_file
    )


def test_local_home_directory():
    """Tests if the local home directory is correct."""
    assert get_local_home_directory() == str(pathlib.Path.home())


def test_local_cache_directory():
    """Tests if the local cache directory is correct."""
    assert (
        get_local_cache_directory()
        == get_local_home_directory() + "/.salmon/cache"
    )


def test_local_model_cache_directory():
    """Tests if the local model cache is correct."""
    assert (
        get_local_model_cache_directory()
        == get_local_cache_directory() + "/" + "models"
    )
