"""Conftest definition."""
import os
import pathlib
import tarfile
from typing import Any
from typing import Dict

import boto3
import pytest
from moto import mock_s3

from salmon.metadata.model_metadata import extract_modelcard_metadata
from salmon.models import Model
from salmon.storage.storage_config import StorageConfig
from tests.common.s3_test_utils import create_mock_s3_bucket


@pytest.fixture(name="mc_file", scope="function")
def fixture_mc_file() -> str:
    """Location of the test model card."""
    # Note the different way of naming this fixture.
    # More details: https://stackoverflow.com/a/57015304/2263844
    yield "tests/models/model-card.md"


@pytest.fixture(scope="function")
def mc_dict(mc_file) -> Dict:
    """Load the content of a model card into a dict."""
    yield extract_modelcard_metadata(mc_file)


@pytest.fixture(scope="function")
def lm_training_file() -> str:
    """Location of the language model training file."""
    yield "tests/data/lex_glue/train_lm_scotus_small.jsonl"


@pytest.fixture(scope="function")
def lm_test_file() -> str:
    """Location of the text classification test file."""
    yield "tests/data/lex_glue/test_lm_scotus_small.jsonl"


@pytest.fixture(scope="function")
def text_classification_training_file() -> str:
    """Location of the text classification training file."""
    yield "tests/data/lex_glue/train_ds_tc_scotus_small.jsonl"


@pytest.fixture(scope="function")
def text_classification_test_file() -> str:
    """Location of the text classification test file."""
    yield "tests/data/lex_glue/test_ds_tc_scotus_small.jsonl"


@pytest.fixture(scope="function")
def text_classification_val_file() -> str:
    """Location of the text classification validation file."""
    yield "tests/data/lex_glue/val_ds_tc_scotus_small.jsonl"


@pytest.fixture(scope="function")
def ner_training_file() -> str:
    """Location of the NER training file."""
    yield "tests/data/un_ner/un_train_ner_small.jsonl"


@pytest.fixture(scope="function")
def ner_test_file() -> str:
    """Location of the NER test file."""
    yield "tests/data/un_ner/un_test_ner_small.jsonl"


@pytest.fixture(scope="function")
def ner_val_file() -> str:
    """Location of the NER validation file."""
    yield "tests/data/un_ner/un_val_ner_small.jsonl"


@pytest.fixture(scope="function")
def tiny_base_lm() -> str:
    """Name of the tiny language model for test."""
    yield "sshleifer/tiny-distilroberta-base"


@pytest.fixture(scope="function")
def base_roberta_tokenizer() -> str:
    """Name of the base roberta tokenizer."""
    yield "distilroberta-base"


@pytest.fixture(name="tiny_base_lm_local", scope="function")
def fixture_tiny_base_lm_local() -> str:
    """Name of the tiny language model for test."""
    yield "tiny-distilroberta-base"


@pytest.fixture(name="tiny_base_lm_local_file", scope="function")
def fixture_tiny_base_lm_local_file(tiny_base_lm_local) -> str:
    """Name of the tiny language model for test."""
    yield f"tests/models/{tiny_base_lm_local}.tar.gz"


@pytest.fixture(name="local_test_tmp", scope="function")
def fixture_local_test_tmp(tmpdir) -> str:
    """
    Get temporary directory for unit-tests.

    Returns
    -------
    Returns temporary directory for unit-tests.
    """
    tmpdir_str = str(tmpdir)
    yield tmpdir_str


@pytest.fixture(name="local_test_repository", scope="function")
def fixture_local_test_repository(local_test_tmp) -> str:
    """
    Get repository directory for unit-tests.

    Returns
    -------
    Returns repository directory for unit-tests.
    """
    directory = local_test_tmp + "/repository"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    yield directory


@pytest.fixture(name="local_test_working_dir", scope="function")
def fixture_local_test_working_dir(local_test_tmp) -> str:
    """
    Get working directory for unit-tests.

    Returns
    -------
    Returns working directory for unit-tests.
    """
    directory = local_test_tmp + "/working"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    yield directory


@pytest.fixture(name="local_test_download_dir", scope="function")
def fixture_local_test_download_dir(local_test_tmp) -> str:
    """
    Get download directory for unit-tests.

    Returns
    -------
    Returns download directory for unit-tests.
    """
    directory = local_test_tmp + "/download"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="package")
def distilroberta_base_model() -> Model:
    """Load a distilroberta base huggingface model."""
    mdl = Model("distilroberta-base")
    return mdl


@pytest.fixture(scope="package")
def tiny_distilroberta_base_model() -> Model:
    """Load a tiny distilroberta base huggingface model."""
    mdl = Model("sshleifer/tiny-distilroberta-base")
    return mdl


@pytest.fixture(name="test_aws_region", scope="function")
def fixture_test_aws_region() -> str:
    """S3 region name."""
    yield "eu-central-1"


@pytest.fixture(name="aws_credentials", scope="function")
def fixture_aws_credentials(test_aws_region) -> None:
    """Mock AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    # No LocationConstraint ==> us-east-1
    os.environ["AWS_DEFAULT_REGION"] = test_aws_region


@pytest.fixture(scope="function")
def s3_client(aws_credentials, test_aws_region) -> Any:
    """Mock S3 client."""
    # pylint: disable=unused-argument
    # aws_credentials needs to be invoked for the mock to work.
    with mock_s3():
        yield boto3.client("s3", region_name=test_aws_region)


@pytest.fixture(name="s3_resource", scope="function")
def fixture_s3_resource(aws_credentials, test_aws_region) -> Any:
    """Mock S3 resource."""
    # pylint: disable=unused-argument
    # aws_credentials needs to be invoked for the mock to work.
    with mock_s3():
        yield boto3.resource("s3", region_name=test_aws_region)


@pytest.fixture(name="s3_model_bucket_name", scope="function")
def fixture_s3_model_bucket_name() -> str:
    """Salmon test bucket name."""
    yield "test-salmon-models"


@pytest.fixture(name="init_config_s3", scope="function")
def fixture_init_config_s3(
    tiny_base_lm_local,
    mc_file,
    s3_model_bucket_name,
    local_test_working_dir,
    test_aws_region,
    s3_resource,
) -> StorageConfig:
    """
    Test utility function to initiate an StorageConfig instance for S3 tests.

    Returns
    -------
    An instance of StorageConfig.
    """
    local_model_path = local_test_working_dir + f"/{tiny_base_lm_local}"
    create_mock_s3_bucket(s3_resource, s3_model_bucket_name, test_aws_region)

    # extract the test model into the working directory
    test_model_file = f"tests/models/{tiny_base_lm_local}.tar.gz"
    if not os.path.exists(local_model_path):
        with tarfile.open(test_model_file, "r:gz") as tar:
            tar.extractall(path=local_test_working_dir)
    config = StorageConfig(
        s3_model_repository_bucket=s3_model_bucket_name,
        local_model_path=local_model_path,
        model_name=tiny_base_lm_local,
        model_card_path=mc_file,
    )
    return config
