"""Test Salmon S3 storage layer."""
import filecmp
import os

from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import RobertaTokenizerFast

from salmon.metadata.model_metadata import SalmonModelCard
from salmon.storage.storage_config import StorageConfig
from salmon.storage.storage_config import get_local_model_cache_directory
from tests.common.local_storage_test_utils import extract_model_locally
from tests.common.s3_test_utils import create_mock_s3_bucket
from tests.common.s3_test_utils import put_model_card_into_s3_repository
from tests.common.s3_test_utils import put_model_into_s3_repository

# pylint: disable=import-outside-toplevel
# Needs to be done this way for the moto mocks to work.


def test_create_bucket(s3_client, test_aws_region):
    """Test moto mocking."""
    # s3 is a fixture defined in conftest.py that yields a boto3 s3 client.
    # No LocationConstraint ==> us-east-1
    if test_aws_region == "us-east-1":
        s3_client.create_bucket(
            Bucket="somebucket",
        )
    else:
        s3_client.create_bucket(
            Bucket="somebucket",
            CreateBucketConfiguration={"LocationConstraint": test_aws_region},
        )

    result = s3_client.list_buckets()

    assert len(result["Buckets"]) == 1
    assert result["Buckets"][0]["Name"] == "somebucket"


def test_mock_upload_model(
    s3_resource,
    s3_model_bucket_name,
    tmpdir,
    tiny_base_lm_local_file,
    tiny_base_lm_local,
    test_aws_region,
):
    """Test mock s3 upload model."""
    test_model_file = tiny_base_lm_local_file
    model_key = f"{tiny_base_lm_local}.tar.gz"
    download_file = f"{str(tmpdir)}/{tiny_base_lm_local}.tar.gz"

    bucket = create_mock_s3_bucket(
        s3_resource, s3_model_bucket_name, test_aws_region
    )
    bucket.upload_file(test_model_file, model_key)
    bucket.download_file(model_key, download_file)
    assert filecmp.cmp(download_file, test_model_file)


def test_load_model_from_s3_repository(
    tiny_base_lm_local,
    s3_model_bucket_name,
    tiny_base_lm_local_file,
    s3_resource,
    test_aws_region,
):
    """Tests if `load_model_from_s3_repository` loads a model correctly."""
    from salmon.storage.storage_factory import load_model_from_s3_repository

    # import is here because the mocks should be done before the clients are
    # setup, see this for more info:
    # http://docs.getmoto.org/en/latest/docs/getting_started.html
    # #what-about-those-pesky-imports
    model_name = tiny_base_lm_local

    bucket = create_mock_s3_bucket(
        s3_resource, s3_model_bucket_name, test_aws_region
    )

    put_model_into_s3_repository(
        model_name,
        tiny_base_lm_local_file,
        bucket,
    )

    config = StorageConfig(
        s3_model_repository_bucket=s3_model_bucket_name,
        model_name=model_name,
    )

    model = load_model_from_s3_repository(config)
    assert isinstance(model, RobertaModel)


def test_cache_model_from_s3(
    tiny_base_lm_local,
    s3_model_bucket_name,
    tiny_base_lm_local_file,
    s3_resource,
    test_aws_region,
):
    """Tests if `cache_from_s3_repository` caches a model correctly."""
    from salmon.storage.storage_factory import cache_from_s3_repository

    model_name = tiny_base_lm_local

    bucket = create_mock_s3_bucket(
        s3_resource, s3_model_bucket_name, test_aws_region
    )

    put_model_into_s3_repository(
        model_name,
        tiny_base_lm_local_file,
        bucket,
    )

    config = StorageConfig(
        s3_model_repository_bucket=s3_model_bucket_name,
        model_name=model_name,
    )
    cache_from_s3_repository(config)

    cached_file = get_local_model_cache_directory() + f"/{model_name}.tar.gz"

    assert filecmp.cmp(tiny_base_lm_local_file, cached_file)


def test_upload_to_s3_repository(
    tiny_base_lm_local,
    mc_file,
    s3_model_bucket_name,
    local_test_working_dir,
    s3_resource,
    test_aws_region,
    tiny_base_lm_local_file,
):
    """Tests if `upload_to_s3_repository` uploads correctly."""
    from salmon.storage.storage_factory import upload_to_s3_repository

    model_name = tiny_base_lm_local

    local_working_directory = local_test_working_dir
    local_model_path = local_working_directory + f"/{model_name}"

    extract_model_locally(local_model_path, local_working_directory, model_name)

    config = StorageConfig(
        s3_model_repository_bucket=s3_model_bucket_name,
        local_model_path=local_model_path,
        model_name=model_name,
    )

    bucket = create_mock_s3_bucket(
        s3_resource, s3_model_bucket_name, test_aws_region
    )

    # to test overwrite
    put_model_into_s3_repository(
        model_name,
        tiny_base_lm_local_file,
        bucket,
    )
    # to test overwrite
    put_model_card_into_s3_repository(model_name, mc_file, bucket)

    mode_card = SalmonModelCard.load_modelcard(mc_file)

    upload_to_s3_repository(config, mode_card)
    model_repo_file = f"{model_name}.tar.gz"
    model_card_repo_file = f"{model_name}.md"

    objs_model = list(bucket.objects.filter(Prefix=model_repo_file))
    objs_card = list(bucket.objects.filter(Prefix=model_card_repo_file))

    assert len(objs_model) > 0 and len(objs_card) > 0


def test_download_from_s3_repository(
    tiny_base_lm_local,
    s3_model_bucket_name,
    s3_resource,
    test_aws_region,
    tiny_base_lm_local_file,
    local_test_download_dir,
):
    """Tests if `download_from_s3_repository` downloads correctly."""
    from salmon.storage.storage_factory import download_from_s3_repository

    model_name = tiny_base_lm_local

    local_working_directory = local_test_download_dir
    local_model_path = local_working_directory + f"/{model_name}"

    bucket = create_mock_s3_bucket(
        s3_resource, s3_model_bucket_name, test_aws_region
    )

    put_model_into_s3_repository(
        model_name,
        tiny_base_lm_local_file,
        bucket,
    )

    config = StorageConfig(
        s3_model_repository_bucket=s3_model_bucket_name,
        local_model_path=local_model_path,
        model_name=model_name,
        local_model_download_directory=local_working_directory,
    )

    download_from_s3_repository(config)
    target_dir = (
        config.local_model_download_directory + f"/{config.model_name}/"
    )
    assert os.path.exists(target_dir) and os.path.isdir(target_dir)


def test_load_tokenizer_from_s3_repository(
    tiny_base_lm_local,
    s3_model_bucket_name,
    s3_resource,
    test_aws_region,
    tiny_base_lm_local_file,
):
    """Tests if a tokenizer can be loaded from s3 correctly."""
    from salmon.storage.storage_factory import load_tokenizer_from_s3_repository

    model_name = tiny_base_lm_local

    bucket = create_mock_s3_bucket(
        s3_resource, s3_model_bucket_name, test_aws_region
    )

    put_model_into_s3_repository(
        model_name,
        tiny_base_lm_local_file,
        bucket,
    )

    config = StorageConfig(
        s3_model_repository_bucket=s3_model_bucket_name,
        model_name=model_name,
    )
    tokenizer = load_tokenizer_from_s3_repository(config)
    assert isinstance(tokenizer, RobertaTokenizerFast)


def test_s3_from_pretrained(init_config_s3):
    """Tests if s3 storage model load works."""
    from salmon.storage.model_storage import S3ModelStorage

    config = init_config_s3

    # upload a model first
    S3ModelStorage.upload(config)
    # download the model and check if they are the same
    model = S3ModelStorage.from_pretrained(config)

    local_model = AutoModel.from_pretrained(config.local_model_path)

    assert type(model) is type(local_model)


def test_s3_from_pretrained_tokenizer(init_config_s3):
    """Tests if the s3 storage tokenizer load works."""
    from salmon.storage.model_storage import S3ModelStorage

    config = init_config_s3
    # upload a model first
    S3ModelStorage.upload(config)
    # download the model and check if they are the same
    tokenizer = S3ModelStorage.from_pretrained_tokenizer(config)

    local_tokenizer = AutoTokenizer.from_pretrained(config.local_model_path)

    assert type(tokenizer) is type(local_tokenizer)


def test_s3_download(init_config_s3, local_test_download_dir):
    """Tests of the s3 storage upload works."""
    from salmon.storage.model_storage import S3ModelStorage

    config = init_config_s3

    config.local_model_download_directory = local_test_download_dir

    # upload a model first
    S3ModelStorage.upload(config)

    # download the model and check if they are the same
    S3ModelStorage.download(config)

    downloaded_model = S3ModelStorage.from_pretrained(config)

    local_model = AutoModel.from_pretrained(
        config.local_model_download_directory + f"/{config.model_name}"
    )

    assert type(downloaded_model) is type(local_model)


def test_s3_upload(init_config_s3, s3_resource):
    """Tests if the s3 storage download works."""
    from salmon.storage.model_storage import S3ModelStorage

    config = init_config_s3
    # upload a model first
    S3ModelStorage.upload(config)
    model_repo_file = f"{config.model_name}.tar.gz"

    model_card_repo_file = f"{config.model_name}.md"

    # check if the model is uploaded in the repository
    bucket = s3_resource.Bucket(config.s3_model_repository_bucket)
    objs_model = list(bucket.objects.filter(Prefix=model_repo_file))
    objs_card = list(bucket.objects.filter(Prefix=model_card_repo_file))
    assert len(objs_model) > 0 and len(objs_card) > 0
