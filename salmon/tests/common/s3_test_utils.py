"""S3 test utilities."""
from typing import Any


def put_model_into_s3_repository(
    model_name: str, local_model_file: str, bucket: Any
) -> None:
    """
    Put a model into the s3 repository.

    Parameters
    ----------
    model_name: name of the model.
    local_model_file: location of the model tar.gz file in the local machine.
    bucket: s3 bucket name.

    Returns
    -------
    None

    """
    model_key = f"{model_name}.tar.gz"
    bucket.upload_file(local_model_file, model_key)


def put_model_card_into_s3_repository(
    model_name: str, model_card_file: str, bucket: Any
) -> None:
    """
    Put a model card into the s3 repository.

    Parameters
    ----------
    model_name: name of the model.
    model_card_file: location of the model card file in the local machine.
    bucket: s3 bucket name.

    Returns
    -------
    None
    """
    mc_key = f"{model_name}.md"
    bucket.upload_file(model_card_file, mc_key)


def create_mock_s3_bucket(
    s3_resource: Any, s3_model_bucket_name: str, test_aws_region: str
) -> Any:
    """
    Create a mock s3 bucket.

    Parameters
    ----------
    s3_resource: s3 resource.
    s3_model_bucket_name: s3 bucket name
    test_aws_region: s3 region name.

    Returns
    -------
    Bucket instance.

    """
    bucket = s3_resource.Bucket(s3_model_bucket_name)
    if test_aws_region == "us-east-1":
        bucket.create()
    else:
        bucket.create(
            CreateBucketConfiguration={"LocationConstraint": test_aws_region}
        )
    return bucket
