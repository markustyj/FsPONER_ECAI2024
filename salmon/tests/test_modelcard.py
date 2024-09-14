"""Model card tests."""
import pytest

from salmon.errors.exceptions import ModelCardError
from salmon.metadata.model_metadata import SalmonModelCard
from salmon.metadata.model_metadata import validate_model_card


def test_salmon_model_card_instance(mc_dict):
    """Test case for initialization."""
    smc = SalmonModelCard(**mc_dict)
    assert isinstance(smc, SalmonModelCard)


def test_salmon_model_card_load_modelcard(mc_dict, mc_file):
    """Test case for loading a model card."""
    smc1 = SalmonModelCard(**mc_dict)
    smc2 = SalmonModelCard.load_modelcard(mc_file)
    assert isinstance(smc2, SalmonModelCard) and smc1 == smc2


def test_salmon_model_card_serialize(tmpdir, mc_dict):
    """Test case for model card serialization."""
    smc1 = SalmonModelCard(**mc_dict)
    tmpdir_str = str(tmpdir)
    file = f"{tmpdir_str}/test_model_card.md"
    smc1.serialize(file)
    smc2 = SalmonModelCard.load_modelcard(file)
    assert smc1 == smc2


@pytest.mark.parametrize(
    "key, expected_error",
    [
        (
            "language",
            "`language` missing.",
        ),
        (
            "license",
            "`license` missing.",
        ),
        (
            "tags",
            "`tags` missing.",
        ),
        (
            "datasets",
            "`datasets` missing.",
        ),
        (
            "metrics",
            "`metrics` missing.",
        ),
    ],
)
def test_validate_model_card_missing_top_level(key, expected_error, mc_dict):
    """Test case for model card format validation of level 1."""
    mc_dict.pop(key)
    try:
        validate_model_card(mc_dict)
    except ModelCardError as err:
        assert str(err) == expected_error


@pytest.mark.parametrize(
    "key, expected_error",
    [
        (
            "name",
            "`name` missing under `model-index` in model-index 0.",
        ),
        (
            "results",
            "`results` missing under `model-index` in model-index 0.",
        ),
    ],
)
def test_validate_model_card_missing_level_2(key, expected_error, mc_dict):
    """Test case for model card format validation of level 2."""
    mc_dict["model-index"][0].pop(key)
    try:
        validate_model_card(mc_dict)
    except ModelCardError as err:
        assert str(err) == expected_error


@pytest.mark.parametrize(
    "key, expected_error",
    [
        (
            "task",
            "`task` missing in result number 0 in model-index 0.",
        ),
        (
            "dataset",
            "`dataset` missing in result number 0 in model-index 0.",
        ),
        (
            "metrics",
            "`metrics` missing in result number 0 in model-index 0.",
        ),
    ],
)
def test_validate_model_card_missing_level_3(key, expected_error, mc_dict):
    """Test case for model card format validation of level 3."""
    mc_dict["model-index"][0]["results"][0].pop(key)
    try:
        validate_model_card(mc_dict)
    except ModelCardError as err:
        assert str(err) == expected_error


@pytest.mark.parametrize(
    "level_key, key, expected_error",
    [
        (
            "task",
            "type",
            "`task.type` missing in result number 0 in model-index 0.",
        ),
        (
            "dataset",
            "type",
            "`dataset.type` missing in result number 0 in model-index 0.",
        ),
        (
            "task",
            "name",
            "`task.name` missing in result number 0 in model-index 0.",
        ),
        (
            "dataset",
            "name",
            "`dataset.name` missing in result number 0 in model-index 0.",
        ),
    ],
)
def test_validate_model_card_missing_level_4(
    level_key, key, expected_error, mc_dict
):
    """Test case for model card format validation of level 4."""
    mc_dict["model-index"][0]["results"][0][level_key].pop(key)
    try:
        validate_model_card(mc_dict)
    except ModelCardError as err:
        assert str(err) == expected_error


@pytest.mark.parametrize(
    "key, expected_error",
    [
        (
            "type",
            "`metrics.type` missing in metric number 0 in result number "
            "0 in model-index 0.",
        ),
        (
            "value",
            "`metrics.value` missing in metric number 0 in result number"
            " 0 in model-index 0.",
        ),
    ],
)
def test_validate_model_card_missing_level_5(key, expected_error, mc_dict):
    """Test case for model card format validation of level 5."""
    mc_dict["model-index"][0]["results"][0]["metrics"][0].pop(key)
    try:
        validate_model_card(mc_dict)
    except ModelCardError as err:
        assert str(err) == expected_error
