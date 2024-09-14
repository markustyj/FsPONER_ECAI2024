"""Utility functions for dataset preprocessing."""
import logging
from typing import List

from datasets import DatasetDict

logger = logging.getLogger(__name__)


def drop_features_except(
    dataset: DatasetDict, retain_fields: List[str]
) -> DatasetDict:
    """
    Drop all features from the given Dataset except those specified.

    Typically this function is used to retain the "text" and "label" features.

    Parameters
    ----------
    dataset: DatasetDict
        A dataset, typically containing train and test sets
    retain_fields: List[str]
        A list of features which will be retained in each dataset

    Returns
    -------
    DatasetDict:
        The new dataset containing only the features which were retained

    Raises
    ------
    ValueError:
        If the fields to retain are not in the data set
    """
    if not retain_fields:
        msg = (
            "There should be at least one feature to retain "
            "(an empty list was passed)."
        )
        logger.warning(msg)
        raise ValueError(msg)

    result = DatasetDict()
    for name, single_dataset in dataset.items():
        missing_fields = set(retain_fields).difference(
            set(single_dataset.column_names)
        )
        if missing_fields:
            msg = (
                f"Some features are not part of this dataset and could not"
                f" be retained: {missing_fields}. Features in the "
                f"dataset: {single_dataset.column_names}"
            )
            logger.warning(msg)
            raise ValueError(msg)

        remove_these_columns = set(single_dataset.column_names).difference(
            set(retain_fields)
        )
        result[name] = single_dataset.remove_columns(remove_these_columns)

    return result
