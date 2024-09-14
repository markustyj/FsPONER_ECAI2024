"""Module for model card related code."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Dict

import yaml

from salmon.errors.exceptions import ModelCardError

logger = logging.getLogger(__name__)


class SalmonModelCard:
    """Model metadata."""

    def __init__(self, **kwargs):
        """
        Initiate a model card.

        Parameters
        ----------
        kwargs: **dictionary containing the model card.
        """
        validate_model_card(kwargs)
        self.language = kwargs.pop("language", {})
        self.license = kwargs.pop("license", {})
        self.tags = kwargs.pop("tags", {})
        self.datasets = kwargs.pop("datasets", {})
        self.metrics = kwargs.pop("metrics", {})

        # Open additional attributes
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                msg = f"Can't set {key} with value {value} for {self}"
                logger.error(msg)
                raise

    def serialize(self, file_path):
        """
        Serialize the model card to a JSON file.

        Parameters
        ----------
        file_path: the path of the file where the JSON output should be
        serialized.
        """
        header_dict = self.__dict__.copy()
        footer = ""
        if "md_text" in header_dict:
            footer = header_dict.pop("md_text")
            if not footer.startswith("\n"):
                footer = "\n" + footer
        mc_str = f"---\n{yaml.dump(header_dict)}---{footer}"
        with open(file_path, "w", encoding="utf-8") as mc_file:
            mc_file.write(mc_str)

    @classmethod
    def load_modelcard(cls, file: str) -> SalmonModelCard:
        """
        Load model card from a file.

        Typing annotation for self:
        https://stackoverflow.com/q/33533148/2263844

        Parameters
        ----------
        file: model card metadata file.

        Returns
        -------
        An instance of the model card.

        """
        meta_dict = extract_modelcard_metadata(file)
        return cls(**meta_dict)

    def __eq__(self, other):
        """
        Check model card equality.

        Parameters
        ----------
        other: an instance of SalmonModelCard.

        Returns
        -------
        True or False.
        """
        if not isinstance(other, SalmonModelCard):
            # don't attempt to compare against unrelated types
            return NotImplemented
        self_att_dict = self.__dict__
        self_hash = hashlib.sha1(
            json.dumps(self_att_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()
        other_att_dict = other.__dict__
        other_hash = hashlib.sha1(
            json.dumps(other_att_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self_hash == other_hash


# pylint: disable=too-many-branches
def validate_model_card(mode_card_dict: Dict) -> bool:
    """
    Validate a model card.

    Parameters
    ----------
    mode_card_dict: a dictionary where the model card is loaded.

    Returns
    -------
    True if the model card is valid. Otherwise it throws an exception.
    """
    if "language" not in mode_card_dict:
        raise ModelCardError("`language` missing.")

    if "license" not in mode_card_dict:
        raise ModelCardError("`license` missing.")

    if "tags" not in mode_card_dict:
        raise ModelCardError("`tags` missing.")

    if "datasets" not in mode_card_dict:
        raise ModelCardError("`datasets` missing.")

    if "metrics" not in mode_card_dict:
        raise ModelCardError("`metrics` missing.")

    if "model-index" in mode_card_dict:
        for i, model_index in enumerate(mode_card_dict["model-index"]):
            model_number_text = f"in model-index {i}."
            if "name" not in model_index:
                raise ModelCardError(
                    "`name` missing under `model-index` " + model_number_text
                )
            if "results" not in model_index:
                raise ModelCardError(
                    "`results` missing under `model-index` " + model_number_text
                )
            for resi, result in enumerate(model_index["results"]):
                res_number_text = f"in result number {resi} in model-index {i}."
                if "task" not in result:
                    raise ModelCardError("`task` missing " + res_number_text)
                if "dataset" not in result:
                    raise ModelCardError("`dataset` missing " + res_number_text)
                if "metrics" not in result:
                    raise ModelCardError("`metrics` missing " + res_number_text)

                if "type" not in result["task"]:
                    raise ModelCardError(
                        "`task.type` missing " + res_number_text
                    )
                if "type" not in result["dataset"]:
                    raise ModelCardError(
                        "`dataset.type` missing " + res_number_text
                    )
                if "name" not in result["task"]:
                    raise ModelCardError(
                        "`task.name` missing " + res_number_text
                    )
                if "name" not in result["dataset"]:
                    raise ModelCardError(
                        "`dataset.name` missing " + res_number_text
                    )

                for meti, met in enumerate(result["metrics"]):
                    met_num_txt = (
                        f"in metric number {meti} in result "
                        f"number {resi} in model-index {i}."
                    )
                    if "type" not in met:
                        raise ModelCardError(
                            "`metrics.type` missing " + met_num_txt
                        )
                    if "value" not in met:
                        raise ModelCardError(
                            "`metrics.value` missing " + met_num_txt
                        )

    return True


def extract_modelcard_metadata(file: str) -> Dict:
    """
    Extract model card metadata from a file.

    Parameters
    ----------
    file: model card metadata file.

    Returns
    -------
    A dictionary with model card metadata as specified
    here https://huggingface.co/docs/hub/model-repos with an additon key
    called `md_text` which holds the markdown free text when available.
    """
    with open(file, "r", encoding="utf-8") as model_card:
        model_card_text = model_card.read()
        matches = re.findall(
            "^.*?---(.*?)---(.*$)", model_card_text, flags=re.DOTALL
        )
        if not len(matches) == 1:
            raise ModelCardError("Model card markup not found!")
        dct = yaml.safe_load(matches[0][0])
        md_txt = ""
        if len(matches[0]) == 2:
            md_txt = matches[0][1]

        dct["md_text"] = md_txt
        return dct
