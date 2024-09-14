"""Tests for Salmon models."""
from typing import List

import numpy as np
import pytest

from salmon.cli import EmbeddingOutputFormat
from salmon.models import Model


@pytest.mark.parametrize(
    "docs, raises_valueerror",
    [
        (["This is a document"], False),
        (["This is a document"] * 10, False),
        ([], True),
    ],
)
def test_document_embedding(
    tiny_distilroberta_base_model: Model,
    docs: List[str],
    raises_valueerror: bool,
):
    """Check that the document embedding is the right size."""
    if raises_valueerror:
        with pytest.raises(ValueError):
            _emb = tiny_distilroberta_base_model.document_embedding(
                docs,
                aggregation="cls",
                padding=True,
                truncation=True,
                max_length=100,
            )
    else:
        embedding_dim = (
            tiny_distilroberta_base_model.hf_model.config.hidden_size
        )
        expected_embedding_shape = (len(docs), embedding_dim)

        emb = tiny_distilroberta_base_model.document_embedding(
            docs,
            aggregation="cls",
            padding=True,
            truncation=True,
            max_length=100,
        )

        assert emb.shape == expected_embedding_shape


@pytest.mark.parametrize(
    "serialisation_format, expected_error",
    [("numpy", None), ("json", NotImplementedError)],
)
def test_document_embedding_serialisation(
    tmp_path,
    tiny_distilroberta_base_model,
    serialisation_format,
    expected_error,
):
    """Check that embeddings are serialised correctly."""
    docs = ["This is a document"]
    embedding_dim = tiny_distilroberta_base_model.hf_model.config.hidden_size
    expected_embedding_shape = (len(docs), embedding_dim)

    emb = tiny_distilroberta_base_model.document_embedding(
        docs,
        aggregation="cls",
        padding=True,
        truncation=True,
        max_length=100,
    )

    output_format = EmbeddingOutputFormat(serialisation_format)
    output_path = tmp_path / "outputfile"

    if expected_error:
        with pytest.raises(expected_error):
            output_format.write(emb, output_path)
    else:
        output_format.write(emb, output_path)

        # read back in a check the shape
        with open(str(output_path), "rb") as f:
            arr = np.load(f)
            assert arr.shape == expected_embedding_shape
