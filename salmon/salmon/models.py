"""Functions for using pretrained models."""
import logging
from typing import List
from typing import Optional

import numpy as np
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class Model:
    """Wrapper class for huggingface models."""

    def __init__(
        self, model_name: str, tokeniser_name: Optional[str] = None
    ) -> None:
        """Create the Model."""
        if not tokeniser_name:
            tokeniser_name = model_name

        self.hf_toknsr: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokeniser_name
        )
        self.hf_model: PreTrainedModel = AutoModel.from_pretrained(model_name)
        logger.info(
            "Loaded model %s of type %s", model_name, type(self.hf_model)
        )

    def document_embedding(
        self,
        documents: List[str],
        aggregation,
        padding,
        truncation,
        max_length: Optional[int],
    ) -> np.array:
        """
        Compute an embedding for each of the given documents.

        Parameters
        ----------
        documents: List[str]
            A list of documents
        aggregation: str
        padding
        truncation
        max_length

        Raises
        ------
        ValueError:
            If the list of documents is empty.

        Returns
        -------
        np.array:
            The embeddings of the documents, shape=(n, 768) for n documents
        """
        if not documents:
            # the list of documents is empty. That's not allowed
            raise ValueError("The list of document may not be empty")

        # is this now torch-specific since we're asking for pt tensors?
        inputs = self.hf_toknsr(
            documents,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        outputs = self.hf_model(**inputs)
        lhs = outputs.last_hidden_state

        logging.info(
            "Computing doc embeddings with aggregation method %s", aggregation
        )
        cls_embedding = lhs[:, 0, :]

        return cls_embedding.detach().numpy()


def compute_embedding(
    model_name: str,
    tokenizer_name: str,
    documents: List[str],
    aggregation,
    padding,
    truncation,
    max_length: Optional[int],
) -> np.array:
    """
    Compute embeddings for the given documents from the given model.

    Parameters
    ----------
    model_name: str
        A huggingface model or local model name
    tokenizer_name: str
        A huggingface tokeniser or local tokeniser name
    documents: List[str]
        A list of documents
    aggregation
    padding
    truncation
    max_length

    Returns
    -------
    np.array:
        The embeddings of the documents, shape=(n, 768) for n documents
    """
    mdl = Model(model_name=model_name, tokeniser_name=tokenizer_name)
    embeddings = mdl.document_embedding(
        documents,
        aggregation=aggregation,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )
    return embeddings
