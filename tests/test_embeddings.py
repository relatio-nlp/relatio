import numpy as np
from numpy.linalg import norm

from relatio.embeddings import Embeddings

spacy_model_sif = Embeddings(
    "spaCy",
    "en_core_web_md",
    sentences=["this is a nice world", "hello world", "hello everybody"],
)


def test_sif_weight_out_of_dict():
    # "zero" is not part of the senteces given to Embeddings
    assert spacy_model_sif._sif_dict["no"] == 1.0


def test_get_vector_for_sif_out_of_dict():
    res = spacy_model_sif.get_vector("no world")
    expected = (
        spacy_model_sif._get_default_vector("no")
        + spacy_model_sif._get_default_vector("world")
        * spacy_model_sif._sif_dict["world"]
    )

    np.testing.assert_array_equal(res, expected / norm(expected))
