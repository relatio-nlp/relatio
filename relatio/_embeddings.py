# MIT License

# Copyright (c) 2020-2021 ETH Zurich, Andrei V. Plamada
# Copyright (c) 2020-2021 ETH Zurich, Elliott Ash
# Copyright (c) 2020-2021 University of St.Gallen, Philine Widmer
# Copyright (c) 2020-2021 Ecole Polytechnique, Germain Gauthier

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import spacy
from numpy.linalg import norm
from spacy.cli import download as spacy_download

from relatio.utils import count_words


class EmbeddingsBase(ABC):
    @abstractmethod
    def _get_default_vector(self, phrase: str) -> np.ndarray:
        pass


class Embeddings(EmbeddingsBase):
    # TODO: lower the case of the input (phrase and sentences)
    # TODO: for SIF many things can go wrong since we do the linear combination of potential nan values
    """
    When sentences is used in the constructor the embeddings are weighted by the smoothed inverse frequency of each token.
    For further details, see: https://github.com/PrincetonML/SIF


    Examples:
        >>> model = Embeddings("TensorFlow_USE","https://tfhub.dev/google/universal-sentence-encoder/4")
        >>> model.get_vector("hello world").shape
        (512,)
        >>> model = Embeddings("spaCy", "en_core_web_md")
        >>> model.get_vector("") is None
        True
        >>> model.get_vector("hello world").shape
        (300,)
        >>> norm(model.get_vector("hello world")) < 1.001
        True
        >>> model = Embeddings("spaCy", "en_core_web_md", normalize=False)
        >>> norm(model.get_vector("hello world")) < 1.001
        False
        >>> model = Embeddings("Gensim_pretrained", "glove-twitter-25")
        >>> model.get_vector("world").shape
        (25,)
        >>> model = Embeddings("Gensim_pretrained", "glove-twitter-25",sentences = ["this is a nice world","hello world","hello everybody"])
        >>> model.get_vector("hello world").shape
        (25,)

    """

    def __init__(
        self,
        embeddings_type: str,
        embeddings_model: Union[Path, str],
        normalize: bool = True,
        sentences: Optional[List[str]] = None,
        alpha: float = 0.001,
        **kwargs,
    ) -> None:

        EmbeddingsClass: Union[
            Type[TensorFlowUSEEmbeddings],
            Type[GensimWord2VecEmbeddings],
            Type[GensimPreTrainedEmbeddings],
            Type[spaCyEmbeddings],
        ]
        if embeddings_type == "TensorFlow_USE":
            EmbeddingsClass = TensorFlowUSEEmbeddings
        elif embeddings_type == "Gensim_Word2Vec":
            EmbeddingsClass = GensimWord2VecEmbeddings
        elif embeddings_type == "Gensim_pretrained":
            EmbeddingsClass = GensimPreTrainedEmbeddings
        elif embeddings_type == "spaCy":
            EmbeddingsClass = spaCyEmbeddings
        else:
            raise ValueError(f"Unknown embeddings_type={embeddings_type}")

        self._embeddings_model = EmbeddingsClass(embeddings_model, **kwargs)
        self._normalize: bool = normalize
        if sentences is not None:
            self._sif_dict = self.compute_sif_weights(sentences=sentences, alpha=alpha)
            self._use_sif = True
        else:
            self._sif_dict = {}
            self._use_sif = False

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def use_sif(self) -> bool:
        return self._use_sif

    # One cannot add a setter since it is added next to the child classes

    def get_vector(self, phrase: str) -> Optional[np.ndarray]:
        tokens = phrase.split()

        if self.use_sif and len(tokens) > 1:

            res = np.mean(
                [
                    self._sif_dict[token] * self._get_default_vector(token)
                    for token in tokens
                ],
                axis=0,
            )

        else:
            res = self._get_default_vector(phrase)

        # in case the result is fishy it will return a None
        if res is None:
            return None
        elif np.isnan(res).any() or np.count_nonzero(res) == 0:
            return None

        if self.normalize:
            return res / norm(res)
        else:
            return res

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        return self._embeddings_model._get_default_vector(phrase)

    @staticmethod
    def compute_sif_weights(sentences: List[str], alpha: float) -> Dict[str, float]:

        """

        A function that computes smooth inverse frequency (SIF) weights based on word frequencies.
        (See "Arora, S., Liang, Y., & Ma, T. (2016). A simple but tough-to-beat baseline for sentence embeddings.")

        The sentences are used to build the counter dictionary {"word": frequency} which is further used to compute the sif weights
        Args:
            sentences: a list of sentences
            alpha: regularization parameter

        Returns:
            A dictionary {"word": SIF weight}

        """
        words_counter = count_words(sentences)

        sif_dict = {}

        for word, count in words_counter.items():
            sif_dict[word] = alpha / (alpha + count)

        return sif_dict


class spaCyEmbeddings(EmbeddingsBase):
    def __init__(self, model: str) -> None:
        if not spacy.util.is_package(model):
            spacy_download(model)
        self._nlp = spacy.load(model)

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        return np.array(self._nlp(phrase).vector)


class TensorFlowUSEEmbeddings(EmbeddingsBase):
    def __init__(self, path: str) -> None:
        try:
            import tensorflow_hub as hub
        except ModuleNotFoundError:
            print("Please install tensorflow_hub package")
            raise
        self._embed = hub.load(path)

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        return self._embed([phrase]).numpy()[0]

    def get_vector(self, phrase: str) -> np.ndarray:
        return self._get_default_vector(phrase)


class GensimWord2VecEmbeddings(EmbeddingsBase):
    def __init__(
        self,
        path: str,
    ):

        self._model = self._load_keyed_vectors(path)
        self._vocab = self._model.vocab

    def _load_keyed_vectors(self, path):
        try:
            from gensim.models import Word2Vec

        except ModuleNotFoundError:
            print("Please install gensim package")
            raise

        return Word2Vec.load(path).wv

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        return self._model[phrase]


class GensimPreTrainedEmbeddings(EmbeddingsBase):

    """

    A class to call a pre-trained embeddings model from gensim's library.

    # The list of pre-trained embeddings may be browsed by typing:

        import gensim.downloader as api
        list(api.info()['models'].keys())

    """

    def __init__(
        self,
        model: str,
    ):

        self._model = self._load_keyed_vectors(model)
        self._vocab = self._model.vocab

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        return self._model[phrase]

    def _load_keyed_vectors(self, model):
        try:
            import gensim.downloader as api

        except ModuleNotFoundError:
            print("Please install gensim package")
            raise

        return api.load(model)
