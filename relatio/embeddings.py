# MIT License

# Copyright (c) 2020-2021 ETH Zurich, Andrei V. Plamada
# Copyright (c) 2020-2021 ETH Zurich, Elliott Ash
# Copyright (c) 2020-2021 University of St.Gallen, Philine Widmer
# Copyright (c) 2020-2021 Ecole Polytechnique, Germain Gauthier

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import spacy
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from spacy.cli import download as spacy_download
from tqdm import tqdm

from relatio.supported_models import LANGUAGE_MODELS
from relatio.utils import count_words


class EmbeddingsBase(ABC):
    @abstractmethod
    def _get_default_vector(self, phrase: str) -> np.ndarray:
        pass


# TODO: add size_vectors as a class property (now it is in Embeddings as an instance attribute)


class Embeddings(EmbeddingsBase):
    """
    If sentences is used in the constructor the embeddings are weighted by the smoothed inverse frequency of each token.
    For further details, see: https://github.com/PrincetonML/SIF
    The input is expected in lowe case.
    Examples:
        >>> model = Embeddings("TensorFlow_USE","https://tfhub.dev/google/universal-sentence-encoder/4")
        >>> model.get_vector("hello world").shape
        (512,)
        >>> model = Embeddings("spaCy", "en_core_web_md")
        >>> np.isnan(model.get_vector("")).any()
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
        >>> model = Embeddings("Gensim_pretrained", "glove-twitter-25", sentences = ["this is a nice world","hello world","hello everybody"])
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
            Type[phraseBERTEmbeddings],
        ]
        if embeddings_type == "TensorFlow_USE":
            EmbeddingsClass = TensorFlowUSEEmbeddings
        elif embeddings_type == "Gensim_Word2Vec":
            EmbeddingsClass = GensimWord2VecEmbeddings
        elif embeddings_type == "Gensim_pretrained":
            EmbeddingsClass = GensimPreTrainedEmbeddings
        elif embeddings_type == "spaCy":
            EmbeddingsClass = spaCyEmbeddings
        elif embeddings_type == "phrase-BERT":
            EmbeddingsClass = phraseBERTEmbeddings
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

        if embeddings_type != "Gensim_Word2Vec":
            self.size_vectors = LANGUAGE_MODELS[embeddings_model]["size_vectors"]
        else:
            self.size_vectors = self._embeddings_model.size_vectors

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def use_sif(self) -> bool:
        return self._use_sif

    # One cannot add a setter since it is added next to the child classes
    def get_vector(self, phrase: str) -> Optional[np.ndarray]:
        tokens = phrase.split()

        if self.use_sif:
            for token in tokens:
                if token not in self._sif_dict:
                    warnings.warn(
                        f"No frequency information for token: {token}. Its corresponding weight is 1.0.",
                        RuntimeWarning,
                    )
            res = np.sum(
                [
                    self._sif_dict[token] * self._get_default_vector(token)
                    for token in tokens
                ],
                axis=0,
            )
        else:
            res = self._get_default_vector(phrase)

        # In case the result is fishy it will return a vector of np.nans and raise a warning
        if np.isnan(res).any() or np.count_nonzero(res) == 0:
            warnings.warn(
                f"Unable to compute an embedding for phrase: {phrase}.", RuntimeWarning
            )
            a = np.empty((self.size_vectors,))
            a[:] = np.nan

            return a

        if self.normalize:
            return res / norm(res)
        else:
            return res

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        return self._embeddings_model._get_default_vector(phrase)

    # This will require refactoring for speed (in the case of spacy and USE)
    def get_vectors(self, phrases: str, progress_bar: bool = False) -> np.ndarray:

        if progress_bar:
            print("Computing phrase embeddings...")
            phrases = tqdm(phrases)

        vectors = []
        for i, phrase in enumerate(phrases):
            vector = self.get_vector(phrase)
            vectors.append(np.array([vector]))
        vectors = np.concatenate(vectors)
        return vectors

    @staticmethod
    def compute_sif_weights(sentences: List[str], alpha: float) -> Dict[str, float]:
        """
        A function that computes smooth inverse frequency (SIF) weights based on word frequencies.
        (See "Arora, S., Liang, Y., & Ma, T. (2016). A simple but tough-to-beat baseline for sentence embeddings.")
        The sentences are used to build the counter dictionary {"word": frequency} which is further used to compute the sif weights. If the word is not in the dictionary, 1 is returned.
        Args:
            sentences: a list of sentences
            alpha: regularization parameter
        Returns:
            A dictionary {"word": SIF weight}
        """
        words_counter = count_words(sentences)

        sif_dict = defaultdict(lambda: 1.0)

        for word, count in words_counter.items():
            sif_dict[word] = alpha / (alpha + count)

        return sif_dict


class spaCyEmbeddings(EmbeddingsBase):
    def __init__(self, model: str) -> None:
        if not spacy.util.is_package(model):
            spacy_download(model)
        self._nlp = spacy.load(
            model, disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        )

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
    def __init__(self, path: str):

        self._model = self._load_keyed_vectors(path)
        self._vocab = self._model.vocab
        self.size_vectors = self._model[list(self._vocab)[0]].shape[0]

    def _load_keyed_vectors(self, path):
        try:
            from gensim.models import Word2Vec

        except ModuleNotFoundError:
            print("Please install gensim package")
            raise

        return Word2Vec.load(path).wv

    def _get_default_vector(self, phrase: str) -> np.ndarray:

        tokens = phrase.split()
        embeddable_tokens = []
        for token in tokens:
            if token in self._vocab:
                embeddable_tokens.append(token)
            else:
                warnings.warn(
                    f"No vector for token: {token}. It is not used to compute the embedding of: {phrase}.",
                    RuntimeWarning,
                )
        res = np.mean([self._model[token] for token in embeddable_tokens], axis=0)
        return res


class GensimPreTrainedEmbeddings(GensimWord2VecEmbeddings, EmbeddingsBase):

    """
    A class to call a pre-trained embeddings model from gensim's library.
    # The list of pre-trained embeddings may be browsed by typing:
        import gensim.downloader as api
        list(api.info()['models'].keys())
    """

    def __init__(self, model: str):

        self._model = self._load_keyed_vectors(model)
        self._vocab = self._model.vocab

    def _load_keyed_vectors(self, model):
        try:
            import gensim.downloader as api

        except ModuleNotFoundError:
            print("Please install gensim package")
            raise

        return api.load(model)


class phraseBERTEmbeddings(EmbeddingsBase):
    """
    path = "whaleloops/phrase-bert"
    model = Embeddings("phrase-BERT", path)
    """

    def __init__(self, path: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer

        except ModuleNotFoundError:
            print("Please install sentence_transformers package")
            raise

        self._model = SentenceTransformer(path)

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        return self._model.encode(phrase)


def _compute_distances(vectors1, vectors2):
    """
    Compute pairwise distances of columns between two numpy arrays.
    """
    distances = cdist(vectors1, vectors2, metric="euclidean")
    return distances


def _get_min_distances(distances):
    """
    Returns the minimum distance per column.
    """
    return np.min(distances, axis=1)


def _get_index_min_distances(distances):
    """
    Returns the index of the minimum distance per column.
    """
    return np.argmin(distances, axis=1)


def _embeddings_similarity(vectors1, vectors2, threshold: float = 100):
    """
    Computes the pairwise distances between two numpy arrays,
    keeps minimum distances which are below the threshold and returns
    two arrays of indices:
    - index are the columns which satisfy the threshold requirement
    - index_min_distances are their associated index for the minimum distance
    """
    distances = _compute_distances(vectors1, vectors2)
    index_min_distances = _get_index_min_distances(distances)
    min_distances = _get_min_distances(distances)
    index = list(np.where(min_distances <= threshold))[0]
    index_min_distances = index_min_distances[index]
    return index, index_min_distances


def _remove_nan_vectors(vectors):
    """
    Remove columns with np.nan values in a numpy array.
    """
    return vectors[~np.isnan(vectors).any(axis=1)]
