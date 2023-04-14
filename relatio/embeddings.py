import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import gensim.downloader as api
import numpy as np
import spacy
from gensim.models import Word2Vec
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from spacy.cli import download as spacy_download
from tqdm import tqdm

from relatio.supported_models import LANGUAGE_MODELS
from relatio.utils import count_words


class EmbeddingsBase(ABC):
    @abstractmethod
    def _get_default_vector(self, phrase: str) -> np.ndarray:
        pass


class Embeddings(EmbeddingsBase):
    """
    If sentences is used in the constructor the embeddings are weighted by the smoothed inverse frequency of each token.
    For further details, see: https://github.com/PrincetonML/SIF

    Args:
        embeddings_type: The type of embeddings to use. Supported types are: "SentenceTransformer", "GensimWord2Vec", "GensimPretrained", "spaCy"
        embeddings_model: The model to use. Supported models are: "all-MiniLM-L6-v2", "distiluse-base-multilingual-cased-v2", "whaleloops/phrase-bert", "fasttext-wiki-news-subwords-300", "word2vec-google-news-300", "glove-wiki-gigaword-50", "glove-wiki-gigaword-100", "glove-wiki-gigaword-200", "glove-wiki-gigaword-300", "glove-twitter-25", "glove-twitter-50", "glove-twitter-100", "glove-twitter-200", "en_core_web_sm", "en_core_web_md", "en_core_web_lg", "fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"
        normalize: Whether to normalize the vectors to unit length
        sentences: A list of sentences to use for weighting the embeddings by the smoothed inverse frequency of each token
        alpha: The smoothing parameter for the smoothed inverse frequency of each token

    Examples:
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
        >>> model = Embeddings("GensimPretrained", "glove-twitter-25")
        >>> model.get_vector("world").shape
        (25,)
        >>> model = Embeddings("GensimPretrained", "glove-twitter-25", sentences = ["this is a nice world","hello world","hello everybody"])
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
            Type[GensimWord2VecEmbeddings],
            Type[GensimPreTrainedEmbeddings],
            Type[spaCyEmbeddings],
            Type[SentenceTransformerEmbeddings],
        ]
        if embeddings_type == "SentenceTransformer":
            EmbeddingsClass = SentenceTransformerEmbeddings
        elif embeddings_type == "GensimWord2Vec":
            EmbeddingsClass = GensimWord2VecEmbeddings
        elif embeddings_type == "GensimPretrained":
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

        if embeddings_type != "GensimWord2Vec":
            self._size_vectors = LANGUAGE_MODELS[embeddings_model]["size_vectors"]
        else:
            self._size_vectors = self._embeddings_model.size_vectors

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def use_sif(self) -> bool:
        return self._use_sif

    @property
    def size_vectors(self) -> int:
        return self._size_vectors

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

        vectors_list = []
        for i, phrase in enumerate(phrases):
            vector = self.get_vector(phrase)
            vectors_list.append(np.array([vector]))
        vectors = np.concatenate(vectors_list)
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


class SentenceTransformerEmbeddings(EmbeddingsBase):
    """
    Choose your favorite model from https://www.sbert.net/docs/pretrained_models.html

    Args:
        path: path to the model
    """

    def __init__(self, path: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(path)

    def _get_default_vector(self, phrase: str) -> np.ndarray:
        return self._model.encode(phrase)


class GensimWord2VecEmbeddings(EmbeddingsBase):
    def __init__(self, path: str):
        self._model = self._load_keyed_vectors(path)
        self._vocab = self._model.key_to_index
        self.size_vectors = self._model[list(self._vocab)[0]].shape[0]

    def _load_keyed_vectors(self, path):
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
        self._vocab = self._model.key_to_index

    def _load_keyed_vectors(self, model):
        return api.load(model)


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
