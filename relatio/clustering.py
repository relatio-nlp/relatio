# Vectors and Clustering
# ..................................................................................................................
# ..................................................................................................................

import time
import warnings
from collections import Counter
from copy import deepcopy
from typing import List, Optional, Union

import gensim.downloader as api
import numpy as np
import tensorflow_hub as hub
from gensim.models import Word2Vec
from numpy.linalg import norm
from sklearn.cluster import KMeans
from tqdm import tqdm

from .utils import count_values, count_words


def compute_sif_weights(words_counter: dict, alpha: Optional[float] = 0.001) -> dict:

    """

    A function that computes smooth inverse frequency (SIF) weights based on word frequencies.
    (See "Arora, S., Liang, Y., & Ma, T. (2016). A simple but tough-to-beat baseline for sentence embeddings.")

    Args:
        words_counter: a dictionary {"word": frequency}
        alpha: regularization parameter

    Returns:
        A dictionary {"word": SIF weight}

    """

    sif_dict = {}

    for word, count in words_counter.items():
        sif_dict[word] = alpha / (alpha + count)

    return sif_dict


class USE:
    def __init__(self, path: str):
        self._embed = hub.load(path)

    def __call__(self, tokens: List[str]) -> np.ndarray:
        return self._embed([" ".join(tokens)]).numpy()[0]


class SIF_word2vec:
    def __init__(
        self,
        path: str,
        sentences=List[str],
        alpha: Optional[float] = 0.001,
        normalize: bool = True,
    ):

        self._model = self._load_keyed_vectors(path)

        self._words_counter = count_words(sentences)

        self._sif_dict = compute_sif_weights(self._words_counter, alpha)

        self._vocab = self._model.vocab

        self._normalize = normalize

    def _load_keyed_vectors(self, path):
        return Word2Vec.load(path).wv

    def __call__(self, tokens: List[str]):
        res = np.mean(
            [self._sif_dict[token] * self._model[token] for token in tokens], axis=0
        )
        if self._normalize:
            res = res / norm(res)
        return res

    def most_similar(self, v):
        return self._model.most_similar(positive=[v], topn=1)[0]


class SIF_keyed_vectors(SIF_word2vec):
    def _load_keyed_vectors(self, path):
        return api.load(path)


def get_vector(tokens: List[str], model: Union[USE, SIF_word2vec, SIF_keyed_vectors]):

    """

    A function that computes an embedding vector for a list of tokens.

    Args:
        tokens: list of string tokens to embed
        model: trained embedding model. It can be either:
         - Universal Sentence Encoders (USE)
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)

    Returns:
        A two-dimensional numpy array (1,dimension of the embedding space)

    """

    if not isinstance(model, (USE, SIF_word2vec, SIF_keyed_vectors)):
        raise TypeError("Union[USE, SIF_Word2Vec, SIF_keyed_vectors]")

    if isinstance(model, SIF_word2vec) or isinstance(model, SIF_keyed_vectors):
        if not tokens:
            res = None
        elif any(token not in model._sif_dict for token in tokens):
            res = None
        elif any(token not in model._vocab for token in tokens):
            res = None
        else:
            res = model(tokens)
            res = np.array(
                [res]
            )  # correct format to feed the vectors to sklearn clustering methods
    else:
        res = model(tokens)
        res = np.array(
            [res]
        )  # correct format to feed the vectors to sklearn clustering methods

    return res


def get_vectors(
    postproc_roles,
    model: Union[USE, SIF_word2vec, SIF_keyed_vectors],
    used_roles=List[str],
):

    """

    A function to train a K-Means model on the corpus.

    Args:
        postproc_roles: list of statements
        model: trained embedding model. It can be either:
         - Universal Sentence Encoders (USE)
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
        used_roles: list of semantic roles to cluster together

    Returns:
        A list of vectors

    """

    role_counts = count_values(postproc_roles, keys=used_roles)

    role_counts = [role.split() for role in list(role_counts)]

    vecs = []
    for role in role_counts:
        vec = get_vector(role, model)
        if vec is not None:
            vecs.append(vec)

    vecs = np.concatenate(vecs)

    return vecs


def train_cluster_model(
    vecs,
    model: Union[USE, SIF_word2vec, SIF_keyed_vectors],
    n_clusters,
    random_state: Optional[int] = 0,
    verbose: Optional[int] = 0,
):

    """

    Train a kmeans model on the corpus.

    Args:
        vecs: list of vectors
        model: trained embedding model. It can be either:
         - Universal Sentence Encoders (USE)
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
        random_state: seed for replication (default is 0)
        verbose: see Scikit-learn documentation for details

    Returns:
        A Scikit-learn kmeans model

    """

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, verbose=verbose
    ).fit(vecs)

    return kmeans


def get_clusters(
    postproc_roles: List[dict],
    model: Union[USE, SIF_word2vec, SIF_keyed_vectors],
    kmeans,
    used_roles=List[str],
    progress_bar: bool = False,
    suffix: str = "_lowdim",
) -> List[dict]:

    """

    Predict clusters based on a pre-trained kmeans model.

    Args:
        postproc_roles: list of statements
        model: trained embedding model
        (e.g. either Universal Sentence Encoders, a full gensim Word2Vec model or gensim Keyed Vectors)
        kmeans = a pre-trained sklearn kmeans model
        used_roles: list of semantic roles to consider
        progress_bar: print a progress bar (default is False)
        suffix: suffix for the new dimension-reduced role's name (e.g. 'ARGO_lowdim')

    Returns:
        A list of dictionaries with the predicted cluster for each role

    """

    roles_copy = deepcopy(postproc_roles)

    if progress_bar:
        print("Assigning clusters to roles...")
        time.sleep(1)
        postproc_roles = tqdm(postproc_roles)

    for i, statement in enumerate(postproc_roles):
        for role, tokens in statement.items():
            if role in used_roles:
                vec = get_vector(tokens.split(), model)
                if vec is not None:
                    clu = kmeans.predict(vec)[0]
                    roles_copy[i][role] = clu
                else:
                    roles_copy[i].pop(role, None)
            else:
                roles_copy[i].pop(role, None)

    roles_copy = [
        {str(k + suffix): v for k, v in statement.items()} for statement in roles_copy
    ]

    return roles_copy


def label_clusters_most_freq(
    clustering_res: List[dict], postproc_roles: List[dict]
) -> dict:

    """

    A function which labels clusters by their most frequent term.

    Args:
        clustering_res: list of dictionaries with the predicted cluster for each role
        postproc_roles: list of statements

    Returns:
        A dictionary associating to each cluster number a label (e.g. the most frequent term in this cluster)

    """

    temp = {}
    labels = {}

    for i, statement in enumerate(clustering_res):
        for role, cluster in statement.items():
            tokens = postproc_roles[i][role]
            cluster_num = cluster
            if cluster_num not in temp:
                temp[cluster_num] = [tokens]
            else:
                temp[cluster_num].append(tokens)

    for cluster_num, tokens in temp.items():
        token_most_common = Counter(tokens).most_common(2)
        if len(token_most_common) > 1 and (
            token_most_common[0][1] == token_most_common[1][1]
        ):
            warnings.warn(
                f"Multiple labels for cluster {cluster_num}- 2 shown: {token_most_common}. First one is picked.",
                RuntimeWarning,
            )
        labels[cluster_num] = token_most_common[0][0]

    return labels


def label_clusters_most_similar(kmeans, model) -> dict:

    """

    A function which labels clusters by the term closest to the centroid in the embedding
    (i.e. distance is cosine similarity)

    Args:
        kmeans: the trained kmeans model
        model: trained embedding model. It can be either:
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)

    Returns:
        A dictionary associating to each cluster number a label
        (e.g. the most similar term to cluster's centroid)

    """

    labels = {}

    for i, vec in enumerate(kmeans.cluster_centers_):
        most_similar_term = model.most_similar(vec)
        labels[i] = most_similar_term[0]

    return labels
