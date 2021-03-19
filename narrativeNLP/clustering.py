import warnings
from collections import Counter
from itertools import groupby
from typing import Dict, Union

import numpy as np
from gensim.models import Word2Vec
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.utils import resample

from .utils import UsedRoles
from .word_embedding import SIF_Word2Vec


class Clustering:
    def __init__(self, cluster, n_clusters, used_roles: UsedRoles, sample_seed=123):
        self._embed_roles = used_roles.embeddable
        if not isinstance(cluster, dict):
            self._cluster = {el: clone(cluster) for el in self._embed_roles}
        else:
            self._cluster = cluster
        if not isinstance(n_clusters, dict):
            self._n_clusters = {el: n_clusters for el in self._embed_roles}
        else:
            self._n_clusters = n_clusters

        self._dtype = {}
        for el, value in self._n_clusters.items():
            self._dtype[el] = np.uint8
            if value > np.iinfo(np.uint8).max:
                self._dtype[el] = np.uint16
            elif value > np.iinfo(np.uint16).max:
                self._dtype[el] = np.uint32
            elif value > np.iinfo(np.uint32).max:
                raise ValueError(f"m_clusters for {el} > {np.iinfo(np.uint32).max}")

        for el in self._embed_roles:
            self._cluster[el].n_clusters = self._n_clusters[el]

    def resample(
        self,
        vectors,
        sample_size: Union[int, float, Dict[str, Union[int, float]]] = 1,
        random_state: Union[int, Dict[str, int]] = 0,
    ):
        if not isinstance(random_state, dict):
            random_state = {el: random_state for el in self._embed_roles}
        else:
            random_state = random_state

        if not isinstance(sample_size, dict):
            sample_size = {el: sample_size for el in self._embed_roles}
        else:
            sample_size = sample_size

        sample_vectors = {}
        for el in self._embed_roles:
            if sample_size[el] in [1, 1.0]:
                sample_vectors[el] = vectors[el]
            else:
                _size = vectors[el].shape[0]
                _sample_size = sample_size[el]
                _n_samples = (
                    _sample_size
                    if isinstance(_sample_size, int)
                    else int(_size * _sample_size)
                )
                sample_vectors[el] = resample(
                    vectors[el],
                    n_samples=_n_samples,
                    replace=False,
                    random_state=random_state[el],
                )
        return sample_vectors

    def __getitem__(self, role_name):
        return self._cluster[role_name]

    def fit(self, vectors):
        for el in self._embed_roles:
            self._cluster[el] = self._cluster[el].fit(vectors[el])

    def predict(self, vectors):
        res = {}
        for el in self._embed_roles:
            res[el] = np.asarray(
                self._cluster[el].predict(vectors[el]), dtype=self._dtype[el]
            )
        return res

    def compute_distance(self, vectors, predicted_cluster):
        res = {}
        for el in self._embed_roles:
            res[el] = self._cluster[el].transform(vectors[el])[
                np.arange(predicted_cluster[el].size), predicted_cluster[el]
            ]
        return res

    def distance_mask(self, distance, threshold: Union[float, Dict[str, float]] = 2.0):
        if not isinstance(threshold, dict):
            threshold = {el: threshold for el in self._embed_roles}
        else:
            threshold = threshold

        res = {}
        for el in self._embed_roles:
            res[el] = distance[el] <= threshold[el]
        return res

    def label_most_similar_in_w2v(self, word2vec: SIF_Word2Vec):
        labels = {}
        for el in self._embed_roles:
            labels[el] = {}
            for i, vec in enumerate(self._cluster[el].cluster_centers_):
                labels[el][i] = list(word2vec.most_similar(vec))
        return labels

    def normalise_centroids(self):
        # TODO
        pass


def label_clusters(
    *,
    clustering_res,
    distance,
    postproc_roles,
    statement_index,
    top: int = 1,
    drop_duplicates: bool = True,
):
    labels = {}
    for role, clustering in clustering_res.items():
        labels[role] = {}
        for cluster_id in np.unique(clustering):
            dist = np.ma.MaskedArray(distance[role], clustering != cluster_id)
            argsorts = dist.argsort()[:top]
            labels[role][cluster_id] = [
                (
                    "_".join(postproc_roles[statement_index[role][i]][role]),
                    dist[i],
                )
                for i in argsorts
            ]
            if top == 1:
                labels[role][cluster_id] = list(labels[role][cluster_id][0])
            elif drop_duplicates:
                labels[role][cluster_id] = sorted(
                    list(set(labels[role][cluster_id])), key=lambda x: x[1]
                )
    return labels


def label_clusters_most_freq(
    *,
    clustering_res,
    postproc_roles,
    statement_index,
    clustering_mask=True,
):
    labels = {}
    for role, clustering in clustering_res.items():
        labels[role] = {}
        grouped_data = groupby(
            sorted(
                (
                    (
                        int(value),
                        "_".join(postproc_roles[statement_index[role][i]][role]),
                    )
                    for i, value in enumerate(clustering)
                    if clustering_mask is True or clustering_mask[role][i]
                ),
                key=lambda x: x[0],
            ),
            key=lambda x: x[0],
        )

        labels[role] = {
            k: Counter(el[1] for el in ngrams).most_common(2)
            for k, ngrams in grouped_data
        }

        for k, v in labels[role].items():
            if len(v) > 1 and (v[0][1] == v[1][1]):
                warnings.warn(
                    f"Multiple labels - 2 shown: \n  labels[{role}][{k}]={v}. First one is picked.",
                    RuntimeWarning,
                )
            labels[role][k] = list(v[0])
    return labels
