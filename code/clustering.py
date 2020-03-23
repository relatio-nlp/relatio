import random
from typing import List, Dict, Union, Any

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.utils import resample

from utils import UsedRoles
from word_embedding import SIF_Word2Vec


class Clustering:
    def __init__(self, cluster, n_clusters, used_roles: UsedRoles, sample_seed=123):
        self._embed_roles = used_roles.embeddable
        if not isinstance(cluster, dict):
            self._cluster = {el: clone(cluster) for el in self._embed_roles}
        else:
            self.cluster = cluster
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

    def label_most_similar_in_w2v(self, word2vec: SIF_Word2Vec):
        labels = {}
        for el in self._embed_roles:
            labels[el] = {}
            for i, vec in enumerate(self._cluster[el].cluster_centers_):
                labels[el][i] = word2vec.most_similar(vec)
        return labels

    def normalise_centroids(self):
        # TODO
        pass

    def compute_distance(self, distance, predicted, centroids_like):
        # TODO
        # for each predicted get the label and pick from centroids_like the one with the same label and apply distance
        pass


def assign_cluster(centroids, points):
    num_centroids, dim = centroids.shape
    num_points, _ = points.shape
    # Tile and reshape both arrays into `[num_points, num_centroids, dim]`.
    centroids = np.tile(centroids, [num_points, 1]).reshape(
        [num_points, num_centroids, dim]
    )
    points = np.tile(points, [1, num_centroids]).reshape(
        [num_points, num_centroids, dim]
    )
    # Compute all distances and select the closest centroid.
    distances = np.sum(np.square(centroids - points), axis=2)
    assigned_cluster = np.argmin(distances, axis=1)
    return assigned_cluster


## OLD CODE


def compute_kmeans_clusters(
    clustering,
    vectors: List[Dict[str, np.ndarray]],
    size_sample: float,
    roles_num_clusters: Dict["str", int],
    roles: List[str] = ["ARGO", "B-V"],
    seed: int = 1234,
):
    if size_sample < 0 or size_sample > 1:
        raise ValueError("size_sample should be between 0 and 1")

    kmeans_dict = {}
    for role in roles:
        kmeans_output_dict = {}
        temp_dict = {}

        i = 0
        for sentence in vectors:
            for statement in sentence:
                if statement.get(role) is not None:
                    temp_dict[i] = statement[role]
                    i = i + 1
        if not temp_dict:
            continue
        random.seed(seed)
        keys = random.sample(list(temp_dict.keys()), int(size_sample * len(temp_dict)))
        sample = {k: temp_dict[k] for k in keys}

        del temp_dict

        A = pd.DataFrame.from_dict(sample, orient="index")

        kmeans_output_dict["kmeans"] = KMeans(n_clusters=roles_num_clusters[role]).fit(
            A
        )
        kmeans_output_dict["centroids"] = kmeans_output_dict[
            "kmeans"
        ].cluster_centers_.astype(np.float32)

        kmeans_dict[role] = kmeans_output_dict

    lst = []
    for sentence in vectors:
        sentence_list = []
        for statement in sentence:
            statement_dict = {}
            for role, vector in statement.items():
                if role in roles:
                    statement_dict[role] = assign_cluster(
                        kmeans_dict[role]["centroids"],
                        np.expand_dims(statement[role], 0),
                    )[0]
            sentence_list.append(statement_dict)
        if not sentence_list:
            sentence_list = [{}]
        lst.append(sentence_list)

    return kmeans_dict, lst


def compute_cluster_labels(
    model: Union[str, Word2Vec], kmeans_dict: Dict[str, Dict[str, Any]], topn: int = 1
):
    if isinstance(model, str):
        model = Word2Vec.load(model)
    elif isinstance(model, str):
        pass
    else:
        raise TypeError("model is either the a string or an Word2Vec object")
    res = {}
    for role, kmeans_output_dict in kmeans_dict.items():
        kmeans = kmeans_output_dict["kmeans"]
        centroids = kmeans_output_dict["centroids"]
        labels = {}
        num_clusters = len(set(kmeans.labels_))
        for num_cluster in range(num_clusters):
            top_closest_words = model.wv.most_similar(
                positive=[centroids[num_cluster]], topn=topn
            )
            top_closest_words = [word[0] for word in top_closest_words]
            top_closest_words = ", ".join(top_closest_words)
            labels[num_cluster] = top_closest_words
        res[role] = labels
    return res
