import random
from typing import List, Dict, Union, Any

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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


def compute_kmeans_clusters(
    clustering,
    vectors: List[Dict[str, np.ndarray]],
    sample_size: float,
    role_n_clusters: Dict["str", int],
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
