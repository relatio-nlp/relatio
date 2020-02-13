import itertools
import random
from typing import List, Dict

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
    vectors: List[List[Dict[str, np.ndarray]]],
    size_sample: float,
    num_clusters: int,
    roles: List[str] = ["ARGO", "B-V"],
    seed: int = 1234,
):
    if size_sample < 0 or size_sample > 1:
        raise ValueError("size_sample should be beteen 0 and 1")

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

        kmeans_output_dict["kmeans"] = KMeans(n_clusters=num_clusters).fit(A)
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

    return lst
