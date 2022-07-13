import math
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from typing import List, Optional, Type

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import spacy
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.model_selection import RandomizedSearchCV
from spacy.cli import download as spacy_download
from tqdm import tqdm
from wpca import WPCA

from relatio.embeddings import (
    Embeddings,
    _compute_distances,
    _embeddings_similarity,
    _get_index_min_distances,
    _get_min_distances,
    _remove_nan_vectors,
)
from relatio.utils import count_values, is_subsequence, make_list_from_key, prettify


class NarrativeModel:
    """
    A general class to build a model that extracts latent narratives from a list of SRL statements.
    """

    def __init__(
        self,
        model_type="hdbscan",
        PCA_type="PCA",
        roles_considered: List[str] = [
            "ARG0",
            "B-V",
            "B-ARGM-NEG",
            "B-ARGM-MOD",
            "ARG1",
            "ARG2",
        ],
        roles_with_known_entities: str = ["ARG0", "ARG1", "ARG2"],
        known_entities: Optional[List[str]] = None,
        assignment_to_known_entities: str = "character_matching",
        roles_with_unknown_entities: List[str] = ["ARG0", "ARG1", "ARG2"],
        embeddings_model: Optional[Type[Embeddings]] = None,
        threshold: int = 0.1,
    ):

        if model_type not in ["deterministic", "kmeans", "hdbscan"]:
            raise ValueError(
                "Only three options for model_type: deterministic, kmeans, or hdbscan."
            )

        if PCA_type not in ["PCA", "IncrementalPCA", "WPCA"]:
            raise ValueError(
                "Only three options for model_type: PCA, IncrementalPCA, or WPCA."
            )

        if (
            is_subsequence(
                roles_considered,
                ["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
            )
            is False
        ):
            raise ValueError(
                "Some roles_considered are not supported. Roles supported: ARG0, B-V, B-ARGM-NEG, B-ARGM-MOD, ARG1, ARG2"
            )

        if roles_with_known_entities is not None:
            if is_subsequence(roles_with_known_entities, roles_considered) is False:
                raise ValueError(
                    "roles_with_known_entities should be in roles_considered."
                )

        if roles_with_unknown_entities is not None:
            if is_subsequence(roles_with_unknown_entities, roles_considered) is False:
                raise ValueError(
                    "roles_with_unknown_entities should be a subset of roles_considered."
                )
            if ["B-ARGM-NEG", "B-ARGM-MOD", "B-V"] in roles_with_unknown_entities:
                raise ValueError(
                    "Negations, verbs and modals cannot be embedded and clustered."
                )

        if assignment_to_known_entities not in ["character_matching", "embeddings"]:
            raise ValueError(
                "Only two options for assignment_to_known_entities: character_matching or embeddings."
            )

        self.model_type = model_type
        self.PCA_type = PCA_type
        self.roles_considered = roles_considered
        self.roles_with_unknown_entities = roles_with_unknown_entities
        self.roles_with_known_entities = roles_with_known_entities
        self.known_entities = known_entities
        self.vectors_known_entities = None
        self.assignment_to_known_entities = assignment_to_known_entities
        self.threshold = threshold

        if embeddings_model is None:
            self.embeddings_model = Embeddings(
                "TensorFlow_USE",
                "https://tfhub.dev/google/universal-sentence-encoder/4",
            )
        else:
            self.embeddings_model = embeddings_model

        if (
            self.known_entities is not None
            and self.assignment_to_known_entities == "embeddings"
        ):
            self.vectors_known_entities = self.embeddings_model.get_vectors(
                self.known_entities
            )

        self.pca_args = []
        self.umap_args = []
        self.cluster_args = []
        self.scores = []
        self.vectors_unknown_entities = []
        self.labels_unknown_entities = {}
        self.vocab_unknown_entities = {}
        self.clustering_model = []
        self.training_vectors = []
        self.unique_vectors = []
        self.phrases_to_embed = []

    def fit(
        self,
        srl_res,
        pca_args=None,
        umap_args=None,
        cluster_args=None,
        progress_bar=True,
        weight_by_frequency=False,
    ):

        if self.model_type == "deterministic":
            print("No fitting required, this model is deterministic!")
        if self.model_type in ["hdbscan", "kmeans"]:
            self.fit_static_clustering(
                srl_res,
                pca_args,
                umap_args,
                cluster_args,
                progress_bar,
                weight_by_frequency,
            )
        if self.model_type == "dynamic":
            pass

    def fit_static_clustering(
        self,
        srl_res,
        pca_args,
        umap_args,
        cluster_args,
        progress_bar,
        weight_by_frequency,
    ):

        phrases_to_embed = []
        counter_for_phrases = Counter()

        for role in self.roles_with_unknown_entities:

            temp_counter = count_values(srl_res, keys=[role])
            counter_for_phrases = counter_for_phrases + temp_counter
            phrases = list(temp_counter)

            # remove known entities for the training of unknown entities
            if role in self.roles_with_known_entities:
                if self.assignment_to_known_entities == "character_matching":
                    idx = self.character_matching(phrases, progress_bar)[0]
                elif self.assignment_to_known_entities == "embeddings":
                    vectors = self.embeddings_model.get_vectors(phrases, progress_bar)
                    idx = _embeddings_similarity(
                        vectors, self.vectors_known_entities, self.threshold
                    )[0]

                phrases = [phrase for l, phrase in enumerate(phrases) if l not in idx]

                phrases_to_embed.extend(phrases)

        if weight_by_frequency:
            self.phrases_to_embed = phrases_to_embed
        else:
            phrases_to_embed = sorted(list(set(phrases_to_embed)))
            self.phrases_to_embed = phrases_to_embed

        # Remove np.nans to train the model (or it will break down)
        vectors = self.embeddings_model.get_vectors(phrases_to_embed, progress_bar)
        self.training_vectors = _remove_nan_vectors(vectors)

        # Dimension reduction via PCA + UMAP

        if pca_args is None:
            if self.PCA_type == "PCA":
                pca_args = {"n_components": 50, "svd_solver": "full"}
            elif self.PCA_type == "WPCA":
                pca_args = {"n_components": 50}
            elif self.PCA_type == "IncrementalPCA":
                pca_args = {"n_components": 50, "batch_size": None}

        if umap_args is None:

            umap_args = {"n_neighbors": 15, "n_components": 2, "random_state": 0}

        if progress_bar:
            print("Dimension reduction via PCA + UMAP...")
            print("PCA parameters:")
            print(pca_args)
            print("UMAP parameters:")
            print(umap_args)

        self.pca_args = pca_args
        if self.PCA_type == "PCA":
            self.pca = PCA(**pca_args).fit(self.training_vectors)
        elif self.PCA_type == "WPCA":
            self.pca = WPCA(**pca_args).fit(self.training_vectors)
        elif self.PCA_type == "IncrementalPCA":
            self.pca = IncrementalPCA(**pca_args).fit(self.training_vectors)
        self.pca_embeddings = self.pca.transform(self.training_vectors)

        self.umap_args = umap_args
        self.umap = umap.UMAP(**umap_args).fit(self.pca_embeddings)
        self.umap_embeddings = self.umap.transform(self.pca_embeddings)

        # Clustering
        if progress_bar:
            print("Clustering phrases into clusters...")

        if self.model_type == "kmeans":

            if cluster_args is None:

                l = max(int(len(phrases_to_embed) / 100), 1)
                l = min(l, 1000)
                q0 = max(int(np.quantile(list(range(l)), 0.1)), 1)
                q1 = max(int(np.quantile(list(range(l)), 0.25)), 1)
                q2 = max(int(np.quantile(list(range(l)), 0.5)), 1)
                q3 = max(int(np.quantile(list(range(l)), 0.75)), 1)

                cluster_args = {"n_clusters": [q0, q1, q2, q3, l], "random_state": 0}

            # Grid search
            models = []
            for num_clusters in cluster_args["n_clusters"]:
                args = {}
                args["n_clusters"] = num_clusters

                for k, v in cluster_args.items():
                    if k not in ["n_clusters"]:
                        args[k] = v

                kmeans = KMeans(**args).fit(self.umap_embeddings)

                models.append(kmeans)

            scores = []

            for model in models:
                scores.append(
                    silhouette_score(
                        self.umap_embeddings,
                        model.labels_,
                        random_state=cluster_args["random_state"],
                    )
                )

        if self.model_type == "hdbscan":

            if cluster_args is None:

                l = max(int(math.sqrt(len(phrases_to_embed))), 2)
                l = min(l, 100)
                q1 = max(int(np.quantile(list(range(l)), 0.25)), 2)
                q2 = max(int(np.quantile(list(range(l)), 0.5)), 2)
                q3 = max(int(np.quantile(list(range(l)), 0.75)), 2)

                cluster_args = {
                    "min_cluster_size": [q1, q2, q3, l],
                    "min_samples": [1, 10, 20],
                    "cluster_selection_method": ["eom"],
                    "gen_min_span_tree": True,
                    "approx_min_span_tree": False,
                    "prediction_data": True,
                }

            # Grid search
            models = []
            scores = []
            for i in cluster_args["min_cluster_size"]:
                for j in cluster_args["min_samples"]:
                    for h in cluster_args["cluster_selection_method"]:
                        args = {}
                        args["min_cluster_size"] = i
                        args["min_samples"] = j
                        args["cluster_selection_method"] = h

                        for k, v in cluster_args.items():
                            if k not in [
                                "min_cluster_size",
                                "min_samples",
                                "cluster_selection_method",
                            ]:
                                args[k] = v

                        hdb = hdbscan.HDBSCAN(**args).fit(self.umap_embeddings)

                        models.append(hdb)

                        score = hdbscan.validity.validity_index(
                            self.umap_embeddings.astype(np.float64), hdb.labels_
                        )
                        scores.append(score)

        self.clustering_model = models[np.argmax(scores)]
        self.cluster_args = cluster_args
        self.scores = scores

        if progress_bar:
            print("Clustering parameters chosen in this range:")
            print(cluster_args)

        self.label_clusters(counter_for_phrases, phrases_to_embed, progress_bar)

        if self.model_type == "kmeans":
            self.vectors_unknown_entities = self.clustering_model.cluster_centers_

    def predict(self, srl_res, progress_bar: bool = False):
        """
        Predict the narratives underlying SRL statements.
        """

        narratives = deepcopy(srl_res)

        for role in self.roles_considered:

            if role in ["B-ARGM-NEG", "B-ARGM-MOD", "B-V"]:
                continue

            if progress_bar:
                print("\nPredicting entities for role: %s..." % role)

            flag_computed_vectors = False
            index1, phrases = make_list_from_key(role, srl_res)
            index2 = []
            index3 = []

            # Match known entities (with character matching)
            if (
                role in self.roles_with_known_entities
                and self.assignment_to_known_entities == "character_matching"
            ):
                index2, labels_known_entities = self.character_matching(
                    phrases, progress_bar
                )

            # Match known entities (with embeddings distance)
            if (
                role in self.roles_with_known_entities
                and self.assignment_to_known_entities == "embeddings"
            ):
                vectors = self.embeddings_model.get_vectors(phrases, progress_bar)

                if progress_bar:
                    print("Matching known entities (with embeddings distance)...")

                index2, index_known_entities = _embeddings_similarity(
                    vectors, self.vectors_known_entities, self.threshold
                )
                labels_known_entities = self.label_with_known_entity(
                    index_known_entities
                )
                flag_computed_vectors = True

            # Predict unknown entities (with clustering model)
            if role in self.roles_with_unknown_entities:

                if progress_bar:
                    print("Matching unknown entities (with clustering model)...")

                if flag_computed_vectors == False:
                    vectors = self.embeddings_model.get_vectors(phrases, progress_bar)

                if progress_bar:
                    print("Dimension reduction of vectors (PCA + UMAP)...")

                pca_embeddings = self.pca.transform(vectors)

                umap_embeddings = self.umap.transform(pca_embeddings)

                if progress_bar:
                    print("Assignment to clusters...")

                if self.model_type == "hdbscan":
                    index_clusters = hdbscan.approximate_predict(
                        self.clustering_model, umap_embeddings
                    )[0]
                    index3 = list(range(len(index_clusters)))

                else:

                    index3, index_clusters = _embeddings_similarity(
                        umap_embeddings, self.vectors_unknown_entities
                    )

                cluster_labels = self.label_with_most_frequent_phrase(index_clusters)

            # Assign labels
            if progress_bar:
                print("Assigning labels to matches...")

            all_labels = ["" for i in phrases]
            for i, k in enumerate(index2):
                all_labels[k] = labels_known_entities[i]
            for i, k in enumerate(index3):
                if all_labels[k] == "":
                    all_labels[k] = cluster_labels[i]
            for i, k in enumerate(phrases):
                if all_labels[i] != "":
                    narratives[index1[i]][role] = all_labels[i]
                elif role in narratives[index1[i]]:
                    narratives[index1[i]].pop(role)

        return narratives

    def character_matching(self, phrases, progress_bar: bool = False):

        if progress_bar:
            print("Matching known entities (with character matching)...")
            phrases = tqdm(phrases)

        labels_known_entities = []
        index = []
        for i, phrase in enumerate(phrases):
            matched_entities = []
            for entity in self.known_entities:
                if is_subsequence(entity.split(), phrase.split()):
                    matched_entities.append(entity)
            if len(matched_entities) != 0:
                matched_entities = "|".join(matched_entities)
                labels_known_entities.append(matched_entities)
                index.append(i)

        return index, labels_known_entities

    def label_clusters(self, counter_for_phrases, phrases_to_embed, progress_bar):

        if progress_bar:
            print("Labeling the clusters by the most frequent phrases...")

        labels = list(set(self.clustering_model.labels_))

        for clu in labels:
            self.vocab_unknown_entities[clu] = Counter()

        for j, clu in enumerate(self.clustering_model.labels_):
            self.vocab_unknown_entities[clu][phrases_to_embed[j]] = counter_for_phrases[
                phrases_to_embed[j]
            ]

        for clu in labels:
            token_most_common = self.vocab_unknown_entities[clu].most_common(2)
            if len(token_most_common) > 1 and (
                token_most_common[0][1] == token_most_common[1][1]
            ):
                warnings.warn(
                    f"Multiple labels for cluster {clu}- 2 shown: {token_most_common}. First one is picked.",
                    RuntimeWarning,
                )
            self.labels_unknown_entities[clu] = token_most_common[0][0]

        if self.model_type == "hdbscan":
            self.labels_unknown_entities[-1] = ""

    def label_with_known_entity(self, index):
        return [self.known_entities[i] for i in index]

    def label_with_most_frequent_phrase(self, index):
        return [self.labels_unknown_entities[i] for i in index]

    def inspect_cluster(self, label, topn=10):
        key = [k for k, v in self.labels_unknown_entities.items() if v == label][0]
        return self.vocab_unknown_entities[key].most_common(topn)

    def clusters_to_txt(self, path="clusters.txt", topn=10, add_frequency_info=True):

        with open(path, "w") as f:
            for k, v in self.vocab_unknown_entities.items():
                f.write("Cluster %s" % k)
                f.write("\n")
                for i in v.most_common(topn):
                    if add_frequency_info == True:
                        f.write("%s (%s), " % (i[0], i[1]))
                    else:
                        f.write("%s, " % i[0])
                f.write("\n")
                f.write("\n")

    def plot_clusters(self, path=None, figsize=(14, 8), s=0.1):

        if self.umap_args["n_components"] != 2:
            raise ValueError(
                "Cluster visualization is only possible for UMAP with two components."
            )

        clustered = self.clustering_model.labels_ >= 0
        plt.figure(figsize=figsize, dpi=80)
        plt.scatter(
            self.umap_embeddings[~clustered, 0],
            self.umap_embeddings[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=s,
            alpha=0.5,
        )
        plt.scatter(
            self.umap_embeddings[clustered, 0],
            self.umap_embeddings[clustered, 1],
            c=self.clustering_model.labels_[clustered],
            s=s,
            cmap="Spectral",
        )
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    def plot_selection_metric(self, path=None, figsize=(14, 8)):

        if self.model_type == "hdbscan":
            raise ValueError(
                "Plotting the selection metric is only possible for a kmeans model."
            )

        if self.model_type == "kmeans":
            plt.figure(figsize=figsize)
            plt.plot(self.cluster_args["n_clusters"], self.scores, "bx-")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Score")

        if path is None:
            plt.show()
        else:
            plt.savefig(path)
