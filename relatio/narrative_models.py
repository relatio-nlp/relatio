import math
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from relatio.embeddings import Embeddings, _embeddings_similarity
from relatio.utils import count_values, is_subsequence, make_list_from_key


class NarrativeModel:
    def __init__(
        self,
        clustering: Optional[str] = "kmeans",
        PCA: bool = True,
        UMAP: bool = True,
        roles_considered: List[str] = [
            "ARG0",
            "B-V",
            "B-ARGM-NEG",
            "B-ARGM-MOD",
            "ARG1",
            "ARG2",
        ],
        roles_with_known_entities: List[str] = ["ARG0", "ARG1", "ARG2"],
        known_entities: Optional[List[str]] = None,
        assignment_to_known_entities: str = "character_matching",
        roles_with_unknown_entities: List[str] = ["ARG0", "ARG1", "ARG2"],
        embeddings_type: str = "SentenceTransformer",
        embeddings_model: Union[Path, str] = "all-MiniLM-L6-v2",
        threshold: float = 0.1,
    ):
        """
        A general class to build a model that extracts latent narratives from a list of SRL statements.

        Args:
            clustering (Optional[str], optional): The clustering algorithm to use. Defaults to "kmeans".
            PCA (bool, optional): Whether to perform PCA on the embeddings. Defaults to True.
            UMAP (bool, optional): Whether to perform UMAP on the embeddings. Defaults to True.
            roles_considered (List[str], optional): The semantic roles to consider. Defaults to ["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"].
            roles_with_known_entities (List[str], optional): The semantic roles that have known entities. Defaults to ["ARG0", "ARG1", "ARG2"].
            known_entities (Optional[List[str]], optional): The known entities (can be obtained via named entity recognition, for instance). Defaults to None.
            assignment_to_known_entities (str, optional): The method to assign the known entities to the roles. The methods are either "character_matching" or "embeddings". "character_matching" matches the exact phrase. "embeddings" assigns to the entity any phrase with a high cosine similarity (see threshold). Defaults to "character_matching".
            roles_with_unknown_entities (List[str], optional): The semantic roles that have unknown entities. Defaults to ["ARG0", "ARG1", "ARG2"].
            embeddings_type (str, optional): The type of embeddings to use. Defaults to "SentenceTransformer" (corresponds to Universal Sentence Encoder).
            embeddings_model (Union[Path, str], optional): The path to the embeddings model. Defaults to "all-MiniLM-L6-v2".
            threshold (float, optional): The threshold for the cosine similarity between the embeddings of the known entities and the embeddings of the phrases. Defaults to 0.1.
        """

        if clustering is not None:
            if clustering not in ["kmeans", "hdbscan"]:
                raise ValueError(
                    "Only three options for clustering: None, kmeans, or hdbscan."
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

        self.clustering = clustering
        self.PCA = PCA
        self.UMAP = UMAP
        self.roles_considered = roles_considered
        self.roles_with_unknown_entities = roles_with_unknown_entities
        self.roles_with_known_entities = roles_with_known_entities
        self.known_entities = known_entities
        self.vectors_known_entities = None
        self.assignment_to_known_entities = assignment_to_known_entities
        self.threshold = threshold

        self.embeddings_model = Embeddings(
            embeddings_type=embeddings_type, embeddings_model=embeddings_model
        )

        if (
            self.known_entities is not None
            and self.assignment_to_known_entities == "embeddings"
        ):
            self.vectors_known_entities = self.embeddings_model.get_vectors(
                self.known_entities
            )

        self.pca_args = {}
        self.umap_args = {}
        self.cluster_args = {}
        self.training_vectors = []
        self.phrases_to_embed = []
        self.scores = {}
        self.labels_unknown_entities = []
        self.vocab_unknown_entities = []
        self.clustering_models = []
        self.index_optimal_model = None

    def fit(
        self,
        srl_res,
        pca_args={"n_components": 50, "svd_solver": "full"},
        umap_args={"n_neighbors": 15, "n_components": 2, "random_state": 0},
        cluster_args=None,
        weight_by_frequency=False,
        progress_bar=True,
    ):
        """
        Fits the model to the SRL statements.

        Args:
            srl_res (List[Dict]): The SRL statements.
            pca_args (Dict, optional): The arguments for the PCA. Defaults to {"n_components": 50, "svd_solver": "full"}.
            umap_args (Dict, optional): The arguments for the UMAP. Defaults to {"n_neighbors": 15, "n_components": 2, "random_state": 0}.
            cluster_args (Dict, optional): The arguments for the clustering algorithm: cluster_args = {"n_clusters": [k1, k2, k3, ...], "random_state": 0}. "n_clusters" is a list for the number of clusters to select from. "random_state" is the seed. Defaults to None. In this case, sensible cluster_args are chosen automatically.
            weight_by_frequency (bool, optional): Whether to weight the phrases by their frequency for the clustering algorithm. Defaults to False.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.
        """

        if self.clustering is None:
            print("No fitting required, this model is deterministic!")
        if self.clustering in ["hdbscan", "kmeans"]:
            self.fit_static_clustering(
                srl_res,
                pca_args,
                umap_args,
                cluster_args,
                weight_by_frequency,
                progress_bar,
            )
        if self.clustering == "dynamic":
            pass

    def fit_static_clustering(
        self,
        srl_res,
        pca_args,
        umap_args,
        cluster_args,
        weight_by_frequency,
        progress_bar,
    ):
        """
        Fits the model with K-means or HDBSCAN.
        """

        phrases_to_embed = []
        counter_for_phrases = Counter()

        for role in self.roles_with_unknown_entities:
            temp_counter = count_values(srl_res, keys=[role])
            counter_for_phrases = counter_for_phrases + temp_counter
            phrases = list(temp_counter)

            # Remove known entities for the training of unknown entities
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

        if not weight_by_frequency:
            phrases_to_embed = sorted(list(set(phrases_to_embed)))

        vectors = self.embeddings_model.get_vectors(phrases_to_embed, progress_bar)

        # Remove np.nans to train the model (or it will break down) and only keep phrases that have a vector (or it will mess up labels)
        idx = [i[0] for i in np.argwhere(np.isnan(vectors).any(axis=1))]
        self.phrases_to_embed = [
            phrase for i, phrase in enumerate(phrases_to_embed) if i not in idx
        ]
        self.training_vectors = vectors[~np.isnan(vectors).any(axis=1)]

        # Dimension reduction via PCA
        if self.PCA:
            if progress_bar:
                print("Dimension reduction via PCA...")
                print("PCA parameters:")
                print(pca_args)
            self.pca_args = pca_args
            self.pca_model = PCA(**pca_args).fit(self.training_vectors)
            self.training_vectors = self.pca_model.transform(self.training_vectors)

        # Dimension reduction via UMAP
        if self.UMAP:
            if progress_bar:
                print("Dimension reduction via UMAP...")
                print("UMAP parameters:")
                print(umap_args)
            self.umap_args = umap_args
            self.umap_model = umap.umap_.UMAP(**umap_args).fit(self.training_vectors)
            self.training_vectors = self.umap_model.transform(self.training_vectors)

        # Clustering
        if progress_bar:
            print("Clustering phrases into clusters...")

        if self.clustering == "kmeans":
            if cluster_args is None:
                l = max(int(len(phrases_to_embed) / 100), 2)
                l = min(l, 1000)
                q0 = max(int(np.quantile(list(range(l)), 0.1)), 2)
                q1 = max(int(np.quantile(list(range(l)), 0.25)), 2)
                q2 = max(int(np.quantile(list(range(l)), 0.5)), 2)
                q3 = max(int(np.quantile(list(range(l)), 0.75)), 2)

                cluster_args = {"n_clusters": [q0, q1, q2, q3, l], "random_state": 0}

            if progress_bar:
                print("Clustering parameters chosen in this range:")
                print(cluster_args)

            # Grid search
            models = []
            for num_clusters in cluster_args["n_clusters"]:
                args = {}
                args["n_clusters"] = num_clusters

                for k, v in cluster_args.items():
                    if k not in ["n_clusters"]:
                        args[k] = v

                kmeans = KMeans(**args).fit(self.training_vectors)

                models.append(kmeans)

            silhouette_scores = []
            for model in models:
                silhouette_scores.append(
                    silhouette_score(
                        self.training_vectors,
                        model.labels_,
                        random_state=cluster_args["random_state"],
                    )
                )

            self.scores["silhouette"] = silhouette_scores

            l = np.argmax(self.scores["silhouette"])
            k = cluster_args["n_clusters"][l]

            print(
                "The silhouette score suggests the optimal number of clusters is {0}. This corresponds to index {1}.".format(
                    k, l
                )
            )

            inertia_scores = []
            for model in models:
                inertia_scores.append(model.inertia_)

            self.scores["inertia"] = inertia_scores

            kneedle = KneeLocator(
                cluster_args["n_clusters"],
                self.scores["inertia"],
                curve="convex",
                direction="decreasing",
            )

            k = kneedle.knee
            if k is None:
                warnings.warn(
                    f"Not enough clustering scenarios to find the elbow. Defaulting to silhouette score.",
                    RuntimeWarning,
                )
            else:
                l = [i for i, n in enumerate(cluster_args["n_clusters"]) if n == k][0]
                print(
                    "The elbow method (inertia score) suggests the optimal number of clusters is {0}. This corresponds to index {1}.".format(
                        k, l
                    )
                )

        if self.clustering == "hdbscan":
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
            else:
                if ["min_cluster_size", "min_samples"] not in cluster_args.keys():
                    raise ValueError(
                        "Please provide at least min_cluster_size and min_samples in cluster_args"
                    )

            if progress_bar:
                print("Clustering parameters chosen in this range:")
                print(cluster_args)

            # Grid search
            models = []
            dbcv_scores = []
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

                        hdb = hdbscan.HDBSCAN(**args).fit(self.training_vectors)
                        models.append(hdb)
                        score = hdbscan.validity.validity_index(
                            self.training_vectors.astype(np.float64), hdb.labels_
                        )
                        dbcv_scores.append(score)

            self.scores["DBCV"] = dbcv_scores
            l = np.argmax(self.scores["DBCV"])
            print(
                "The DBCV score suggests the index of the optimal clustering model is {0}.".format(
                    l
                )
            )

        self.index_optimal_model = l
        self.clustering_models = models
        self.cluster_args = cluster_args

        print("Labeling the clusters by the most frequent phrases...")
        self.vocab_unknown_entities = [{} for i, m in enumerate(self.clustering_models)]
        self.labels_unknown_entities = [
            {} for i, m in enumerate(self.clustering_models)
        ]
        for index_clustering_model, clustering_model in enumerate(
            self.clustering_models
        ):
            self.label_clusters(
                counter_for_phrases, phrases_to_embed, index_clustering_model
            )

    def predict(
        self,
        srl_res,
        index_clustering_model: Optional[int] = None,
        progress_bar: bool = False,
    ):
        """
        Predicts the narratives underlying SVO/AVP statements.
        """

        if index_clustering_model is None:
            clustering_model = self.clustering_models[self.index_optimal_model]

        narratives = deepcopy(srl_res)

        for role in self.roles_considered:
            if role in ["B-ARGM-NEG", "B-ARGM-MOD", "B-V"]:
                continue

            if progress_bar:
                print("\nPredicting entities for role: %s..." % role)

            flag_computed_vectors = False
            srl_index, phrases = make_list_from_key(role, srl_res)
            all_labels = ["" for i in phrases]

            # Match known entities (with character matching)
            if (
                role in self.roles_with_known_entities
                and self.assignment_to_known_entities == "character_matching"
            ):
                idx, labels_known_entities = self.character_matching(
                    phrases, progress_bar
                )

                for i, k in enumerate(idx):
                    all_labels[k] = labels_known_entities[i]

                phrase_index, phrases_to_embed = [
                    i for i, p in enumerate(phrases) if i not in idx
                ], [p for i, p in enumerate(phrases) if i not in idx]

            # Match known entities (with embeddings distance)
            if (
                role in self.roles_with_known_entities
                and self.assignment_to_known_entities == "embeddings"
            ):
                phrases_to_embed = phrases.copy()
                phrase_index = [i for i, p in enumerate(phrases_to_embed)]

                vectors = self.embeddings_model.get_vectors(
                    phrases_to_embed, progress_bar
                )
                nan_index = np.argwhere(np.isnan(vectors).any(axis=1))
                vectors = vectors[~np.isnan(vectors).any(axis=1)]
                phrase_index = [i for i in phrase_index if i not in nan_index]
                phrases_to_embed = [
                    phrase
                    for i, phrase in enumerate(phrases_to_embed)
                    if i in phrase_index
                ]

                if progress_bar:
                    print("Matching known entities (with embeddings distance)...")

                idx, index_known_entities = _embeddings_similarity(
                    vectors, self.vectors_known_entities, self.threshold
                )
                labels_known_entities = self.label_with_known_entity(
                    index_known_entities
                )
                flag_computed_vectors = True

                for i, k in enumerate(idx):
                    all_labels[phrase_index[k]] = labels_known_entities[i]

            # Predict unknown entities (with clustering model)
            if role in self.roles_with_unknown_entities:
                if progress_bar:
                    print("Matching unknown entities (with clustering model)...")

                if flag_computed_vectors == False:
                    vectors = self.embeddings_model.get_vectors(
                        phrases_to_embed, progress_bar
                    )
                    nan_index = np.argwhere(np.isnan(vectors).any(axis=1))
                    vectors = vectors[~np.isnan(vectors).any(axis=1)]
                    phrase_index = [i for i in phrase_index if i not in nan_index]
                    phrases_to_embed = [
                        phrase
                        for i, phrase in enumerate(phrases_to_embed)
                        if i in phrase_index
                    ]

                if self.PCA:
                    if progress_bar:
                        print("Dimension reduction of vectors (PCA)...")

                    vectors = self.pca_model.transform(vectors)

                if self.UMAP:
                    if progress_bar:
                        print("Dimension reduction of vectors (UMAP)...")

                    vectors = self.umap_model.transform(vectors)

                if progress_bar:
                    print("Assignment to clusters...")

                if self.clustering == "hdbscan":
                    cluster_index = hdbscan.approximate_predict(
                        clustering_model, vectors
                    )[0]
                    idx = list(range(len(cluster_index)))

                else:
                    idx, cluster_index = _embeddings_similarity(
                        vectors, clustering_model.cluster_centers_
                    )

                cluster_labels = self.label_with_most_frequent_phrase(
                    cluster_index, index_clustering_model
                )

                for i, k in enumerate(idx):
                    all_labels[phrase_index[k]] = cluster_labels[i]

            # Assign labels to AVP/SVO statements
            for i, k in enumerate(phrases):
                if all_labels[i] != "":
                    narratives[srl_index[i]][role] = all_labels[i]
                elif role in narratives[srl_index[i]]:
                    narratives[srl_index[i]].pop(role)

        return narratives

    def character_matching(self, phrases, progress_bar: bool = False):
        """
        Character matching between a list of phrases and the list of known entitites.

        Args:
            phrases (list): list of phrases to match with known entities.
            progress_bar (bool): whether to show a progress bar.
        """

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

    def label_clusters(
        self, counter_for_phrases, phrases_to_embed, index_clustering_model: int
    ):
        """
        Method to label clusters with the most frequent phrase.

        Args:
            counter_for_phrases (Counter): Counter of phrases.
            phrases_to_embed (list): list of phrases to embed.
            index_clustering_model (int): index of the clustering model to be used.
        """

        labels = list(set(self.clustering_models[index_clustering_model].labels_))

        for clu in labels:
            self.vocab_unknown_entities[index_clustering_model][clu] = Counter()

        for j, clu in enumerate(self.clustering_models[index_clustering_model].labels_):
            self.vocab_unknown_entities[index_clustering_model][clu][
                phrases_to_embed[j]
            ] = counter_for_phrases[phrases_to_embed[j]]

        for clu in labels:
            token_most_common = self.vocab_unknown_entities[index_clustering_model][
                clu
            ].most_common(2)
            if len(token_most_common) > 1 and (
                token_most_common[0][1] == token_most_common[1][1]
            ):
                warnings.warn(
                    f"Multiple labels for cluster {clu}- 2 shown: {token_most_common}. First one is picked.",
                    RuntimeWarning,
                )
            self.labels_unknown_entities[index_clustering_model][
                clu
            ] = token_most_common[0][0]

        if self.clustering == "hdbscan":
            self.labels_unknown_entities[index_clustering_model][-1] = ""

    def label_with_known_entity(self, index):
        return [self.known_entities[i] for i in index]

    def label_with_most_frequent_phrase(
        self, index, index_clustering_model: Optional[int] = None
    ):
        if index_clustering_model is None:
            index_clustering_model = self.index_optimal_model

        return [self.labels_unknown_entities[index_clustering_model][i] for i in index]

    def inspect_cluster(
        self, label, index_clustering_model: Optional[int] = None, topn=10
    ):
        """
        Show the most frequent phrases in a cluster.

        Args:
            label (str): label of the cluster.
            index_clustering_model (int): index of the clustering model to be used.
            topn (int): number of most frequent phrases to show.
        """

        if index_clustering_model is None:
            index_clustering_model = self.index_optimal_model

        key = [
            k
            for k, v in self.labels_unknown_entities[index_clustering_model].items()
            if v == label
        ][0]
        return self.vocab_unknown_entities[index_clustering_model][key].most_common(
            topn
        )

    def clusters_to_txt(
        self,
        index_clustering_model: Optional[int] = None,
        path="clusters.txt",
        topn=10,
        add_frequency_info=True,
    ):
        """
        Prints the most frequent phrases in each cluster to a txt file.

        Args:
            index_clustering_model (int): index of the clustering model to be used.
            path (str): path to the txt file.
            topn (int): number of most frequent phrases to show.
            add_frequency_info (bool): whether to add the frequency of the phrases.
        """

        if index_clustering_model is None:
            index_clustering_model = self.index_optimal_model

        with open(path, "w") as f:
            for k, v in self.vocab_unknown_entities[index_clustering_model].items():
                f.write("Cluster %s" % k)
                f.write("\n")
                for i in v.most_common(topn):
                    if add_frequency_info == True:
                        f.write("%s (%s), " % (i[0], i[1]))
                    else:
                        f.write("%s, " % i[0])
                f.write("\n")
                f.write("\n")

    def plot_clusters(
        self,
        index_clustering_model: Optional[int] = None,
        path=None,
        figsize=(14, 8),
        s=0.1,
    ):
        """
        Plots the clusters in 2D using UMAP for dimension reduction.

        Args:
            index_clustering_model (int): index of the clustering model to be used. If None, the optimal model is used.
            path (str): path to the image file. If None, the figure is shown.
            figsize (width, height): size of the figure.
            s (float): size of the points on the figure.
        """

        if index_clustering_model is None:
            index_clustering_model = self.index_optimal_model

        if self.umap_args["n_components"] != 2:
            umap_args = {"n_neighbors": 15, "n_components": 2, "random_state": 0}
            umap_model = umap.UMAP(**umap_args).fit(self.training_vectors)
            vectors = umap_model.transform(self.training_vectors)
            clustered = self.clustering_models[index_clustering_model].labels_ >= 0
            plt.figure(figsize=figsize, dpi=80)
            plt.scatter(
                vectors[~clustered, 0],
                vectors[~clustered, 1],
                color=(0.5, 0.5, 0.5),
                s=s,
                alpha=0.5,
            )
            plt.scatter(
                vectors[clustered, 0],
                vectors[clustered, 1],
                c=self.clustering_models[index_clustering_model].labels_[clustered],
                s=s,
                cmap="Spectral",
            )
        else:
            clustered = self.clustering_models[index_clustering_model].labels_ >= 0
            plt.figure(figsize=figsize, dpi=80)
            plt.scatter(
                self.training_vectors[~clustered, 0],
                self.training_vectors[~clustered, 1],
                color=(0.5, 0.5, 0.5),
                s=s,
                alpha=0.5,
            )
            plt.scatter(
                self.training_vectors[clustered, 0],
                self.training_vectors[clustered, 1],
                c=self.clustering_models[index_clustering_model].labels_[clustered],
                s=s,
                cmap="Spectral",
            )

        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    def plot_selection_metric(
        self, metric: Optional[str] = None, path=None, figsize=(14, 8)
    ):
        """
        Plots the selection metric for the clustering models.

        Args:
            metric (str): metric to be plotted. If None, defaults to inertia for KMeans and DBCV for HDBSCAN.
            path (str): path to the image file. If None, the figure is shown.
            figsize (width, height): size of the figure.
        """

        if not metric:
            if self.clustering == "hdbscan":
                metric = "DBCV"
            elif self.clustering == "kmeans":
                metric = "inertia"

        if self.clustering == "hdbscan":
            if metric == "DBCV":
                plot_args: Dict[str, list] = {}
                plot_args["min_cluster_size"], plot_args["min_samples"] = [], []
                best_score_args: Dict[str, int] = {}
                for i in self.cluster_args["min_cluster_size"]:
                    for j in self.cluster_args["min_samples"]:
                        plot_args["min_cluster_size"].append(i)
                        plot_args["min_samples"].append(j)

                max_index = np.argmax(self.scores["DBCV"])
                best_score_args["min_cluster_size"] = plot_args["min_cluster_size"][
                    max_index
                ]
                best_score_args["min_samples"] = plot_args["min_samples"][max_index]
                best_score_args["DBCV"] = self.scores["DBCV"][max_index]

                fig = plt.figure(figsize=figsize)
                ax = Axes3D(fig)
                ax.set_title("DBCV score plot", fontsize="x-large")
                ax.set_xlabel("Minimum cluster size")
                ax.set_ylabel("Minimum samples")
                ax.set_zlabel("DBCV Score")
                ax.scatter(
                    plot_args["min_cluster_size"],
                    plot_args["min_samples"],
                    self.scores["DBCV"],
                    "o",
                    s=[(i + 1) * 100 for i in self.scores["DBCV"]],
                    c="#808080",
                )
                ax.scatter(
                    best_score_args["min_cluster_size"],
                    best_score_args["min_samples"],
                    best_score_args["DBCV"],
                    "o",
                    s=[(best_score_args["DBCV"] + 1) * 100],
                    c="#2ca02c",
                )
                ax.set_xticks(list(set(plot_args["min_cluster_size"])))
                ax.set_yticks(list(set(plot_args["min_samples"])))
            else:
                raise ValueError("This metric is not available for HDBSCAN.")

        if self.clustering == "kmeans":
            if metric == "silhouette":
                plt.figure(figsize=figsize)
                plt.plot(
                    self.cluster_args["n_clusters"], self.scores["silhouette"], "bx-"
                )
                plt.xlabel("Number of Clusters")
                plt.ylabel("Silhouette Score")
            elif metric == "inertia":
                plt.figure(figsize=figsize)
                plt.plot(self.cluster_args["n_clusters"], self.scores["inertia"], "bx-")
                plt.xlabel("Number of Clusters")
                plt.ylabel("Inertia")
            else:
                raise ValueError("This metric is not available for KMeans.")

        if path is None:
            plt.show()
        else:
            plt.savefig(path)
