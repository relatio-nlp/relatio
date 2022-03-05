import warnings
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from typing import List, Optional, Type

import spacy
from sklearn.cluster import KMeans
from spacy.cli import download as spacy_download
from tqdm import tqdm

from relatio._embeddings import (
    Embeddings,
    _compute_distances,
    _embeddings_similarity,
    _get_index_min_distances,
    _get_min_distances,
    _remove_nan_vectors,
)
from relatio.utils import count_values, is_subsequence, make_list_from_key, prettify


class NarrativeModelBase(ABC):
    """
    A general class to build a model that extracts latent narratives from a list of SRL statements.
    """

    def __init__(
        self,
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
        roles_with_unknown_entities: List[List[str]] = [["ARG0", "ARG1", "ARG2"]],
        embeddings_model: Optional[Type[Embeddings]] = None,
        threshold: int = 0.1,
    ):

        self.roles_considered = roles_considered
        self.roles_with_unknown_entities = roles_with_unknown_entities
        self.roles_with_known_entities = roles_with_known_entities
        self.known_entities = known_entities
        self.vectors_known_entities = None
        self.assignment_to_known_entities = assignment_to_known_entities
        self.threshold = threshold

        # Default embeddings model is a small spacy model.
        if embeddings_model is None:
            if not spacy.util.is_package("en_core_web_sm"):
                spacy_download(spacy_model)
            self.embeddings_model = Embeddings("spaCy", "en_core_web_sm")
        else:
            self.embeddings_model = embeddings_model

        if (
            self.known_entities is not None
            and self.assignment_to_known_entities == "embeddings"
        ):
            self.vectors_known_entities = self.embeddings_model._get_vectors(
                self.known_entities
            )

        self.vectors_unknown_entities = []
        self.labels_unknown_entities = []
        self.vocab_unknown_entities = []

        for l in roles_with_unknown_entities:
            self.vectors_unknown_entities.append([])
            self.labels_unknown_entities.append({})
            self.vocab_unknown_entities.append({})

    @abstractmethod
    def train(self, srl_res):
        pass

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
                index2, labels_known_entities = self._character_matching(
                    phrases, progress_bar
                )

            # Match known entities (with embeddings distance)
            if (
                role in self.roles_with_known_entities
                and self.assignment_to_known_entities == "embeddings"
            ):
                vectors = self.embeddings_model._get_vectors(phrases, progress_bar)

                if progress_bar:
                    print("Matching known entities (with embeddings distance)...")

                index2, index_known_entities = _embeddings_similarity(
                    vectors, self.vectors_known_entities, self.threshold
                )
                labels_known_entities = self._label_with_known_entity(
                    index_known_entities
                )
                flag_computed_vectors = True

            # Match unknown entities (with embeddings distance)
            for k, roles in enumerate(self.roles_with_unknown_entities):
                if role in roles and len(self.vectors_unknown_entities[k]) != 0:

                    if progress_bar:
                        print("Matching unknown entities (with embeddings distance)...")

                    if flag_computed_vectors == False:
                        vectors = self.embeddings_model._get_vectors(
                            phrases, progress_bar
                        )

                    index3, index_clusters = _embeddings_similarity(
                        vectors, self.vectors_unknown_entities[k]
                    )
                    cluster_labels = self._label_with_most_frequent_phrase(
                        index_clusters, k
                    )

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

    def _character_matching(self, phrases, progress_bar: bool = False):

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

    def _label_with_known_entity(self, index):
        return [self.known_entities[i] for i in index]

    def _label_with_most_frequent_phrase(self, index, k):
        return [self.labels_unknown_entities[k][i] for i in index]


class DeterministicModel(NarrativeModelBase):
    """
    A subclass of NarrativeModel(), which does nothing more.
    """

    def train(self, srl_res):
        print("No training required: the model is deterministic.")


class StaticModel(NarrativeModelBase):
    """
    A subclass of NarrativeModel(), which mines K latent entities via a one-off clustering algorithm
    on a training set (e.g., K-Means).
    """

    def __init__(
        self,
        roles_considered,
        roles_with_known_entities,
        known_entities,
        assignment_to_known_entities,
        roles_with_unknown_entities,
        embeddings_model,
        threshold: int,
        n_clusters: List[int],
    ):

        super().__init__(
            roles_considered,
            roles_with_known_entities,
            known_entities,
            assignment_to_known_entities,
            roles_with_unknown_entities,
            embeddings_model,
            threshold,
        )

        self.n_clusters = n_clusters
        self.clustering_models = []
        self.training_vectors = []

        for l in roles_with_unknown_entities:
            self.clustering_models.append({})
            self.training_vectors.append({})

    def train(
        self,
        srl_res,
        random_state: int = 1,
        verbose: int = 0,
        max_iter: int = 300,
        progress_bar: bool = False,
    ):

        for i, roles in enumerate(self.roles_with_unknown_entities):
            if progress_bar:

                print("Focus on roles: %s" % "-".join(roles))
                print("Ignoring known entities...")

            phrases_to_embed = []
            counter_for_phrases = Counter()

            for role in roles:

                temp_counter = count_values(srl_res, keys=[role])
                counter_for_phrases = counter_for_phrases + temp_counter
                phrases = list(temp_counter)

                # remove known entities for the training of unknown entities
                if role in self.roles_with_known_entities:
                    if self.assignment_to_known_entities == "character_matching":
                        idx = self._character_matching(phrases)[0]
                    elif self.assignment_to_known_entities == "embeddings":
                        vectors = self.embeddings_model._get_vectors(
                            phrases, progress_bar
                        )
                        idx = _embeddings_similarity(
                            vectors, self.vectors_known_entities, self.threshold
                        )[0]
                    phrases = [
                        phrase for l, phrase in enumerate(phrases) if l not in idx
                    ]

                phrases_to_embed.extend(phrases)

            phrases_to_embed = list(set(phrases_to_embed))

            # Remove np.nans to train the KMeans model (or it will break down)
            vectors = self.embeddings_model._get_vectors(phrases_to_embed, progress_bar)
            self.training_vectors[i] = _remove_nan_vectors(vectors)

            self._train_kmeans(i, random_state, verbose, max_iter, progress_bar)
            self._label_clusters(i, counter_for_phrases, phrases_to_embed, progress_bar)

    def _train_kmeans(self, i, random_state, verbose, max_iter, progress_bar):

        if progress_bar:
            print("Clustering phrases into %s clusters..." % self.n_clusters[i])

        kmeans = KMeans(
            n_clusters=self.n_clusters[i],
            random_state=random_state,
            verbose=verbose,
            max_iter=max_iter,
        ).fit(self.training_vectors[i])

        self.clustering_models[i] = kmeans
        self.vectors_unknown_entities[i] = kmeans.cluster_centers_

    def _label_clusters(self, i, counter_for_phrases, phrases_to_embed, progress_bar):

        if progress_bar:
            print("Labeling the clusters by the most frequent phrases...")

        for clu in range(self.n_clusters[i]):
            self.vocab_unknown_entities[i][clu] = Counter()
        for j, clu in enumerate(self.clustering_models[i].labels_):
            self.vocab_unknown_entities[i][clu][
                phrases_to_embed[j]
            ] = counter_for_phrases[phrases_to_embed[j]]
        for clu in range(self.n_clusters[i]):
            token_most_common = self.vocab_unknown_entities[i][clu].most_common(2)
            if len(token_most_common) > 1 and (
                token_most_common[0][1] == token_most_common[1][1]
            ):
                warnings.warn(
                    f"Multiple labels for cluster {clu}- 2 shown: {token_most_common}. First one is picked.",
                    RuntimeWarning,
                )
            self.labels_unknown_entities[i][clu] = token_most_common[0][0]


class DynamicModel(NarrativeModelBase):
    """
    A subclass of NarrativeModel(), which mines latent entities on-the-fly based on a dynamic clustering algorithm.
    """

    def __init__(
        self,
        roles_considered,
        roles_with_known_entities,
        known_entities,
        assignment_to_known_entities,
        roles_with_unknown_entities,
        embeddings_model,
        threshold: int = 0.1,
    ):

        super().__init__(
            roles_considered,
            roles_with_known_entities,
            known_entities,
            assignment_to_known_entities,
            roles_with_unknown_entities,
            embeddings_model,
            threshold,
        )

    def _process_srl_item_for_training(self, role, content):

        # Character matching of known entities (skipped for training)
        if role in self.roles_with_known_entities:
            if self.assignment_to_known_entities == "character_matching":
                for entity in self.known_entities:
                    if is_subsequence(entity.split(), content.split()):
                        return None

        # Clustering with embeddings
        for i, l in enumerate(self.roles_with_unknown_entities):
            if role in l:
                vector = self.embeddings_model.get_vector(content)
                if vector is not None:

                    # Known entities (skipped for training)
                    if role in self.roles_with_known_entities:
                        if self.assignment_to_known_entities == "embeddings":
                            distances = _compute_distances(
                                vector, self.vectors_unknown_entities_of_known_entities
                            )
                            nmin = min(distances)
                            if nmin <= self.threshold:
                                return None

                    # Unknown entities
                    if len(self.vectors_unknown_entities[i]) == 0:
                        self.vectors_unknown_entities[i] = np.array([vector])
                        self.vocab_unknown_entities[i][0] = Counter()
                        self.vocab_unknown_entities[i][0][content] = 1
                        self.labels_unknown_entities[i][0] = content
                    else:
                        clu = len(
                            self.vectors_unknown_entities[i]
                        )  # some refactoring required here
                        distances = _compute_distances(
                            vector, self.vectors_unknown_entities[i]
                        )
                        nmin = min(distances)
                        if nmin <= self.threshold:
                            clu = np.where(distances == np.amin(distances))[0][0]
                            self.vocab_unknown_entities[i][clu][content] += 1

                            token_most_common = self.vocab_unknown_entities[i][
                                clu
                            ].most_common(2)
                            if len(token_most_common) > 1 and (
                                token_most_common[0][1] == token_most_common[1][1]
                            ):
                                warnings.warn(
                                    f"Multiple labels for cluster {clu}- 2 shown: {token_most_common}. First one is picked.",
                                    RuntimeWarning,
                                )
                            self.labels_unknown_entities[i][clu] = token_most_common[0][
                                0
                            ]
                        else:
                            self.vectors_unknown_entities[i] = np.append(
                                self.vectors_unknown_entities[i], [vector], axis=0
                            )
                            self.vocab_unknown_entities[i][clu] = Counter()
                            self.vocab_unknown_entities[i][clu][content] = 1
                            self.labels_unknown_entities[i][clu] = content

    def train(self, srl_res, progress_bar: bool = False):

        if progress_bar:
            srl_res = tqdm(srl_res)

        for srl_res in srl_res:
            for role, content in srl_res.items():
                if role in self.roles_considered and role in ["ARG0", "ARG1", "ARG2"]:
                    self._process_srl_item_for_training(role, content)


class NarrativeModel(NarrativeModelBase):

    """
    The NarrativeModel class for users.

    Args:
        model_type: 'deterministic', 'static' and 'dynamic'
        roles_considered: list of semantic roles to consider
        (default: ["ARG0", "B-V", "ARGM-MOD", "ARG1", "ARG2"])
        roles_with_known_entities: roles to consider for the known entities
        (default: ["ARG0", "ARG1", "ARG2"])
        known_entities: a list of known entities
        assignment_to_known_entities: character_matching or embeddings (default: character_matching)
        roles_with_unknown_entities: list of lists of semantic roles to embed and cluster
        (i.e. each list represents semantic roles that should be clustered together)
        embeddings_model: an object of type Embeddings
        threshold: If the assignment to known entities is performed via embeddings, we compute the distance between known entities and a phrase. Define the threshold below which a phrase is considered a known entitiy (default: 0.1. This is a very low threshold which requires strong similarity between the phrase and the known entities).
    """

    def __init__(
        self,
        model_type,
        roles_considered: List[str] = [
            "ARG0",
            "B-V",
            "B-ARGM-MOD",
            "B-ARGM-NEG",
            "ARG1",
            "ARG2",
        ],
        roles_with_known_entities: str = ["ARG0", "ARG1", "ARG2"],
        known_entities: List[str] = [],
        assignment_to_known_entities: str = "character_matching",
        roles_with_unknown_entities: List[str] = [["ARG0", "ARG1", "ARG2"]],
        embeddings_model: Optional[Type[Embeddings]] = None,
        threshold: int = 0.1,
        **kwargs,
    ):

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
            for roles in roles_with_unknown_entities:
                if is_subsequence(roles, roles_considered) is False:
                    raise ValueError(
                        "each list in roles_with_unknown_entities should be a subset of roles_considered."
                    )
                if ["B-ARGM-NEG", "B-ARGM-MOD", "B-V"] in roles:
                    raise ValueError(
                        "Negations, verbs and modals cannot be embedded and clustered."
                    )

        if assignment_to_known_entities not in ["character_matching", "embeddings"]:
            raise ValueError(
                "Only two options for assignment_to_known_entities: character_matching or embeddings."
            )

        if model_type == "dynamic":
            _MODEL_CLASS = DynamicModel
        elif model_type == "static":
            _MODEL_CLASS = StaticModel
        elif model_type == "deterministic":
            _MODEL_CLASS = DeterministicModel
        else:
            raise ValueError(
                "Only three possible options for model_type: deterministic, static or dynamic."
            )

        self._model_obj = _MODEL_CLASS(
            roles_considered,
            roles_with_known_entities,
            known_entities,
            assignment_to_known_entities,
            roles_with_unknown_entities,
            embeddings_model,
            threshold,
            **kwargs,
        )

    def train(self, srl_res, **kwargs):
        self._model_obj.train(srl_res, **kwargs)

    def predict(self, srl_res, progress_bar: bool = False):
        return self._model_obj.predict(srl_res, progress_bar)
