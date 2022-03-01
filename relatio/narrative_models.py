from abc import ABC, abstractmethod
from typing import Type
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
import warnings

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from relatio._embeddings import *
from relatio.utils import is_subsequence, count_values, prettify

from spacy.cli import download as spacy_download


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
        roles_with_entities: str = ["ARG0", "ARG1", "ARG2"],
        list_of_known_entities: Optional[List[str]] = None,
        assignment_to_known_entities: str = "character_matching",
        roles_with_embeddings: List[List[str]] = [["ARG0", "ARG1", "ARG2"]],
        embeddings_model: Optional[Type[Embeddings]] = None,
        threshold: int = 0.1,
    ):

        self.roles_considered = roles_considered
        self.roles_with_embeddings = roles_with_embeddings
        self.roles_with_entities = roles_with_entities
        self.list_of_known_entities = list_of_known_entities
        self.vectors_of_known_entities = None
        self.assignment_to_known_entities = assignment_to_known_entities
        self.threshold = threshold

        if embeddings_model is None:
            if not spacy.util.is_package("en_core_web_sm"):
                spacy_download(spacy_model)
            self.embeddings_model = Embeddings("spaCy", "en_core_web_sm")
        else:
            self.embeddings_model = embeddings_model

        if self.list_of_known_entities is not None:
            if self.assignment_to_known_entities == "embeddings":
                print("Computing vectors for known entities...")
                self.vectors_of_known_entities = []
                for i, entity in enumerate(self.list_of_known_entities):
                    vector = self.embeddings_model.get_vector(entity)
                    if vector is not None:
                        self.vectors_of_known_entities.append(np.array([vector]))

                self.vectors_of_known_entities = np.concatenate(
                    self.vectors_of_known_entities
                )

        self.vectors = []
        self.labels = []
        self.vocab = []
        for l in roles_with_embeddings:
            self.vectors.append([])
            self.labels.append({})
            self.vocab.append({})

    @abstractmethod
    def train(self, list_of_srl_res):
        pass

    def compute_distances(self, vector, vectors):
        distances = cdist(vectors, vector.reshape(1, -1), metric="euclidean").flatten()
        return distances

    def _process_srl_item_for_prediction(self, role, content):

        # Character matching of known entities
        if role in self.roles_with_entities:
            if self.assignment_to_known_entities == "character_matching":
                list_of_matched_entities = []
                for entity in self.list_of_known_entities:
                    if is_subsequence(entity.split(), content.split()):
                        list_of_matched_entities.append(entity)
                if len(list_of_matched_entities) != 0:
                    return "|".join(list_of_matched_entities)

        # Clustering with embeddings
        for i, l in enumerate(self.roles_with_embeddings):
            if role in l:
                vector = self.embeddings_model.get_vector(content)
                if vector is not None:

                    # Known entities
                    if role in self.roles_with_entities:
                        if self.assignment_to_known_entities == "embeddings":
                            distances = self.compute_distances(
                                vector, self.vectors_of_known_entities
                            )
                            nmin = min(distances)
                            if nmin <= self.threshold:
                                entity_index = np.where(
                                    distances == np.amin(distances)
                                )[0][0]
                                return self.list_of_known_entities[entity_index]

                    # Unknown entities
                    if len(self.vectors[i]) != 0:
                        distances = self.compute_distances(vector, self.vectors[i])
                        nmin = min(distances)
                        if nmin <= self.threshold:
                            clu = np.where(distances == np.amin(distances))[0][0]
                            return self.labels[i][clu]

    def predict(
        self, list_of_srl_res, prettify: bool = False, progress_bar: bool = False
    ):
        """

        Predict the latent narrative statement based of a SRL statement.

        """

        list_of_narratives = deepcopy(list_of_srl_res)

        if progress_bar:
            list_of_srl_res = tqdm(list_of_srl_res)

        for i, srl_res in enumerate(list_of_srl_res):
            for role, content in srl_res.items():
                if role in self.roles_considered:
                    if role in ["ARG0", "ARG1", "ARG2"]:
                        pred = self._process_srl_item_for_prediction(role, content)
                        if pred is not None:
                            list_of_narratives[i][role] = pred
                        else:
                            list_of_narratives[i].pop(role, None)
                else:
                    list_of_narratives[i].pop(role, None)

        if prettify:
            list_of_narratives = [
                prettify(narrative) for narrative in list_of_narratives
            ]

        return list_of_narratives


class DeterministicModel(NarrativeModelBase):
    """

    A subclass of NarrativeModel(), which does nothing more.

    """

    def train(self, list_of_srl_res):
        print("No training required: the model is deterministic.")


class StaticModel(NarrativeModelBase):
    """

    A subclass of NarrativeModel(), which mines K latent entities via a one-off clustering algorithm
    on a training set (e.g., K-Means).

    """

    def __init__(
        self,
        roles_considered,
        roles_with_entities,
        list_of_known_entities,
        assignment_to_known_entities,
        roles_with_embeddings,
        embeddings_model,
        threshold: int,
        n_clusters: List[int],
    ):

        super().__init__(
            roles_considered,
            roles_with_entities,
            list_of_known_entities,
            assignment_to_known_entities,
            roles_with_embeddings,
            embeddings_model,
            threshold,
        )

        self.n_clusters = n_clusters
        self.clustering_models = []
        self.training_vectors = []

        for l in roles_with_embeddings:
            self.clustering_models.append({})
            self.training_vectors.append({})

    def _remove_entities(self, list_of_phrases: List[str]):

        if self.assignment_to_known_entities == "character_matching":
            for phrase in list_of_phrases:
                for entity in self.list_of_known_entities:
                    if is_subsequence(entity.split(), phrase.split()):
                        list_of_phrases.remove(phrase)
                        break
        else:
            for phrase in list_of_phrases:
                vector = self.embeddings_model.get_vector(phrase)
                if vector is not None:
                    distances = self.compute_distances(
                        vector, self.vectors_of_known_entities
                    )
                    nmin = min(distances)
                    if nmin <= self.threshold:
                        list_of_phrases.remove(phrase)

        return list_of_phrases

    def train(
        self,
        list_of_srl_res,
        random_state: int = 1,
        verbose: int = 0,
        max_iter: int = 300,
        progress_bar: bool = False,
    ):

        for i, roles in enumerate(self.roles_with_embeddings):
            if progress_bar:

                print("Focus on roles: %s" % "-".join(roles))
                print("Ignoring known entities...")

            phrases_to_embed = []
            counter_for_phrases = Counter()
            for role in roles:
                temp_counter = count_values(list_of_srl_res, keys=[role])
                counter_for_phrases = counter_for_phrases + temp_counter
                phrases = list(temp_counter)
                if role in self.roles_with_entities:
                    phrases = self._remove_entities(phrases)
                phrases_to_embed.extend(phrases)

            phrases_to_embed = list(set(phrases_to_embed))

            if progress_bar:
                print("Embedding relevant phrases...")

            vecs = []
            for phrase in phrases_to_embed:
                vec = self.embeddings_model.get_vector(phrase)
                if vec is None:
                    phrases_to_embed.remove(phrase)
                else:
                    vec = np.array([vec])
                    if vec is not None:
                        vecs.append(vec)

            vecs = np.concatenate(vecs)
            self.training_vectors[i] = vecs

            if progress_bar:
                print("Clustering phrases into %s clusters..." % self.n_clusters[i])

            kmeans = KMeans(
                n_clusters=self.n_clusters[i],
                random_state=random_state,
                verbose=verbose,
                max_iter=max_iter,
            ).fit(self.training_vectors[i])

            self.clustering_models[i] = kmeans
            self.vectors[i] = kmeans.cluster_centers_

            if progress_bar:
                print("Labeling the clusters by the most frequent phrases...")

            for clu in range(self.n_clusters[i]):
                self.vocab[i][clu] = Counter()
            for j, clu in enumerate(kmeans.labels_):
                self.vocab[i][clu][phrases_to_embed[j]] = counter_for_phrases[
                    phrases_to_embed[j]
                ]
            for clu in range(self.n_clusters[i]):
                token_most_common = self.vocab[i][clu].most_common(2)
                if len(token_most_common) > 1 and (
                    token_most_common[0][1] == token_most_common[1][1]
                ):
                    warnings.warn(
                        f"Multiple labels for cluster {clu}- 2 shown: {token_most_common}. First one is picked.",
                        RuntimeWarning,
                    )
                self.labels[i][clu] = token_most_common[0][0]


class DynamicModel(NarrativeModelBase):
    """

    A subclass of NarrativeModel(), which mines latent entities on-the-fly based on a dynamic clustering algorithm.

    """

    def __init__(
        self,
        roles_considered,
        roles_with_entities,
        list_of_known_entities,
        assignment_to_known_entities,
        roles_with_embeddings,
        embeddings_model,
        threshold: int = 0.1,
    ):

        super().__init__(
            roles_considered,
            roles_with_entities,
            list_of_known_entities,
            assignment_to_known_entities,
            roles_with_embeddings,
            embeddings_model,
            threshold,
        )

    def _process_srl_item_for_training(self, role, content):

        if role in self.roles_considered:
            if role in ["ARG0", "ARG1", "ARG2"]:

                # Character matching of known entities (skipped for training)
                if role in self.roles_with_entities:
                    if self.assignment_to_known_entities == "character_matching":
                        for e in self.list_of_known_entities:
                            if is_subsequence(entity.split(), content.split()):
                                return None

                # Clustering with embeddings
                for i, l in enumerate(self.roles_with_embeddings):
                    if role in l:
                        vector = self.embeddings_model.get_vector(content)
                        if vector is not None:

                            # Known entities (skipped for training)
                            if role in self.roles_with_entities:
                                if self.assignment_to_known_entities == "embeddings":
                                    distances = self.compute_distances(
                                        vector, self.vectors_of_known_entities
                                    )
                                    nmin = min(distances)
                                    if nmin <= self.threshold:
                                        return None

                            # Unknown entities
                            if len(self.vectors[i]) == 0:
                                self.vectors[i] = np.array([vector])
                                self.vocab[i][0] = Counter()
                                self.vocab[i][0][content] = 1
                                self.labels[i][0] = content
                            else:
                                clu = len(self.vectors[i])
                                distances = self.compute_distances(
                                    vector, self.vectors[i]
                                )
                                nmin = min(distances)
                                if nmin <= self.threshold:
                                    clu = np.where(distances == np.amin(distances))[0][
                                        0
                                    ]
                                    self.vocab[i][clu][content] += 1

                                    token_most_common = self.vocab[i][clu].most_common(
                                        2
                                    )
                                    if len(token_most_common) > 1 and (
                                        token_most_common[0][1]
                                        == token_most_common[1][1]
                                    ):
                                        warnings.warn(
                                            f"Multiple labels for cluster {clu}- 2 shown: {token_most_common}. First one is picked.",
                                            RuntimeWarning,
                                        )
                                    self.labels[i][clu] = token_most_common[0][0]
                                else:
                                    self.vectors[i] = np.append(
                                        self.vectors[i], [vector], axis=0
                                    )
                                    self.vocab[i][clu] = Counter()
                                    self.vocab[i][clu][content] = 1
                                    self.labels[i][clu] = content

    def train(self, list_of_srl_res, progress_bar: bool = False):

        if progress_bar:
            list_of_srl_res = tqdm(list_of_srl_res)

        for srl_res in list_of_srl_res:
            for role, content in srl_res.items():
                self._process_srl_item_for_training(role, content)


class NarrativeModel(NarrativeModelBase):

    """
    
    The NarrativeModel class for users.
    
    Args:
        model_type: 'deterministic', 'static' and 'dynamic'
        roles_considered: list of semantic roles to consider 
        (default: ["ARG0", "B-V", "ARGM-MOD", "ARG1", "ARG2"])
        roles_with_entities: roles to consider for the known entities
        (default: ["ARG0", "ARG1", "ARG2"])
        list_of_known_entities: a list of known entities
        assignment_to_known_entities: character_matching or embeddings (default: character_matching)
        roles_with_embeddings: list of lists of semantic roles to embed and cluster
        (i.e. each list represents semantic roles that should be clustered together)
        embeddings_model: an object of type Embeddings
        threshold: If the assignment to known entities is performed via embeddings, we compute the distance between konwn entities and a phrase. Define the threshold below which a phrase is considered a known entitiy (default: 0.1. This is a very low threshold which requires strong similarity between the phrase and the known entities).
        
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
        roles_with_entities: str = ["ARG0", "ARG1", "ARG2"],
        list_of_known_entities: List[str] = [],
        assignment_to_known_entities: str = "character_matching",
        roles_with_embeddings: List[str] = [["ARG0", "ARG1", "ARG2"]],
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

        if roles_with_entities is not None:
            if is_subsequence(roles_with_entities, roles_considered) is False:
                raise ValueError("roles_with_entities should be in roles_considered.")

        if roles_with_embeddings is not None:
            for roles in roles_with_embeddings:
                if is_subsequence(roles, roles_considered) is False:
                    raise ValueError(
                        "each list in roles_with_embeddings should be a subset of roles_considered."
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
            roles_with_entities,
            list_of_known_entities,
            assignment_to_known_entities,
            roles_with_embeddings,
            embeddings_model,
            threshold,
            **kwargs,
        )

    def train(self, list_of_srl_res, **kwargs):
        self._model_obj.train(list_of_srl_res, **kwargs)

    def predict(
        self, list_of_srl_res, prettify: bool = False, progress_bar: bool = False
    ):
        return self._model_obj.predict(list_of_srl_res, prettify, progress_bar)
