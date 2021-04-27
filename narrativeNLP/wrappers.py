# Wrappers
# ..................................................................................................................
# ..................................................................................................................

import json
import os
import pickle as pk
from typing import List, Optional

import numpy as np
import pandas as pd

from .clustering import (
    USE,
    SIF_keyed_vectors,
    SIF_word2vec,
    get_clusters,
    get_vectors,
    label_clusters_most_freq,
    label_clusters_most_similar,
    train_cluster_model,
)
from .named_entity_recognition import map_entities, mine_entities
from .semantic_role_labeling import SRL, extract_roles, rename_arguments, process_roles
from .utils import clean_text, count_values, is_subsequence
from .verbs import clean_verbs


def run_srl(
    path: str,
    sentences: List[str],
    batch_size: Optional[int] = None,
    max_batch_char_length: Optional[int] = 20_000,
    cuda_device: int = -1,
    max_sentence_length: Optional[int] = None,
    max_number_words: Optional[int] = None,
    cuda_empty_cache: bool = None,
    cuda_sleep: float = None,
    output_path: Optional[str] = None,
    progress_bar: bool = False,
):

    """

    A wrapper function to run semantic role labeling on a corpus.

    Args:
        path: location of the SRL model to be used
        sentences: list of sentences
        SRL_options: see class SRL()
        output_path: path to save the narrative model (default is None, which means no saving to disk)
        progress_bar: print a progress bar (default is False)

    Returns:
        A list of dictionaries with the SRL output

    """

    srl = SRL(path=path, cuda_device=cuda_device)

    srl_res = srl(
        sentences=sentences,
        batch_size=batch_size,
        max_batch_char_length=max_batch_char_length,
        max_sentence_length=max_sentence_length,
        max_number_words=max_number_words,
        cuda_empty_cache=cuda_empty_cache,
        cuda_sleep=cuda_sleep,
        progress_bar=progress_bar,
    )

    if output_path is not None:
        with open(output_path, "w") as json_file:
            json.dump(srl_res, json_file)

    return srl_res


def build_narrative_model(  # add more control for the user on clustering (n_jobs, random_state, etc.)
    srl_res: List[dict],
    sentences: List[str],
    roles_considered: List[str] = [
        "ARG0",
        "B-V",
        "B-ARGM-NEG",
        "B-ARGM-MOD",
        "ARG1",
        "ARG2",
    ],
    output_path: Optional[str] = None,
    max_length: Optional[int] = None,
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    stop_words: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_whitespaces: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
    roles_with_embeddings: List[List[str]] = [["ARG0", "ARG1", "ARG2"]],
    embeddings_type: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    n_clusters: List[int] = [1],
    verbose: int = 0,
    random_state: int = 0,
    roles_with_entities: List[str] = ["ARG0", "ARG1", "ARG2"],
    ent_labels: List[str] = ["PERSON", "NORP", "ORG", "GPE", "EVENT"],
    top_n_entities: Optional[int] = None,
    dimension_reduce_verbs: Optional[bool] = True,
    progress_bar: bool = False,
):

    """

    A wrapper function to build the narrative model from a sample of the corpus.

    Args:
        srl_res: sentences labeled with their semantic roles
        sentences: list of sentences
        roles_considered: list of semantic roles to consider
        output_path: path to save the narrative model (default is None, which means no saving to disk)
        preprocessing_options: see clean_text() function
        roles_with_embeddings: list of lists of semantic roles to embed and cluster
        (i.e. each list represents semantic roles that should be clustered together)
        embeddings_type: whether the user wants to use USE / Keyed Vectors or a custom pre-trained Word2Vec
        (e.g. "USE" / "gensim_keyed_vectors" / "gesim_full_model")
        embeddings_path: path for the trained embeddings model
        n_clusters: number of clusters for the clustering model
        verbose: see sklearn.KMeans documentation for details
        roles_with_entities: list of semantic roles with named entities
        ent_labels: list of entity labels to be considered (see SPaCy documentation)
        top_n_entities: number of named entities to keep (default is all and is specified with top_n = 0)
        dimension_reduce_verbs: if True, verbs are replaced by their most frequent synonyms/antonyms
        progress_bar: print a progress bar (default is False)

    Returns:
        A dictionary with the details of the pipeline to extract narratives from text

    """

    # Sanity checks
    if len(srl_res) != len(sentences):
        raise ValueError("srl_res should be the same length sentences.")

    if (
        is_subsequence(
            roles_considered,
            ["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
        )
        is False
    ):
        raise ValueError("Some roles_considered are not supported.")

    if is_subsequence(["ARG0", "B-V", "B-ARGM-NEG", "ARG1"], roles_considered) is False:
        raise ValueError(
            "Minimum roles to consider: ['ARG0', 'B-V', 'B-ARGM-NEG', 'ARG1']"
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

    if roles_with_embeddings is not None:
        if embeddings_type not in ["gensim_keyed_vectors", "gensim_full_model", "USE"]:
            raise TypeError(
                "Only three types of embeddings accepted: gensim_keyed_vectors, gensim_full_model, USE"
            )

    if is_subsequence(ent_labels, ["PERSON", "NORP", "ORG", "GPE", "EVENT"]) is False:
        raise ValueError("Some ent_labels are not supported.")

    if lemmatize is True and stem is True:
        raise ValueError("lemmatize and stemming cannot be both True")

    # Narrative model dictionary
    narrative_model = {}

    narrative_model["roles_considered"] = roles_considered
    narrative_model["roles_with_entities"] = roles_with_entities
    narrative_model["roles_with_embeddings"] = roles_with_embeddings
    narrative_model["dimension_reduce_verbs"] = dimension_reduce_verbs
    narrative_model["clean_text_options"] = {
        "max_length": max_length,
        "remove_punctuation": remove_punctuation,
        "remove_digits": remove_digits,
        "remove_chars": remove_chars,
        "stop_words": stop_words,
        "lowercase": lowercase,
        "strip": strip,
        "remove_whitespaces": remove_whitespaces,
        "lemmatize": lemmatize,
        "stem": stem,
        "tags_to_keep": tags_to_keep,
        "remove_n_letter_words": remove_n_letter_words,
    }

    # Process SRL
    roles, sentence_index = extract_roles(
        srl_res, used_roles=roles_considered, progress_bar=progress_bar
    )

    if (output_path is not None) and os.path.isfile(
        "%spostproc_roles.json" % output_path
    ):
        with open("%spostproc_roles.json" % output_path, "r") as f:
            postproc_roles = json.load(f)
    else:
        postproc_roles = process_roles(
            roles,
            max_length,
            remove_punctuation,
            remove_digits,
            remove_chars,
            stop_words,
            lowercase,
            strip,
            remove_whitespaces,
            lemmatize,
            stem,
            tags_to_keep,
            remove_n_letter_words,
            progress_bar=progress_bar,
        )

    if output_path is not None:
        with open("%spostproc_roles.json" % output_path, "w") as f:
            json.dump(postproc_roles, f)

    # Verb Counts
    if dimension_reduce_verbs:

        if (output_path is not None) and os.path.isfile(
            "%sverb_counts.pk" % output_path
        ):
            with open(output_path + "verb_counts.pk", "rb") as f:
                verb_counts = pk.load(f)
        else:
            verb_counts = count_values(
                postproc_roles, keys=["B-V"], progress_bar=progress_bar
            )

        if output_path is not None:
            with open("%sverb_counts.pk" % output_path, "wb") as f:
                pk.dump(verb_counts, f)

        narrative_model["verb_counts"] = verb_counts

    # Named Entities
    if roles_with_entities is not None:

        if (output_path is not None) and os.path.isfile("%sentities.pk" % output_path):
            with open("%sentities.pk" % output_path, "rb") as f:
                entities = pk.load(f)
        else:
            entities = mine_entities(
                sentences=sentences,
                ent_labels=ent_labels,
                remove_punctuation=remove_punctuation,
                remove_digits=remove_digits,
                remove_chars=remove_chars,
                stop_words=stop_words,
                lowercase=lowercase,
                strip=strip,
                remove_whitespaces=remove_whitespaces,
                lemmatize=lemmatize,
                stem=stem,
                tags_to_keep=tags_to_keep,
                remove_n_letter_words=remove_n_letter_words,
                progress_bar=progress_bar,
            )

        if output_path is not None:
            with open("%sentities.pk" % output_path, "wb") as f:
                pk.dump(entities, f)

        entity_index, postproc_roles = map_entities(
            statements=postproc_roles,
            entities=entities,
            used_roles=roles_with_entities,
            top_n_entities=top_n_entities,
            progress_bar=progress_bar,
        )

        narrative_model["entities"] = entities

    # Embeddings and clustering
    if roles_with_embeddings is not None:
        sentences = clean_text(
            sentences,
            remove_punctuation,
            remove_digits,
            remove_chars,
            stop_words,
            lowercase,
            strip,
            remove_whitespaces,
            lemmatize,
            stem,
            tags_to_keep,
            remove_n_letter_words,
        )

        if progress_bar:
            print("Loading embeddings model...")

        if embeddings_type == "gensim_keyed_vectors":
            model = SIF_keyed_vectors(path=embeddings_path, sentences=sentences)
        elif embeddings_type == "gensim_full_model":
            model = SIF_word2vec(path=embeddings_path, sentences=sentences)
        elif embeddings_type == "USE":
            model = USE(path=embeddings_path)

        narrative_model["embeddings_model"] = model

        narrative_model["cluster_model"] = []
        narrative_model["cluster_labels_most_similar"] = []
        narrative_model["cluster_labels_most_freq"] = []

        for i, roles in enumerate(roles_with_embeddings):

            labels_most_similar_list = []
            kmeans_list = []
            labels_most_freq_list = []

            vecs = get_vectors(postproc_roles, model, used_roles=roles)

            for num in n_clusters[i]:

                if (output_path is not None) and os.path.isfile(
                    output_path + "kmeans_%s_%s.pk" % (i, num)
                ):
                    with open(output_path + "kmeans_%s_%s.pk" % (i, num), "rb") as f:
                        kmeans = pk.load(f)
                else:
                    kmeans = train_cluster_model(
                        vecs,
                        model,
                        n_clusters=num,
                        verbose=verbose,
                        random_state=random_state,
                    )

                if output_path is not None:
                    with open(output_path + "kmeans_%s_%s.pk" % (i, num), "wb") as f:
                        pk.dump(kmeans, f)

                clustering_res = get_clusters(
                    postproc_roles, model, kmeans, used_roles=roles, suffix=""
                )

                labels_most_freq = label_clusters_most_freq(
                    clustering_res=clustering_res, postproc_roles=postproc_roles
                )

                if isinstance(model, (USE)) is False:
                    labels_most_similar = label_clusters_most_similar(kmeans, model)
                    labels_most_similar_list.append(labels_most_similar)

                kmeans_list.append(kmeans)
                labels_most_freq_list.append(labels_most_freq)

            narrative_model["cluster_labels_most_similar"].append(
                labels_most_similar_list
            )
            narrative_model["cluster_model"].append(kmeans_list)
            narrative_model["cluster_labels_most_freq"].append(labels_most_freq_list)

    if output_path is not None:
        with open(output_path + "narrative_model.pk", "wb") as f:
            pk.dump(narrative_model, f)

    return narrative_model


def get_narratives(
    srl_res: List[dict],
    doc_index: List[int],
    narrative_model: dict,
    n_clusters: List[int],  # k means model you want to use
    output_path: Optional[str] = None,
    cluster_labeling: Optional[str] = "most_frequent",
    progress_bar: bool = False,
):

    """

    A wrapper function to obtain the final mined narratives.

    Args:
        srl_res: sentences labeled with their semantic roles
        doc_index: list of indices to keep track of original documents
        narrative_model: dict with the specifics of the narrative model
        output_path: path to save the narrative model (default is None, which means no saving to disk)
        filter_complete_narratives: keep only narratives with at least an agent, a verb and a patient
        (default is True)
        cluster_labeling: either 'most_frequent' or 'most_similar'
        progress_bar: print a progress bar (default is False)

    Returns:
        A pandas dataframe with the resulting narratives

    """

    # Sanity checks
    if cluster_labeling not in ["most_similar", "most_frequent"]:
        raise ValueError("cluster_labeling is either most_similar or most_frequent.")

    if cluster_labeling == "most_similar" and isinstance(
        narrative_model["embeddings_model"], USE
    ):
        raise ValueError(
            "most_similar option is not implemented for Universal Sentence Encoders. Consider switching to other embedding types."
        )

    # Process SRL
    roles, sentence_index = extract_roles(
        srl_res,
        used_roles=narrative_model["roles_considered"],
        progress_bar=progress_bar,
    )

    postproc_roles = process_roles(
        roles,
        narrative_model["clean_text_options"]["max_length"],
        narrative_model["clean_text_options"]["remove_punctuation"],
        narrative_model["clean_text_options"]["remove_digits"],
        narrative_model["clean_text_options"]["remove_chars"],
        narrative_model["clean_text_options"]["stop_words"],
        narrative_model["clean_text_options"]["lowercase"],
        narrative_model["clean_text_options"]["strip"],
        narrative_model["clean_text_options"]["remove_whitespaces"],
        narrative_model["clean_text_options"]["lemmatize"],
        narrative_model["clean_text_options"]["stem"],
        narrative_model["clean_text_options"]["tags_to_keep"],
        narrative_model["clean_text_options"]["remove_n_letter_words"],
        progress_bar=progress_bar,
    )

    final_statements = rename_arguments(postproc_roles, progress_bar, suffix="_highdim")

    # Dimension reduction of verbs
    if narrative_model["dimension_reduce_verbs"]:
        cleaned_verbs = clean_verbs(
            postproc_roles,
            narrative_model["verb_counts"],
            progress_bar,
            suffix="_lowdim",
        )

        for i, statement in enumerate(cleaned_verbs):
            for role, value in statement.items():
                final_statements[i][role] = value

    # Named Entities
    if narrative_model["roles_with_entities"] is not None:
        entity_index, postproc_roles = map_entities(
            statements=postproc_roles,
            entities=narrative_model["entities"],
            used_roles=narrative_model["roles_with_entities"],
            progress_bar=progress_bar,
        )

        for role in narrative_model["roles_with_entities"]:
            for token, indices in entity_index[role].items():
                for index in indices:
                    final_statements[index][str(role + "_lowdim")] = token

    # Embeddings
    if narrative_model["roles_with_embeddings"] is not None:

        for l, roles in enumerate(narrative_model["roles_with_embeddings"]):

            clustering_res = get_clusters(
                postproc_roles,
                narrative_model["embeddings_model"],
                narrative_model["cluster_model"][l][n_clusters[l]],
                used_roles=roles,
                progress_bar=progress_bar,
                suffix="_lowdim",
            )

            if cluster_labeling == "most_frequent":
                for i, statement in enumerate(clustering_res):
                    for role, cluster in statement.items():
                        final_statements[i][role] = narrative_model[
                            "cluster_labels_most_freq"
                        ][l][n_clusters[l]][cluster]

            if cluster_labeling == "most_similar":
                for i, statement in enumerate(clustering_res):
                    for role, cluster in statement.items():
                        final_statements[i][role] = narrative_model[
                            "cluster_labels_most_similar"
                        ][l][n_clusters[l]][cluster]

    # Original sentence and document
    for i, index in enumerate(sentence_index):
        final_statements[i]["sentence"] = index
        final_statements[i]["doc"] = doc_index[index]

    final_statements = pd.DataFrame(final_statements)
    final_statements["statement"] = final_statements.index
    final_statements = final_statements.replace({np.NaN: ""})
    colnames_ordered = [
        "doc",
        "sentence",
        "statement",
        "ARG0_highdim",
        "ARG0_lowdim",
        "B-V_highdim",
        "B-V_lowdim",
        "B-ARGM-NEG_highdim",
        "B-ARGM-NEG_lowdim",
        "B-ARGM-MOD_highdim",
        "ARG1_highdim",
        "ARG1_lowdim",
        "ARG2_highdim",
        "ARG2_lowdim",
    ]
    colnames = [col for col in colnames_ordered if col in list(final_statements)]
    final_statements = final_statements[colnames]

    if output_path is not None:
        final_statements.to_csv(output_path, index=False)

    return final_statements
