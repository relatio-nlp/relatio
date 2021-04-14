# Wrappers
# ..................................................................................................................
# ..................................................................................................................

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Any
import pickle as pk
import json
import pandas as pd
import numpy as np
import os

from .utils import preprocess, is_subsequence, remove_extra_whitespaces
from .semantic_role_labeling import (
    SRL,
    extract_roles,
    postprocess_roles,
    get_role_counts,
    get_raw_arguments,
)

from .named_entity_recognition import mine_entities, pick_top_entities, map_entities
from .verbs import clean_verbs
from .clustering import (
    train_cluster_model,
    get_vectors,
    get_clusters,
    label_clusters_most_freq,
    label_clusters_most_similar,
    SIF_word2vec,
    SIF_keyed_vectors,
    USE,
)


def run_srl(
    path: str,
    sentences: List[str],
    batch_size: Optional[int] = 1,
    cuda_device: Optional[int] = -1,
    max_sentence_length: Optional[int] = None,
    max_number_words: Optional[int] = None,
    cuda_empty_cache: bool = None,
    cuda_sleep: float = None,
    save_to_disk: Optional[str] = None,
    progress_bar: Optional[bool] = False,
):

    """

    A wrapper function to run semantic role labeling on a corpus.

    Args:
        path: location of the SRL model to be used
        sentences: list of sentences
        SRL_options: see class SRL()
        save_to_disk: path to save the narrative model (default is None, which means no saving to disk)
        progress_bar: print a progress bar (default is False)

    Returns:
        A list of dictionaries with the SRL output

    """

    srl = SRL(path=path, cuda_device=cuda_device)

    srl_res = srl(
        sentences=sentences,
        batch_size=batch_size,
        max_sentence_length=max_sentence_length,
        max_number_words=max_number_words,
        cuda_empty_cache=cuda_empty_cache,
        cuda_sleep=cuda_sleep,
        progress_bar=progress_bar,
    )

    if save_to_disk is not None:
        with open(save_to_disk, "w") as json_file:
            json.dump(srl_res, json_file)

    return srl_res


def build_narrative_model(  # add more control for the user on clustering (n_jobs, random_state, etc.)
    srl_res: List[dict],
    sentences: List[str],
    roles_considered: Optional[List[str]] = [
        "ARGO",
        "B-V",
        "B-ARGM-NEG",
        "B-ARGM-MOD",
        "ARG1",
        "ARG2",
    ],
    save_to_disk: Optional[str] = None,
    max_length: Optional[int] = None,
    remove_punctuation: Optional[bool] = True,
    remove_digits: Optional[bool] = True,
    remove_chars: Optional[str] = "",
    stop_words: Optional[List[str]] = None,
    lowercase: Optional[bool] = True,
    strip: Optional[bool] = True,
    remove_whitespaces: Optional[bool] = True,
    lemmatize: Optional[bool] = False,
    stem: Optional[bool] = False,
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
    roles_with_embeddings: Optional[List[List[str]]] = [["ARGO", "ARG1", "ARG2"]],
    embeddings_type: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    n_clusters: Optional[int] = [1],
    verbose: Optional[int] = 0,
    random_state: Optional[int] = 0,
    roles_with_entities: Optional[List[str]] = ["ARGO", "ARG1", "ARG2"],
    ent_labels: Optional[List[str]] = ["PERSON", "NORP", "ORG", "GPE", "EVENT"],
    top_n_entities: Optional[int] = 0,
    dimension_reduce_verbs: Optional[bool] = True,
    progress_bar: Optional[bool] = False,
):

    """

    A wrapper function to build the narrative model from a sample of the corpus.

    Args:
        srl_res: sentences labeled with their semantic roles
        sentences: list of sentences
        roles_considered: list of semantic roles to consider
        save_to_disk: path to save the narrative model (default is None, which means no saving to disk)
        preprocessing_options: see preprocess() function
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
    if (
        is_subsequence(
            roles_considered,
            ["ARGO", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
        )
        == False
    ):
        raise ValueError("Some roles_considered are not supported.")

    if is_subsequence(["ARGO", "B-V", "B-ARGM-NEG", "ARG1"], roles_considered) == False:
        raise ValueError(
            "Minimum roles to consider: ['ARGO', 'B-V', 'B-ARGM-NEG', 'ARG1']"
        )

    if roles_with_entities is not None:
        if is_subsequence(roles_with_entities, roles_considered) == False:
            raise ValueError("roles_with_entities should be in roles_considered.")

    if roles_with_embeddings is not None:
        for roles in roles_with_embeddings:
            if is_subsequence(roles, roles_considered) == False:
                raise ValueError(
                    "each list in roles_with_embeddings should be a subset of roles_considered."
                )
            if ["B-ARGM-NEG", "B-ARGM-MOD"] in roles:
                raise ValueError(
                    "Negations and modals cannot be embedded and clustered."
                )

    if roles_with_embeddings is not None:
        if embeddings_type not in ["gensim_keyed_vectors", "gensim_full_model", "USE"]:
            raise TypeError(
                "Only three types of embeddings accepted: gensim_keyed_vectors, gensim_full_model, USE"
            )

    if is_subsequence(ent_labels, ["PERSON", "NORP", "ORG", "GPE", "EVENT"]) == False:
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
        srl_res, UsedRoles=roles_considered, progress_bar=progress_bar
    )

    if save_to_disk is not None:
        if os.path.isfile("%spostproc_roles.json" % save_to_disk):
            with open("%spostproc_roles.json" % save_to_disk, "r") as f:
                postproc_roles = json.load(f)

        else:
            postproc_roles = postprocess_roles(
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

            with open("%spostproc_roles.json" % save_to_disk, "w") as f:
                json.dump(postproc_roles, f)

    else:
        postproc_roles = postprocess_roles(
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

    # Verb Counts
    if dimension_reduce_verbs:

        if save_to_disk is not None:
            if os.path.isfile("%sverb_counts.pk" % save_to_disk):
                with open(save_to_disk + "verb_counts.pk", "rb") as f:
                    verb_counts = pk.load(f)

            else:
                verb_counts = get_role_counts(
                    postproc_roles, roles=["B-V"], progress_bar=progress_bar
                )

                with open("%sverb_counts.pk" % save_to_disk, "wb") as f:
                    pk.dump(verb_counts, f)
        else:
            verb_counts = get_role_counts(
                postproc_roles, roles=["B-V"], progress_bar=progress_bar
            )

        narrative_model["verb_counts"] = verb_counts

    # Named Entities
    if roles_with_entities is not None:

        if save_to_disk is not None:
            if os.path.isfile("%sentities_sorted.pk" % save_to_disk):
                with open("%sentities_sorted.pk" % save_to_disk, "rb") as f:
                    entities_sorted = pk.load(f)

            else:
                entities_sorted = mine_entities(
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

                with open("%sentities_sorted.pk" % save_to_disk, "wb") as f:
                    pk.dump(entities_sorted, f)

        else:
            entities_sorted = mine_entities(
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

        entities = pick_top_entities(entities_sorted, top_n_entities=top_n_entities)

        entity_index, postproc_roles = map_entities(
            statements=postproc_roles,
            entities=entities,
            UsedRoles=roles_with_entities,
            progress_bar=progress_bar,
        )

        narrative_model["entities"] = entities

    # Embeddings and clustering
    if roles_with_embeddings is not None:
        sentences = preprocess(
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

        if embeddings_type == "gensim_keyed_vectors":
            model = SIF_keyed_vectors(path=embeddings_path, sentences=sentences)
        if embeddings_type == "gensim_full_model":
            model = SIF_word2vec(path=embeddings_path, sentences=sentences)
        if embeddings_type == "USE":
            model = USE(path=embeddings_path)

        narrative_model["embeddings_model"] = model

        narrative_model["cluster_model"] = []
        narrative_model["cluster_labels_most_similar"] = []
        narrative_model["cluster_labels_most_freq"] = []

        for i, roles in enumerate(roles_with_embeddings):

            l1 = []
            l2 = []
            l3 = []

            vecs = get_vectors(postproc_roles, model, UsedRoles=roles)

            for num in n_clusters[i]:

                if save_to_disk is not None:
                    if os.path.isfile(save_to_disk + "kmeans_%s_%s.pk" % (i, num)):
                        with open(
                            save_to_disk + "kmeans_%s_%s.pk" % (i, num), "rb"
                        ) as f:
                            kmeans = pk.load(f)

                    else:
                        kmeans = train_cluster_model(
                            vecs,
                            model,
                            n_clusters=num,
                            verbose=verbose,
                            random_state=random_state,
                        )

                        with open(
                            save_to_disk + "kmeans_%s_%s.pk" % (i, num), "wb"
                        ) as f:
                            pk.dump(kmeans, f)

                else:

                    kmeans = train_cluster_model(
                        vecs,
                        model,
                        n_clusters=num,
                        verbose=verbose,
                        random_state=random_state,
                    )

                clustering_res = get_clusters(
                    postproc_roles, model, kmeans, UsedRoles=roles
                )

                labels_most_freq = label_clusters_most_freq(
                    clustering_res=clustering_res, postproc_roles=postproc_roles
                )

                if isinstance(model, (USE)) == False:
                    labels_most_similar = label_clusters_most_similar(kmeans, model)

                    l1.append(labels_most_similar)

                l2.append(kmeans)
                l3.append(labels_most_freq)

            narrative_model["cluster_labels_most_similar"].append(l1)
            narrative_model["cluster_model"].append(l2)
            narrative_model["cluster_labels_most_freq"].append(l3)

    if save_to_disk is not None:
        with open(save_to_disk + "narrative_model.pk", "wb") as f:
            pk.dump(narrative_model, f)

    return narrative_model


def get_narratives(
    srl_res: List[dict],
    doc_index: List[int],
    narrative_model: dict,
    n_clusters: List[int],  # k means model you want to use
    save_to_disk: Optional[str] = None,
    cluster_labeling: Optional[str] = "most_frequent",
    progress_bar: Optional[bool] = False,
):

    """

    A wrapper function to obtain the final mined narratives.

    Args:
        srl_res: sentences labeled with their semantic roles
        doc_index: list of indices to keep track of original documents
        narrative_model: dict with the specifics of the narrative model
        save_to_disk: path to save the narrative model (default is None, which means no saving to disk)
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
        UsedRoles=narrative_model["roles_considered"],
        progress_bar=progress_bar,
    )

    postproc_roles = postprocess_roles(
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

    final_statements = get_raw_arguments(postproc_roles, progress_bar)

    # Dimension reduction of verbs
    if narrative_model["dimension_reduce_verbs"]:
        cleaned_verbs = clean_verbs(
            postproc_roles, narrative_model["verb_counts"], progress_bar
        )

        for i, statement in enumerate(cleaned_verbs):
            for role, value in statement.items():
                final_statements[i][role] = value

    # Named Entities
    if narrative_model["roles_with_entities"] is not None:
        entity_index, postproc_roles = map_entities(
            statements=postproc_roles,
            entities=narrative_model["entities"],
            UsedRoles=narrative_model["roles_with_entities"],
            progress_bar=progress_bar,
        )

        for role in narrative_model["roles_with_entities"]:
            for token, indices in entity_index[role].items():
                for index in indices:
                    final_statements[index][role] = token

    # Embeddings
    if narrative_model["roles_with_embeddings"] is not None:

        for l, roles in enumerate(narrative_model["roles_with_embeddings"]):

            clustering_res = get_clusters(
                postproc_roles,
                narrative_model["embeddings_model"],
                narrative_model["cluster_model"][l][n_clusters[l]],
                UsedRoles=roles,
                progress_bar=progress_bar,
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

    if save_to_disk is not None:
        final_statements.to_csv(save_to_disk, index=False)

    return final_statements


def build_narratives(  # to be considered as very preliminary
    final_statements,
    narrative_model: dict,
    filter_complete_narratives: Optional[bool] = False,
):

    """

    A function to make columns of 'raw' and 'cleaned' narratives.

    Args:
        final_statements: dataframe with the output of the pipeline
        narrative_model: dict with the specifics of the narrative model
        filter_complete_narratives: keep only narratives with at least an agent, a verb and a patient
        (default is False)

    Returns:
        A pandas dataframe with the resulting narratives and two additional columns:
        narrative-RAW and narrative-CLEANED

    """

    narrative_format = [
        str(role + "-RAW") for role in narrative_model["roles_considered"]
    ]

    final_statements = final_statements.replace({"": np.NaN})

    if filter_complete_narratives:
        list_for_filter = [
            arg
            for arg in narrative_format
            if arg not in ["ARG2-RAW", "B-ARGM-NEG-RAW", "B-ARGM-MOD-RAW"]
        ]
        final_statements = final_statements.dropna(subset=list_for_filter)

    final_statements = final_statements.replace({np.NaN: ""})
    final_statements = final_statements.replace({True: "not"})

    # Check if all columns exist
    for role in narrative_format:
        if role not in final_statements.columns:
            final_statements[role] = ""

    final_statements["narrative-RAW"] = final_statements[narrative_format].agg(
        " ".join, axis=1
    )
    final_statements["narrative-RAW"] = final_statements["narrative-RAW"].apply(
        remove_extra_whitespaces
    )

    narrative_format = []
    for role in narrative_model["roles_considered"]:
        if role == "B-V":
            if narrative_model["dimension_reduce_verbs"] == True:
                narrative_format = narrative_format + ["B-V-CLEANED"]
                narrative_format = narrative_format + ["B-ARGM-NEG-CLEANED"]
            else:
                narrative_format = narrative_format + ["B-V-RAW"]
                narrative_format = narrative_format + ["B-ARGM-NEG-RAW"]

        elif role == "B-ARGM-NEG":
            continue

        elif role == "B-ARGM-MOD":
            narrative_format = narrative_format + ["B-ARGM-MOD-RAW"]

        else:
            if (
                narrative_model["roles_with_embeddings"] is not None
                or narrative_model["roles_with_entities"] is not None
            ):
                narrative_format = narrative_format + [role]
            else:
                narrative_format = narrative_format + [str(role + "-RAW")]

    final_statements["narrative-CLEANED"] = final_statements[narrative_format].agg(
        " ".join, axis=1
    )
    final_statements["narrative-CLEANED"] = final_statements["narrative-CLEANED"].apply(
        remove_extra_whitespaces
    )

    # Re-ordering columns
    columns = ["doc", "sentence", "statement", "narrative-CLEANED", "narrative-RAW"]
    for role in narrative_model["roles_considered"]:
        if role in ["ARGO", "ARG1", "ARG2"]:
            columns = columns + [str(role + "-RAW")]
            columns = columns + [role]
        elif role == "B-ARGM-MOD":
            columns = columns + [str(role + "-RAW")]
        elif role == "B-V":
            if narrative_model["dimension_reduce_verbs"] == True:
                columns = columns + [str(role + "-RAW")]
                columns = columns + [str(role + "-CLEANED")]
            else:
                columns = columns + [str(role + "-RAW")]
        elif role == "B-ARGM-NEG":
            if narrative_model["dimension_reduce_verbs"] == True:
                columns = columns + [str(role + "-RAW")]
                columns = columns + [str(role + "-CLEANED")]
            else:
                columns = columns + [str(role + "-RAW")]

    final_statements = final_statements[columns]

    return final_statements
