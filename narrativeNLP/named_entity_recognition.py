# Named Entity Recognition
# ..................................................................................................................
# ..................................................................................................................

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Any
from collections import Counter
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")
import time

from .utils import clean_text, is_subsequence


def mine_entities(
    sentences: List[str],
    ent_labels: Optional[List[str]] = ["PERSON", "NORP", "ORG", "GPE", "EVENT"],
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
    progress_bar: Optional[bool] = True,
) -> List[Tuple[str, int]]:

    """

    A function that goes through sentences and counts named entities found in the corpus.

    Args:
        sentences: list of sentences
        ent_labels: list of entity labels to be considered (see SPaCy documentation)
        progress_bar: print a progress bar (default is False)
        For other arguments see utils.clean_text.

    Returns:
        List of tuples with the named entity and its associated frequency on the corpus

    """

    entities_all = []

    if progress_bar == True:
        print("Mining named entities...")
        time.sleep(1)
        sentences = tqdm(sentences)

    for sentence in sentences:
        sentence = nlp(str(sentence))
        for ent in sentence.ents:
            if ent.label_ in ent_labels:
                entity = [ent.text]
                entities_all = entity + entities_all

    entities_all = clean_text(
        entities_all,
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

    entity_counts = Counter(entities_all)
    entities_sorted = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)

    # forgetting to remove those will break the pipeline
    entities_sorted = [entity for entity in entities_sorted if entity[0] != ""]

    return entities_sorted


def pick_top_entities(
    entities_sorted: List[Tuple[str, int]], top_n_entities: Optional[int] = 0
) -> List[str]:

    """

    A function that returns the top n most frequent named entities in the corpus.

    Args:
        entities_sorted: list of tuples (named_entity, frequency)
        top_n_entities: number of named entities to keep (default is all and is specified with top_n = 0)

    Returns:
        List of most frequent named entities

    """

    entities = []

    for entity in entities_sorted:
        entities = entities + [entity[0]]

    if top_n_entities == 0:
        top_n_entities = len(entities_sorted)

    return entities[0:top_n_entities]


def map_entities(  # the output could be a list of dictionaries (for consistency with the rest of the pipeline)
    statements: List[dict],
    entities: list,
    UsedRoles: List[str],
    progress_bar: Optional[bool] = False,
) -> Tuple[dict, List[dict]]:

    """

    A function that goes through statements and identifies pre-defined named entities within postprocessed semantic roles.

    Args:
        statements: list of dictionaries of postprocessed semantic roles
        entities: user-defined list of named entities
        roles: a list of roles with named entities (default = ARG0 and ARG1)
        UsedRoles: list of roles for named entity recognition
        progress_bar: print a progress bar (default is False)

    Returns:
        entity_index: dictionary containing statements indices with entities for each role
        roles_copy: new list of postprocessed semantic roles (without the named entities mined since they will not be embedded)

    """

    entity_index = {
        role: {entity: np.asarray([], dtype=int) for entity in entities}
        for role in UsedRoles
    }

    roles_copy = deepcopy(statements)

    if progress_bar == True:
        print("Mapping named entities...")
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, tokens in roles_copy[i].items():
            if role in UsedRoles:
                for entity in entities:
                    if is_subsequence(entity.split(), tokens) == True:
                        entity_index[role][entity] = np.append(
                            entity_index[role][entity], [i]
                        )
                        roles_copy[i][role] = []

    return entity_index, roles_copy
