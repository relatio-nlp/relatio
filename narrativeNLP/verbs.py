# Clean Verbs
# ..................................................................................................................
# ..................................................................................................................

import time
from copy import deepcopy
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from nltk.corpus import wordnet
from tqdm import tqdm


def find_synonyms(verb: str) -> List[str]:

    """

    Find synonyms of a given word based on wordnet.
    Args:
        verb: a verb
    Returns:
        a list of synonyms
    Example:
        >>> find_synonyms('fight')
        ['contend', 'fight', 'struggle', 'fight', 'oppose', 'fight_back', 'fight_down', 'defend', 'fight', 'struggle', 'crusade', 'fight', 'press', 'campaign', 'push', 'agitate']

    """

    synonyms = []

    for syn in wordnet.synsets(verb, pos=wordnet.VERB):
        for l in syn.lemmas():
            synonyms.append(l.name())

    return synonyms


def find_antonyms(verb: str) -> List[str]:

    """

    Find antonyms of a given word based on wordnet.
    Args:
        verb: a verb
    Returns:
        a list of antonyms
    Example:
        >>> find_antonyms('break')
        ['repair', 'keep', 'conform_to', 'make', 'promote']

    """

    antonyms = []

    for syn in wordnet.synsets(verb, pos=wordnet.VERB):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return antonyms


def get_most_frequent(tokens: List[str], token_counts: dict) -> str:

    """

    Find most frequent token in a list of tokens.

    Args:
        tokens: a list of tokens
        token_counts: a dictionary of token frequencies
    Returns:
        the most frequent token in the list of tokens

    """

    freq = 0
    most_freq_token = None

    for candidate in tokens:
        if candidate in token_counts:
            if token_counts[candidate] > freq:
                freq = token_counts[candidate]
                most_freq_token = candidate

    return most_freq_token


def clean_verbs(
    statements: List[dict], verb_counts: dict, progress_bar: bool = False
) -> List[dict]:

    """

    Replace verbs by their most frequent synonym or antonym.
    If a verb is combined with a negation in the statement (e.g. 'not increase'),
    it is replaced by its most frequent antonym and the negation is removed (e.g. "decrease").

    Args:
        statements: a list of dictionaries of postprocessed semantic roles
        verb_counts: a dictionary of verb counts (e.g. d['verb'] = count)
        progress_bar: print a progress bar (default is False)

    Returns:
        a list of dictionaries of postprocessed semantic roles with replaced verbs (same format as statements)

    Example:
        >>> test = [{'B-V': ['increase'], 'B-ARGM-NEG': True},{'B-V': ['decrease']},{'B-V': ['decrease']}]\n
        ... verb_counts = count_values(test, roles = ['B-V'])\n
        ... clean_verbs(test, verb_counts = verb_counts)
        [{'B-V-CLEANED': 'decrease'}, {'B-V-CLEANED': 'decrease'}, {'B-V-CLEANED': 'decrease'}]

    """

    new_roles_all = []

    roles_copy = deepcopy(statements)

    if progress_bar:
        print("Cleaning verbs...")
        time.sleep(1)
        statements = tqdm(statements)

    for i, roles in enumerate(statements):
        new_roles = roles_copy[i]
        new_roles = {
            str(k + "-CLEANED"): v
            for k, v in new_roles.items()
            if k in ["B-V", "B-ARGM-NEG"]
        }
        if "B-V" in roles:
            verb = " ".join(new_roles["B-V-CLEANED"])
            new_roles["B-V-CLEANED"] = verb
            if "B-ARGM-NEG" in roles:
                verbs = find_antonyms(verb)
                most_freq_verb = get_most_frequent(
                    tokens=verbs, token_counts=verb_counts
                )
                if most_freq_verb is not None:
                    new_roles["B-V-CLEANED"] = most_freq_verb
                    del new_roles["B-ARGM-NEG-CLEANED"]
            else:
                verbs = find_synonyms(verb) + [verb]
                most_freq_verb = get_most_frequent(
                    tokens=verbs, token_counts=verb_counts
                )
                if most_freq_verb is not None:
                    new_roles["B-V-CLEANED"] = most_freq_verb
        new_roles_all.append(new_roles)

    return new_roles_all
