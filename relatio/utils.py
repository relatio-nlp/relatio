import json
import pickle as pk
import time
from collections import Counter
from typing import Dict, List, Optional


def replace_sentences(
    sentences: List[str],
    max_sentence_length: Optional[int] = None,
    max_number_words: Optional[int] = None,
) -> List[str]:
    """

    Replace long sentences in list of sentences by empty strings.

    Args:
        sentences: list of sentences
        max_sentence_length: Keep only sentences with a a number of character lower or equal to max_sentence_length. For max_number_words = max_sentence_length = -1 all sentences are kept.
        max_number_words: Keep only sentences with a a number of words lower or equal to max_number_words. For max_number_words = max_sentence_length = -1 all sentences are kept.

    Returns:
        Replaced list of sentences.

    Examples:
        >>> replace_sentences(['This is a house'])
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_sentence_length=15)
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_sentence_length=14)
        ['']
        >>> replace_sentences(['This is a house'], max_number_words=4)
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_number_words=3)
        ['']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=18)
        ['This is a house', 'It is a nice house']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=4, max_sentence_length=18)
        ['This is a house', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=17)
        ['This is a house', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=0, max_sentence_length=18)
        ['', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=0)
        ['', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'])
        ['This is a house', 'It is a nice house']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=4)
        ['This is a house', '']

    """

    if max_sentence_length is not None:
        sentences = [
            "" if (len(sent) > max_sentence_length) else sent for sent in sentences
        ]

    if max_number_words is not None:
        sentences = [
            "" if (len(sent.split()) > max_number_words) else sent for sent in sentences
        ]

    return sentences


def group_sentences_in_batches(
    sentences: List[str],
    max_batch_char_length: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> List[List[str]]:
    """

    Group sentences in batches of given total character length or size (number of sentences).

    In case a sentence is longer than max_batch_char_length it is replaced with an empty string.

    Args:
        sentences: List of sentences
        max_batch_char_length: maximum char length for a batch
        batch_size: number of sentences

    Returns:
        List of batches (list) of sentences.

    Examples:
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=15)
        [['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=14)
        [['', '']]
        >>> group_sentences_in_batches(['This is a house','This is a house', 'This is not a house'], max_batch_char_length=15)
        [['This is a house'], ['This is a house', '']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=29)
        [['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=30)
        [['This is a house', 'This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'])
        [['This is a house', 'This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], max_batch_char_length=29)
        [['This is a house'], ['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], max_batch_char_length=30)
        [['This is a house', 'This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], batch_size=2)
        [['This is a house', 'This is a house'], ['This is a house']]

    """

    batches: List[List[str]] = []

    if max_batch_char_length is not None and batch_size is not None:
        raise ValueError("max_batch_char_length and batch_size are mutually exclusive.")
    elif max_batch_char_length is not None:
        # longer sentences are replaced with an empty string
        sentences = replace_sentences(
            sentences, max_sentence_length=max_batch_char_length
        )
        batch_char_length = 0
        batch: List[str] = []

        for el in sentences:
            length = len(el)
            batch_char_length += length
            if batch_char_length > max_batch_char_length:
                batches.append(batch)
                batch = [el]
                batch_char_length = length
            else:
                batch.append(el)

        if batch:
            batches.append(batch)

    elif batch_size is not None:
        batches = [
            sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)
        ]
    else:
        batches = [sentences]

    return batches


def is_subsequence(v1: list, v2: list) -> bool:
    """

    Check whether v1 is a subset of v2.

    Args:
        v1: lists of elements
        v2: list of elements

    Returns:
        a boolean

    Example:
        >>> is_subsequence(['united', 'states', 'of', 'europe'],['the', 'united', 'states', 'of', 'america'])
        False
        >>> is_subsequence(['united', 'states', 'of'],['the', 'united', 'states', 'of', 'america'])
        True

    """
    # TODO: Check whether the order of elements matter, e.g. is_subsequence(["A","B"],["B","A"])
    return set(v1).issubset(set(v2))


def count_values(
    dicts: List[Dict], keys: Optional[list] = None, progress_bar: bool = False
) -> Counter:
    """

    Get a counter with the values of a list of dictionaries, with the conssidered keys given as argument.

    Args:
        dicts: list of dictionaries
        keys: keys to consider
        progress_bar: print a progress bar (default is False)

    Returns:
        Counter

    Example:
        >>> count_values([{'B-V': 'increase', 'B-ARGM-NEG': True},{'B-V': 'decrease'},{'B-V': 'decrease'}],keys = ['B-V'])
        Counter({'decrease': 2, 'increase': 1})
        >>> count_values([{'B-V': 'increase', 'B-ARGM-NEG': True},{'B-V': 'decrease'},{'B-V': 'decrease'}])
        Counter()

    """

    counts: Dict[str, int] = {}

    if progress_bar:
        print("Computing role frequencies...")
        time.sleep(1)
        dicts = dicts

    if keys is None:
        return Counter()

    for el in dicts:
        for key, value in el.items():
            if key in keys:
                if value in counts:
                    counts[value] += 1
                else:
                    counts[value] = 1

    return Counter(counts)


def count_words(sentences: List[str]) -> Counter:
    """

    A function that computes word frequencies in a list of sentences.

    Args:
        sentences: list of sentences

    Returns:
        Counter {"word": frequency}

    Example:
    >>> count_words(["this is a house"])
    Counter({'this': 1, 'is': 1, 'a': 1, 'house': 1})
    >>> count_words(["this is a house", "this is a house"])
    Counter({'this': 2, 'is': 2, 'a': 2, 'house': 2})
    >>> count_words([])
    Counter()
    """

    words: List[str] = []

    for sentence in sentences:
        words.extend(sentence.split())

    words_counter = Counter(words)

    return words_counter


def make_list_from_key(key, list_of_dicts):
    """

    Extract the content of a specific key in a list of dictionaries.
    Returns a list and the corresponding indices.

    """

    list_from_key = []
    indices = []

    for i, statement in enumerate(list_of_dicts):
        content = statement.get(key)
        if content is not None:
            list_from_key.append(content)
            indices.append(i)

    return indices, list_from_key


def get_element(narrative, role):
    return narrative[role] if role in narrative else ""


def prettify(narrative) -> str:
    """
    Takes a narrative statement dictionary and returns a pretty string.

    Args:
        narrative: a dictionary with the following keys: "ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"

    Returns:
        a concatenated string of text
    """

    ARG0 = get_element(narrative, "ARG0")
    V = get_element(narrative, "B-V")

    NEG = get_element(narrative, "B-ARGM-NEG")
    if NEG is True:
        NEG = "!"
    elif NEG is False:
        NEG = ""

    MOD = get_element(narrative, "B-ARGM-MOD")
    ARG1 = get_element(narrative, "ARG1")
    ARG2 = get_element(narrative, "ARG2")

    pretty_narrative = (ARG0, MOD, NEG, V, ARG1, ARG2)

    pretty_narrative = " ".join([t for t in pretty_narrative if t != ""])

    return pretty_narrative


def save_entities(entity_counts, output_path: str):
    """
    Save the entity counts to a pickle file.
    """
    with open(output_path, "wb") as f:
        pk.dump(entity_counts, f)


def load_entities(input_path: str):
    """
    Load the entity counts from a pickle file.
    """
    with open(input_path, "rb") as f:
        entity_counts = pk.load(f)

    return entity_counts


def save_roles(roles, output_path):
    """
    Save the roles to a json file.
    """
    with open(output_path, "w") as f:
        json.dump(roles, f)


def load_roles(input_path):
    """
    Load the roles from a json file.
    """
    with open(input_path, "r") as f:
        roles = json.load(f)

    return roles
