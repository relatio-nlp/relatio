import re
import string
import warnings
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize


def dict_concatenate(d_list, axis=0):
    d_non_empty = [d for d in d_list if d]
    res = {}
    if not d_non_empty:
        pass
    elif len(d_non_empty) == 1:
        res = d_non_empty[0]
    else:
        for k in d_non_empty[0].keys():
            d_to_concat = [d[k] for d in d_non_empty if d[k].size != 0]
            if not d_to_concat:
                res[k] = d_non_empty[0][k].copy()
            else:
                res[k] = np.concatenate(d_to_concat, axis=axis)
    return res


def tokenize_into_sentences(document: str) -> List[str]:
    """
    Split a document in sentences.

    Args:
        document: The document

    Returns:
        List of sentences

    """
    sentences = sent_tokenize(document)
    return sentences


def filter_sentences(
    sentences: List[str],
    max_sentence_length: Optional[int] = None,
    max_number_words: Optional[int] = None,
) -> List[str]:
    """
    Filter list of sentences based on the number of characters length.

    Args:
        max_sentence_length: Keep only sentences with a a number of character lower or equal to max_sentence_length. For max_number_words = max_sentence_length = -1 all sentences are kept.
        max_number_words: Keep only sentences with a a number of words lower or equal to max_number_words. For max_number_words = max_sentence_length = -1 all sentences are kept.

    Returns:
        Filtered list of sentences.

    Examples:
        >>> filter_sentences(['This is a house'])
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_sentence_length=15)
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_sentence_length=14)
        []
        >>> filter_sentences(['This is a house'], max_number_words=4)
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_number_words=3)
        []
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=18)
        ['This is a house', 'It is a nice house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=4, max_sentence_length=18)
        ['This is a house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=17)
        ['This is a house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=0, max_sentence_length=18)
        []
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=0)
        []
        >>> filter_sentences(['This is a house', 'It is a nice house'])
        ['This is a house', 'It is a nice house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=4)
        ['This is a house']
    """

    if max_sentence_length is None and max_number_words is None:
        pass
    elif max_sentence_length == 0 or max_number_words == 0:
        sentences = []
    else:
        if max_sentence_length is not None:
            sentences = [sent for sent in sentences if len(sent) <= max_sentence_length]

            def filter_funct(sent):
                return len(sent) <= max_sentence_length

        if max_number_words is not None:
            sentences = [
                sent for sent in sentences if len(sent.split()) <= max_number_words
            ]

    return sentences


def group_sentences_in_batches(
    sentences: List[str],
    max_batch_char_length: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> List[List[str]]:
    """
    Group sentences in batches of given total character length.

    Args:
        sentences: List of sentences
        max_batch_char_length: maximum char length for a batch

    Returns:
        List of batches (list) of sentences.

    Examples:
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=15)
        [['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=14)
        []
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=29)
        [['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=30)
        [['This is a house', 'This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'])
        [['This is a house', 'This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], max_batch_char_length=29)
        [['This is a house'], ['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], batch_size=2)
        [['This is a house', 'This is a house'], ['This is a house']]
    """
    batches: List[List[str]] = []

    if max_batch_char_length is None and batch_size is None:
        batches = [sentences]
    elif max_batch_char_length is not None and batch_size is not None:
        raise ValueError("max_batch_char_length and batch_size are mutual exclusive.")
    elif batch_size is not None:
        batches = [
            sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)
        ]
    else:
        batch_char_length = 0
        batch: List[str] = []

        for el in sentences:
            length = len(el)
            batch_char_length += length
            if length > max_batch_char_length:
                warnings.warn(
                    f"The length of the sentence = {length} > max_batch_length={max_batch_char_length}. The following sentence is skipped: \n > {el}",
                    RuntimeWarning,
                )
                continue
            if batch_char_length > max_batch_char_length:
                batches.append(batch)
                batch = [el]
                batch_char_length = length
            else:
                batch.append(el)

        if batch:
            batches.append(batch)

    return batches


def _get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def preprocess(
    sentences: List[str],
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    stop_words: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_whitespaces: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
) -> List[str]:
    """
    Preprocess a list of sentences for word embedding.

    Args:
        sentence: list of sentences
        remove_punctuation: whether to remove string.punctuation
        remove_digits: whether to remove string.digits
        remove_chars: remove the given characters
        stop_words: list of stopwords to remove
        lowercase: whether to lower the case
        strip: whether to strip
        remove_whitespaces: whether to remove superfluous whitespaceing by " ".join(str.split(())
        lemmatize: whether to lemmatize using nltk.WordNetLemmatizer
        stem: whether to stem using nltk.SnowballStemmer("english")
    Returns:
        Processed list of sentences

    Examples:
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'])
        ['return the factorial of n an exact integer']
        >>> preprocess(['Learning is usefull.'])
        ['learning is usefull']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], stop_words=['factorial'])
        ['return the of n an exact integer']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], lemmatize=True)
        ['return the factorial of n an exact integer']
        >>> preprocess(['Learning is usefull.'],lemmatize=True)
        ['learn be usefull']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], stem=True)
        ['return the factori of n an exact integ']
        >>> preprocess(['Learning is usefull.'],stem=True)
        ['learn is useful']
        >>> preprocess(['A1b c\\n\\nde \\t fg\\rkl\\r\\n m+n'])
        ['ab c de fg kl mn']
    """
    if lemmatize is True and stem is True:
        raise ValueError("lemmatize and stemming cannot be both True")
    if stop_words is not None and lowercase is False:
        raise ValueError("remove stop words make sense only for lowercase")

    # remove chars
    if remove_punctuation is True:
        remove_chars += string.punctuation
    if remove_digits is True:
        remove_chars += string.digits
    if remove_chars:
        sentences = [re.sub(f"[{remove_chars}]", "", sent) for sent in sentences]

    # lowercase, strip and remove superfluous white spaces
    if lowercase:
        sentences = [sent.lower() for sent in sentences]
    if strip:
        sentences = [sent.strip() for sent in sentences]
    if remove_whitespaces:
        sentences = [" ".join(sent.split()) for sent in sentences]

    if lemmatize:
        wnl = WordNetLemmatizer()
        f_lemmatize = wnl.lemmatize

        sentences = [
            " ".join(
                [f_lemmatize(word, _get_wordnet_pos(word)) for word in sent.split()]
            )
            for sent in sentences
        ]

    if stem:
        stemmer = SnowballStemmer("english")
        f_stem = stemmer.stem

        sentences = [
            " ".join([f_stem(word) for word in sent.split()]) for sent in sentences
        ]
    if stop_words is not None:
        sentences = [
            " ".join([word for word in sent.split() if word not in stop_words])
            for sent in sentences
        ]
    # TODO ignore stopwords
    # TODO input singledispatch

    return sentences


class UsedRoles:
    """
    A dict like class for used roles.

    The class has predefined keys and one cannot delete nor add new keys.

    Example:
    >>> used_roles = UsedRoles(); used_roles
    {'ARGO': True, 'ARG1': True, 'ARG2': False, 'B-V': True, 'B-ARGM-MOD': True, 'B-ARGM-NEG': True}
    >>> used_roles = UsedRoles(); used_roles["ARG2"]
    False
    >>> used_roles = UsedRoles(); used_roles["ARG2"] = True; used_roles
    {'ARGO': True, 'ARG1': True, 'ARG2': True, 'B-V': True, 'B-ARGM-MOD': True, 'B-ARGM-NEG': True}
    """

    # The order of roles is critical in other modules.
    # In cooccurrence the B-V is grouped with B-... and all roles before B-V
    # should be clustered in word embedding
    _roles = {
        "ARGO": True,
        "ARG1": True,
        "ARG2": False,
        "B-V": True,
        "B-ARGM-MOD": True,
        "B-ARGM-NEG": True,
    }
    _not_embeddable = ("B-ARGM-MOD", "B-ARGM-NEG")

    def __init__(self, roles: Optional[Dict[str, bool]] = None):
        if roles is not None:
            self.update(roles)

    def _check_key(self, key):
        if key not in self._roles.keys():
            raise ValueError(
                f"role_name {key} not in allowed set {self._roles.keys()}."
            )

    def _check_value(self, key, value):
        if not isinstance(value, bool):
            raise ValueError(
                f"role[{key}]={value} where type({value})={type(value)} but only boolean values allowed."
            )

    def __repr__(self):
        return repr(self._roles)

    def __str__(self):
        return str(self._roles)

    def __setitem__(self, role_name: str, value: bool):
        role = {role_name: value}
        self.update(role)

    def __getitem__(self, role_name):
        return self._roles[role_name]

    def __len__(self):
        return len(self._roles)

    def __iter__(self):
        return iter(self._roles)

    def items(self):
        return self._roles.items()

    def keys(self):
        return self._roles.keys()

    def values(self):
        return self._roles.values()

    def update(self, roles: Dict[str, bool]):
        for key, value in roles.items():
            self._check_key(key)
            self._check_value(key, value)
            self._roles[key] = value

    def as_dict(self):
        return self._roles.copy()

    @property
    def embeddable(self):
        role_names = []
        for el, value in self._roles.items():
            if value and (el not in self._not_embeddable):
                role_names.append(el)
        return tuple(role_names)

    @property
    def not_embeddable(self):
        return self._not_embeddable

    @property
    def used(self):
        return tuple([el for el, value in self._roles.items() if value])


class Document(NamedTuple):
    path: str
    statement_start_index: int


class DocumentTracker:
    def __init__(self, documents, sentence_index):
        self._sentence_index = sentence_index
        _df = pd.DataFrame(documents).set_index("path")
        _df["statement_end_index"] = (
            _df.squeeze().shift(-1, fill_value=sentence_index.size) - 1
        )

        ## remove empty documents
        empty_paths = _df["statement_start_index"] > _df["statement_end_index"]
        _df = _df[~empty_paths]

        _df["number_of_sentences"] = (
            sentence_index[_df.loc[:, "statement_end_index"]]
            - sentence_index[_df.loc[:, "statement_start_index"]]
        ) + 1
        self.doc = _df.reset_index()

    def find_doc(self, statement_index):
        mask = (self.doc.loc[:, "statement_start_index"] <= statement_index) & (
            statement_index <= self.doc.loc[:, "statement_end_index"]
        )
        res = self.doc[mask].copy()
        res["sentence_index_inside_doc"] = (
            self._sentence_index[statement_index]
            - self._sentence_index[res.loc[:, "statement_start_index"]]
        )
        return res
