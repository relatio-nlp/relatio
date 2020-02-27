import re
import string
from typing import List, Optional, Dict

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize


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


def filter_sentences(sentences: List[str], max_sentence_length: int = -1) -> List[str]:
    """
    Filter list of sentences based on the number of characters length.

    Args:
        max_sentence_length: Keep only sentences with a a number of character lower or equal to max_sentence_length. For max_sentence_length = -1 all sentences are kept.

    Returns:
        Filtered list of sentences.

    Examples:
        >>> filter_sentences(['This is a house'])
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_sentence_length=15)
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_sentence_length=14)
        []
    """
    if max_sentence_length < -1:
        raise ValueError("max_sentence_length should be greater or equal to -1")
    elif max_sentence_length == -1:
        pass
    elif max_sentence_length == 0:
        sentences = []
    else:
        sentences = [sent for sent in sentences if len(sent) <= max_sentence_length]
    return sentences


def preprocess(
    sentences: List[str],
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    remove_stop_words: bool = False,
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
        remove_stop_words: remove stopwords.words("english")
        lowercase: whether to lower the case
        strip: whether to strip
        remove_whitespaces: whether to remove superfluous whitespaceing by " ".join(str.split(())
        lemmatize: whether to lemmatize using nltk.WordNetLemmatizer
        stem: whether to stem using nltk.SnowballStemmer("english")
    Returns:
        Processed list of sentences

    Examples:
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'])
        ['return factorial n exact integer']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], lemmatize=True)
        ['return factorial n exact integer']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], stem=True)
        ['return factori n exact integ']
        >>> preprocess(['A1b c\\n\\nde \\t fg\\rkl\\r\\n m+n'])
        ['ab c de fg kl mn']
    """
    if lemmatize is True and stem is True:
        raise ValueError("lemmatize and stemming cannot be both True")
    if remove_stop_words is True and lowercase is False:
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
            " ".join([f_lemmatize(word) for word in sent.split()]) for sent in sentences
        ]

    if stem:
        stemmer = SnowballStemmer("english")
        f_stem = stemmer.stem

        sentences = [
            " ".join([f_stem(word) for word in sent.split()]) for sent in sentences
        ]
    if remove_stop_words:
        stop_words = stopwords.words("english")
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

