# Utils
# ..................................................................................................................
# ..................................................................................................................

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Any
from tqdm import tqdm
import json
import re
import string
import spacy

nlp = spacy.load("en_core_web_sm")
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import time


def split_into_sentences(
    dataframe,
    save_to_disk: Optional[str] = None,
    progress_bar: Optional[bool] = False,
):

    """

    A function that splits a list of documents into sentences (using the SpaCy sentence splitter).

    Args:
        dataframe: a pandas dataframe with one column "id" and one column "doc"
        progress_bar: print a progress bar (default is False)

    Returns:
        List of document ids and list of sentences

    """

    list_of_docs = dataframe.to_dict(orient="records")

    sentences = []
    doc_indices = []

    if progress_bar == True:
        print("Splitting into sentences...")
        time.sleep(1)
        list_of_docs = tqdm(list_of_docs)

    for doc_info in list_of_docs:
        for sent in nlp(doc_info["doc"]).sents:
            sent = str(sent)
            sentences = sentences + [sent]
            doc_indices = doc_indices + [doc_info["id"]]

    if save_to_disk is not None:
        with open(save_to_disk, "w") as f:
            json.dump((doc_indices, sentences), f)

    return (doc_indices, sentences)


def remove_extra_whitespaces(s: str) -> str:

    res = " ".join(s.split())

    return res


def _get_wordnet_pos(word):
    """Get POS tag"""
    tag = pos_tag([word])[0][1][0].upper()

    return tag


wnl = WordNetLemmatizer()
f_lemmatize = wnl.lemmatize


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
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
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
        tags_to_keep: list of grammatical tags to keep (common tags: ['V', 'N', 'J'])
        remove_n_letter_words: drop words lesser or equal to n letters (default is None)
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
        >>> preprocess(['This is a sentence with verbs and nice adjectives.'], tags_to_keep = ['V', 'J'])
        ['is nice']
        >>> preprocess(['This is a sentence with one and two letter words.'], remove_n_letter_words = 2)
        ['this sentence with one and two letter words']
    """
    if lemmatize is True and stem is True:
        raise ValueError("lemmatize and stemming cannot be both True")
    if stop_words is not None and lowercase is False:
        raise ValueError("remove stop words make sense only for lowercase")

    # remove chars
    if remove_punctuation:
        remove_chars += string.punctuation
    if remove_digits:
        remove_chars += string.digits
    if remove_chars:
        sentences = [re.sub(f"[{remove_chars}]", "", str(sent)) for sent in sentences]

    # lowercase, strip and remove superfluous white spaces
    if lowercase:
        sentences = [sent.lower() for sent in sentences]
    if strip:
        sentences = [sent.strip() for sent in sentences]
    if remove_whitespaces:
        sentences = [" ".join(sent.split()) for sent in sentences]

    # lemmatize
    if lemmatize:

        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

        sentences = [
            " ".join(
                [
                    f_lemmatize(
                        word, tag_dict.get(_get_wordnet_pos(word), wordnet.NOUN)
                    )
                    for word in sent.split()
                ]
            )
            for sent in sentences
        ]

    # keep specific nltk tags
    # this step should be performed before stemming, but may be performed after lemmatization
    if tags_to_keep is not None:
        sentences = [
            " ".join(
                [
                    word
                    for word in sent.split()
                    if _get_wordnet_pos(word) in tags_to_keep
                ]
            )
            for sent in sentences
        ]

    # stem
    if stem:
        stemmer = SnowballStemmer("english")
        f_stem = stemmer.stem

        sentences = [
            " ".join([f_stem(word) for word in sent.split()]) for sent in sentences
        ]

    # drop stopwords
    # stopwords are dropped after the bulk of preprocessing steps, so they should also be preprocessed with the same standards
    if stop_words is not None:
        sentences = [
            " ".join([word for word in sent.split() if word not in stop_words])
            for sent in sentences
        ]

    # remove short words < n
    if remove_n_letter_words is not None:
        sentences = [
            " ".join(
                [word for word in sent.split() if len(word) > remove_n_letter_words]
            )
            for sent in sentences
        ]

    return sentences


def is_subsequence(v2: list, v1: list) -> bool:

    """

    Check whether v2 is a subsequence of v1.

    Args:
        v2/v1: lists of elements

    Returns:
        a boolean

    Example:
        >>> v1 = ['the', 'united', 'states', 'of', 'america']\n
        ... v2 = ['united', 'states', 'of', 'europe']\n
        ... is_subsequence(v2,v1)
        False

    """
    it = iter(v1)
    return all(c in it for c in v2)
