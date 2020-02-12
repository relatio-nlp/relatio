import re
import string
from typing import List

from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer


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
        lowercase: whether to lower the case
        strip: whether to strip
        remove_whitespaces: whether to remove superfluous whitespaceing by " ".join(str.split(())
        lemmatize: whether to lemmatize using nltk.WordNetLemmatizer
        stem: whether to stem using nltk.SnowballStemmer("english", ignore_stopwords=True)
    Returns:
        Processed list of sentences

    Examples:
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'])
        ['return the factorial of n an exact integer']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], lemmatize=True)
        ['return the factorial of n an exact integer']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], stem=True)
        ['return the factori of n an exact integ']
        
        >>> preprocess(['A1b c\\n\\nde \\t fg\\rkl\\r\\n m+n'])
        ['ab c de fg kl mn']
    """
    if lemmatize is True and stem is True:
        raise ValueError("lemmatize and stemming cannot be both True")
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
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        f_stem = stemmer.stem

        sentences = [
            " ".join([f_stem(word) for word in sent.split()]) for sent in sentences
        ]

    return sentences
