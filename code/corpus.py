from typing import List

from nltk.tokenize import sent_tokenize


def filter_sentences(sentences: List[str], max_sentence_length: int = -1) -> List[str]:
    """
    Filter list of sentences based on their length.

    Args:
        max_sentence_length: Keep only sentences with a length lower or equal to. For max_length = -1 all sentences are kept.

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


def tokenize_into_sentences(article: str) -> List[str]:
    """
    Split an article in sentences.

    Args:
        article: The article

    Returns:
        List of sentences

    """
    sentences = sent_tokenize(article)
    return sentences
