import re
import string
from typing import List

from nltk.tokenize import sent_tokenize


def tokenize_into_sentences(text: str) -> List[str]:
    return sent_tokenize(text)


def normalize_sentence(sentence: str) -> str:
    """
    Normalize a sentence.
    
    The function will perform several steps: l
    - remove punctuation and digits
    - lower the case, strip and remove superfluous white spaces
    
    >>> normalize_sentence(" Return the factorial of n, an  exact integer >= 0.")
    'return the factorial of n an exact integer'
    """
    # remove punctuation and digits
    sentence = re.sub(f"[{string.punctuation}+{string.digits}]", "", sentence)

    # lower the case, strip and remove superfluous white spaces
    sentence = " ".join(sentence.lower().strip().split())
    return sentence


def preprocess(text: str) -> List[str]:
    sentences = tokenize_into_sentences(text)
    return [normalize_sentence(sentence) for sentence in sentences]
