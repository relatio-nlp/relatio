from typing import List
import re
import string

import nltk
from nltk.tokenize import sent_tokenize


def tokenize_into_sentences(text: str) -> List[str]:
    return sent_tokenize(text)


def _normalize_sentence(sentence: str) -> str:
    sentence = " ".join(sentence.lower().strip().split())
    sentence = re.sub(f"[{string.punctuation}+{string.digits}]", "", sentence)
    return sentence


def normalize_sentences(sentences: List[str]) -> List[str]:
    return [_normalize_sentence(sentence) for sentence in sentences]
