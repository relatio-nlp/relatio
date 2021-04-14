# Semantic Role Labeling
# ..................................................................................................................
# ..................................................................................................................

# link to choose the SRL model
# https://storage.googleapis.com/allennlp-public-models/YOUR-PREFERRED-MODEL

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Any
from copy import deepcopy
import numpy as np
import warnings
import torch
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
import json
import time

from .utils import preprocess


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
        ['']
        >>> filter_sentences(['This is a house'], max_number_words=4)
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_number_words=3)
        ['']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=18)
        ['This is a house', 'It is a nice house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=4, max_sentence_length=18)
        ['This is a house', '']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=17)
        ['This is a house', '']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=0, max_sentence_length=18)
        ['', '']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=0)
        ['', '']
        >>> filter_sentences(['This is a house', 'It is a nice house'])
        ['This is a house', 'It is a nice house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=4)
        ['This is a house', '']

    """

    if max_sentence_length is None and max_number_words is None:
        pass
    # elif max_sentence_length == 0 or max_number_words == 0:
    # sentences = []
    else:
        if max_sentence_length is not None:
            sentences = [
                "" if (len(sent) > max_sentence_length) else sent for sent in sentences
            ]

            def filter_funct(sent):
                return len(sent) <= max_sentence_length

        if max_number_words is not None:
            sentences = [
                "" if (len(sent.split()) > max_number_words) else sent
                for sent in sentences
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
        [[''], ['']]
        >>> group_sentences_in_batches(['This is a house','This is a house', 'This is not a house'], max_batch_char_length=15)
        [['This is a house'], ['This is a house'], ['']]
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

    if max_batch_char_length is None and batch_size is None:
        batches = [sentences]
    elif max_batch_char_length is not None and batch_size is not None:
        raise ValueError("max_batch_char_length and batch_size are mutually exclusive.")
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
                el = ""
            if batch_char_length > max_batch_char_length:
                if batch:
                    batches.append(batch)
                batch = [el]
                batch_char_length = length
            else:
                batch.append(el)

        if batch:
            batches.append(batch)

    return batches


class SRL:
    def __init__(
        self,
        path: str,
        cuda_device: int = -1,
        max_batch_char_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_sentence_length: Optional[int] = None,
        max_number_words: Optional[int] = None,
        cuda_empty_cache: bool = True,
        cuda_sleep: float = 0.0,
    ):
        self._predictor = Predictor.from_path(path, cuda_device=cuda_device)
        self._max_batch_char_length = max_batch_char_length
        self._batch_size = batch_size
        self._max_sentence_length = max_sentence_length
        self._max_number_words = max_number_words
        self._cuda_empty_cache = cuda_empty_cache
        self._cuda_device = cuda_device
        self._cuda_sleep = cuda_sleep

    def _clean_cache(self, cuda_sleep, cuda_empty_cache):
        if self._cuda_device > -1 and cuda_empty_cache:
            with torch.cuda.device(self._cuda_device):
                torch.cuda.empty_cache()
                time.sleep(cuda_sleep)

    def __call__(
        self,
        sentences: List[str],
        max_batch_char_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_sentence_length: Optional[int] = None,
        max_number_words: Optional[int] = None,
        cuda_empty_cache: bool = None,
        cuda_sleep: float = None,
        progress_bar: Optional[bool] = False,
    ):
        max_batch_char_length = (
            max_batch_char_length
            if max_batch_char_length is not None
            else self._max_batch_char_length
        )

        batch_size = batch_size if batch_size is not None else self._batch_size

        max_sentence_length = (
            max_sentence_length
            if max_sentence_length is not None
            else self._max_sentence_length
        )

        max_number_words = (
            max_number_words if max_number_words is not None else self._max_number_words
        )

        cuda_empty_cache = (
            cuda_empty_cache if cuda_empty_cache is not None else self._cuda_empty_cache
        )

        cuda_sleep = cuda_sleep if cuda_sleep is not None else self._cuda_sleep

        sentences = filter_sentences(
            sentences,
            max_sentence_length=max_sentence_length,
            max_number_words=max_number_words,
        )

        batches = group_sentences_in_batches(
            sentences,
            max_batch_char_length=max_batch_char_length,
            batch_size=batch_size,
        )

        res = []

        if progress_bar == True:
            print("Running SRL...")
            time.sleep(1)
            batches = tqdm(batches)

        for batch in batches:
            sentences_json = [{"sentence": sent} for sent in batch]
            try:
                res_batch = self._predictor.predict_batch_json(sentences_json)
            except RuntimeError as err:
                warnings.warn(
                    f"empty result {err}",
                    RuntimeWarning,
                )
                res = [{"words": [], "verbs": []}] * len(batch)
                break
            except:
                raise
            finally:
                self._clean_cache(cuda_sleep, cuda_empty_cache)

            res.extend(res_batch)
        return res


def extract_roles(
    srl: List[Dict[str, Any]],
    UsedRoles: List[str],
    progress_bar: Optional[bool] = False,
) -> Tuple[List[Dict[str, List]], List[int]]:

    """

    A function that extracts semantic roles from the SRL output.

    Args:
        srl: srl output
        UsedRoles: list of roles
        progress_bar: print a progress bar (default is False)

    Returns:
        List of statements and numpy array of sentence indices (to keep track of sentences)

    """

    statements_role_list: List[Dict[str, List]] = []
    sentence_index: List[int] = []

    if progress_bar == True:
        print("Processing SRL...")
        time.sleep(1)
        srl = tqdm(srl)

    for i, sentence_dict in enumerate(srl):
        role_per_sentence = extract_role_per_sentence(sentence_dict, UsedRoles)
        sentence_index.extend([i] * len(role_per_sentence))
        statements_role_list.extend(role_per_sentence)

    return statements_role_list, np.asarray(sentence_index, dtype=np.uint32)


def extract_role_per_sentence(sentence_dict: dict, UsedRoles: List[str]) -> List[dict]:

    """

    A function that extracts the semantic roles for a given sentence.

    Args:
        srl: srl output
        UsedRoles: list of roles

    Returns:
        List of statements with their associated roles for a given sentence

    """

    word_list = sentence_dict["words"]
    sentence_role_list = []

    for statement_dict in sentence_dict["verbs"]:
        tag_list = statement_dict["tags"]

        statement_role_dict = {}

        if "ARGO" in UsedRoles:
            indices_agent = [i for i, tok in enumerate(tag_list) if "ARG0" in tok]
            agent = [tok for i, tok in enumerate(word_list) if i in indices_agent]
            statement_role_dict["ARGO"] = agent

        if "ARG1" in UsedRoles:
            indices_patient = [i for i, tok in enumerate(tag_list) if "ARG1" in tok]
            patient = [tok for i, tok in enumerate(word_list) if i in indices_patient]
            statement_role_dict["ARG1"] = patient

        if "ARG2" in UsedRoles:
            indices_attribute = [i for i, tok in enumerate(tag_list) if "ARG2" in tok]
            attribute = [
                tok for i, tok in enumerate(word_list) if i in indices_attribute
            ]
            statement_role_dict["ARG2"] = attribute

        if "B-V" in UsedRoles:
            indices_verb = [i for i, tok in enumerate(tag_list) if "B-V" in tok]
            verb = [tok for i, tok in enumerate(word_list) if i in indices_verb]
            statement_role_dict["B-V"] = verb

        if "B-ARGM-MOD" in UsedRoles:
            indices_modal = [i for i, tok in enumerate(tag_list) if "B-ARGM-MOD" in tok]
            modal = [tok for i, tok in enumerate(word_list) if i in indices_modal]
            statement_role_dict["B-ARGM-MOD"] = modal

        if "B-ARGM-NEG" in UsedRoles:
            role_negation_value = any("B-ARGM-NEG" in tag for tag in tag_list)
            statement_role_dict["B-ARGM-NEG"] = role_negation_value

        key_to_delete = []
        for key, value in statement_role_dict.items():
            if not value:
                key_to_delete.append(key)
        for key in key_to_delete:
            del statement_role_dict[key]
        sentence_role_list.append(statement_role_dict)

    if not sentence_role_list:
        sentence_role_list = [{}]

    return sentence_role_list


def postprocess_roles(
    statements: List[Dict[str, List]],
    max_length: Optional[int] = None,
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
    progress_bar: Optional[bool] = False,
) -> List[Dict[str, List]]:

    """

    max_length = remove roles of more than n tokens (NB: very long roles tend to be uninformative in our context)
    progress_bar: print a progress bar (default is False)
    For other arguments see utils.preprocess.

    """

    roles_copy = deepcopy(statements)

    if progress_bar == True:
        print("Cleaning SRL...")
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, tokens in roles_copy[i].items():
            if isinstance(tokens, list):
                res = [
                    preprocess(
                        [" ".join(tokens)],
                        remove_punctuation=remove_punctuation,
                        remove_digits=remove_digits,
                        remove_chars=remove_chars,
                        stop_words=stop_words,
                        lowercase=lowercase,
                        strip=strip,
                        remove_whitespaces=remove_whitespaces,
                        lemmatize=lemmatize,
                        stem=stem,
                        tags_to_keep=tags_to_keep,
                        remove_n_letter_words=remove_n_letter_words,
                    )[0].split()
                ][0]
                if max_length is not None:
                    if len(res) <= max_length:
                        roles_copy[i][role] = res
                    else:
                        roles_copy[i][role] = []
                else:
                    roles_copy[i][role] = res
            elif isinstance(tokens, bool):
                pass
            else:
                raise ValueError(f"{tokens}")

    return roles_copy


def get_raw_arguments(statements: List[dict], progress_bar: Optional[bool] = False):

    roles_copy = deepcopy(statements)

    if progress_bar == True:
        print("Processing raw arguments...")
        time.sleep(1)
        statements = tqdm(statements)

    final_statements = []
    for i, statement in enumerate(statements):
        for role, tokens in statement.items():
            name = role + "-RAW"
            roles_copy[i][name] = roles_copy[i].pop(role)
            if type(tokens) != bool:
                roles_copy[i][name] = " ".join(tokens)
            else:
                roles_copy[i][name] = tokens

    return roles_copy


def get_role_counts(
    statements: List[dict],
    roles: Optional[list] = ["B-V", "ARGO", "ARG1", "ARG2"],
    progress_bar: Optional[bool] = False,
) -> dict:

    """

    Get role frequency within the corpus from preprocessed semantic roles. Roles considered are specified by the user.
    Args:
        statements: list of dictionaries of postprocessed semantic roles
        roles: list of roles considered
        progress_bar: print a progress bar (default is False)

    Returns:
        Dictionary in which postprocessed semantic roles are keys and their frequency within the corpus are values
        (e.g. d['verb'] = count)

    Example:
        >>> test = [{'B-V': ['increase'], 'B-ARGM-NEG': True},{'B-V': ['decrease']},{'B-V': ['decrease']}]\n
        ... verb_counts = get_role_counts(test, roles = ['B-V'])
        {'increase': 1, 'decrease': 2}

    """

    counts = {}

    if progress_bar == True:
        print("Computing role frequencies...")
        time.sleep(1)
        statements = tqdm(statements)

    for statement in statements:
        for key in statement.keys():
            if key in roles:
                temp = " ".join(statement[key])
                if temp in counts:
                    counts[temp] += 1
                else:
                    counts[temp] = 1

    return counts
