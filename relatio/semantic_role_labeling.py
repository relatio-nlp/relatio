# Semantic Role Labeling
# ..................................................................................................................
# ..................................................................................................................

# link to choose the SRL model
# https://storage.googleapis.com/allennlp-public-models/YOUR-PREFERRED-MODEL

import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

from .utils import clean_text, group_sentences_in_batches, replace_sentences


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
        progress_bar: bool = False,
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

        sentences = replace_sentences(
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

        if progress_bar:
            print("Running SRL...")
            time.sleep(1)
            batches = tqdm(batches)

        for batch in batches:
            sentences_json = [{"sentence": sent} for sent in batch]
            try:
                res_batch = self._predictor.predict_batch_json(sentences_json)
            except RuntimeError as err:
                warnings.warn(f"empty result {err}", RuntimeWarning)
                res = [{"words": [], "verbs": []}] * len(batch)
                break
            except:
                raise
            finally:
                self._clean_cache(cuda_sleep, cuda_empty_cache)

            res.extend(res_batch)
        return res


def extract_roles(
    srl: List[Dict[str, Any]], used_roles: List[str], progress_bar: bool = False
) -> Tuple[List[Dict[str, Union[str, bool]]], List[int]]:

    """

    A function that extracts semantic roles from the SRL output.

    Args:
        srl: srl output
        used_roles: list of semantic roles to extract
        progress_bar: print a progress bar (default is False)

    Returns:
        List of statements and numpy array of sentence indices (to keep track of sentences)

    """

    statements_role_list: List[Dict[str, Union[str, bool]]] = []
    sentence_index: List[int] = []

    if progress_bar:
        print("Processing SRL...")
        time.sleep(1)
        srl = tqdm(srl)

    for i, sentence_dict in enumerate(srl):
        role_per_sentence = extract_role_per_sentence(sentence_dict, used_roles)
        sentence_index.extend([i] * len(role_per_sentence))
        statements_role_list.extend(role_per_sentence)

    return statements_role_list, np.asarray(sentence_index, dtype=np.uint32)


def extract_role_per_sentence(
    sentence_dict: dict, used_roles: List[str]
) -> List[Dict[str, Union[str, bool]]]:

    """

    A function that extracts the semantic roles for a given sentence.

    Args:
        srl: srl output
        used_roles: list of semantic roles to extract

    Returns:
        List of statements with their associated roles for a given sentence

    """

    word_list = sentence_dict["words"]
    sentence_role_list = []

    for statement_dict in sentence_dict["verbs"]:
        tag_list = statement_dict["tags"]

        statement_role_dict: Dict[str, Union[str, bool]] = {}
        for role in ["ARG0", "ARG1", "ARG2", "B-V", "B-ARGM-MOD"]:
            if role in used_roles:
                indices_role = [i for i, tok in enumerate(tag_list) if role in tok]
                toks_role = [
                    tok for i, tok in enumerate(word_list) if i in indices_role
                ]
                statement_role_dict[role] = " ".join(toks_role)

        if "B-ARGM-NEG" in used_roles:
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


def process_roles(
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
    progress_bar: bool = False,
) -> List[Dict[str, List]]:

    """

    Takes a list of raw extracted semantic roles and cleans the text.

    Args:
        max_length = remove roles of more than n characters (NB: very long roles tend to be uninformative)
        progress_bar: print a progress bar (default is False)
        For other arguments see utils.clean_text.

    Returns:
        List of processed statements

    """

    roles_copy = deepcopy(statements)

    if progress_bar:
        print("Cleaning SRL...")
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, role_content in roles_copy[i].items():
            if isinstance(role_content, str):
                res = clean_text(
                    [role_content],
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
                )[0]
                if max_length is not None:
                    if len(res) <= max_length:
                        roles_copy[i][role] = res
                    else:
                        roles_copy[i][role] = ""
                else:
                    roles_copy[i][role] = res
            elif isinstance(role_content, bool):
                pass
            else:
                raise ValueError(f"{role_content}")

    return roles_copy


def rename_arguments(
    statements: List[dict], progress_bar: bool = False, suffix: str = "_highdim"
):

    """

    Takes a list of dictionaries and renames the keys of the dictionary with an extra user-specified suffix.

    Args:
        statements: list of statements
        progress_bar: print a progress bar (default is False)
        suffix: extra suffix to add to the keys of the dictionaries

    Returns:
        List of dictionaries with renamed keys.

    """

    roles_copy = deepcopy(statements)

    if progress_bar:
        print("Processing raw arguments...")
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, role_content in statement.items():
            name = role + suffix
            roles_copy[i][name] = roles_copy[i].pop(role)
            roles_copy[i][name] = role_content

    return roles_copy
