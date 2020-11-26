import pathlib
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from allennlp.predictors.predictor import Predictor

from utils import filter_sentences, group_sentences_in_batches, preprocess


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
        for batch in batches:
            sentences_json = [{"sentence": sent} for sent in batch]
            try:
                res_batch = self._predictor.predict_batch_json(sentences_json)
            except RuntimeError as err:
                warnings.warn(
                    f"empty result {err}", RuntimeWarning,
                )
                res = [None]
                break
            except:
                raise
            finally:
                self._clean_cache(cuda_sleep, cuda_empty_cache)

            res.extend(res_batch)
        return res


def extract_roles(
    srl: List[Dict[str, Any]], modals=True, start: int = 0
) -> Tuple[List[Dict[str, List]], List[int]]:
    # TODO use UsedRoles instead of modals
    statements_role_list: List[Dict[str, List]] = []
    sentence_index: List[int] = []
    for i, sentence_dict in enumerate(srl, start=start):
        role_per_sentence = extract_role_per_sentence(sentence_dict, modals)
        sentence_index.extend([i] * len(role_per_sentence))
        statements_role_list.extend(role_per_sentence)

    return statements_role_list, np.asarray(sentence_index, dtype=np.uint32)


def extract_role_per_sentence(sentence_dict, modals=True):
    # TODO Refactor
    # TODO use UsedRoles instead of modals

    word_list = sentence_dict["words"]
    sentence_role_list = []
    for statement_dict in sentence_dict["verbs"]:
        tag_list = statement_dict["tags"]

        if any("ARG" in tag for tag in tag_list):
            statement_role_dict = {}

            indices_agent = [i for i, tok in enumerate(tag_list) if "ARG0" in tok]
            indices_patient = [i for i, tok in enumerate(tag_list) if "ARG1" in tok]
            indices_attribute = [i for i, tok in enumerate(tag_list) if "ARG2" in tok]
            indices_verb = [i for i, tok in enumerate(tag_list) if "B-V" in tok]
            agent = [tok for i, tok in enumerate(word_list) if i in indices_agent]
            patient = [tok for i, tok in enumerate(word_list) if i in indices_patient]
            attribute = [
                tok for i, tok in enumerate(word_list) if i in indices_attribute
            ]
            verb = [tok for i, tok in enumerate(word_list) if i in indices_verb]
            if modals is True:
                indices_modal = [
                    i for i, tok in enumerate(tag_list) if "B-ARGM-MOD" in tok
                ]
                modal = [tok for i, tok in enumerate(word_list) if i in indices_modal]
                statement_role_dict["B-ARGM-MOD"] = modal

            role_negation_value = any("B-ARGM-NEG" in tag for tag in tag_list)

            statement_role_dict["ARGO"] = agent
            statement_role_dict["ARG1"] = patient
            statement_role_dict["ARG2"] = attribute
            statement_role_dict["B-V"] = verb
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
    remove_n_letter_words: Optional[int] = None
) -> List[Dict[str, List]]:
    """
    For arguments see utils.preprocess .
    """
    roles_copy = deepcopy(statements)
    for i, statement in enumerate(statements):
        for role, tokens in statements[i].items():
            if isinstance(tokens, list):
                res = [
                    preprocess(
                        [" ".join(tokens)],
                        remove_punctuation = remove_punctuation,
                        remove_digits = remove_digits,
                        remove_chars = remove_chars,
                        stop_words = stop_words,
                        lowercase = lowercase,
                        strip = strip,
                        remove_whitespaces = remove_whitespaces,
                        lemmatize = lemmatize,
                        stem = stem,
                        tags_to_keep = tags_to_keep,
                        remove_n_letter_words = remove_n_letter_words
                    )[0].split()
                ][0]
                if max_length is not None:
                    if len(res) <= max_length: 
                        roles_copy[i][role] = res
                    else :
                        roles_copy[i][role] = []
                else:
                    roles_copy[i][role] = res
            elif isinstance(tokens, bool):
                pass
            else:
                raise ValueError(f"{tokens}")
    return roles_copy


def estimate_time(char_length: int, device: str = "RTX2080Ti") -> float:
    """
    Estimate time to solution for SRL done on a given device.
    
    """

    if device == "RTX2080Ti":
        if char_length > 10_000:
            res = char_length * 0.3 / 1_000
        elif char_length > 2_000:
            res = char_length * 0.6 / 1_000
        else:
            res = 1.0
    else:
        raise ValueError("{device} not estimated.")

    return res


def output_file(filepath: pathlib.Path, parent_path: pathlib.Path) -> pathlib.Path:
    return (parent_path / filepath.name).with_suffix(".json")

