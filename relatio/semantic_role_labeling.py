# Semantic Role Labeling
# link to choose the SRL model
# https://storage.googleapis.com/allennlp-public-models/YOUR-PREFERRED-MODEL

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from allennlp_models.structured_prediction.predictors import (
    SemanticRoleLabelerPredictor as Predictor,
)
from tqdm import tqdm

from .utils import (
    group_sentences_in_batches,
    is_subsequence,
    replace_sentences,
    save_roles,
)


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
        """
        A class to perform semantic role labeling on a list of sentences.

        Args:
            path: Path to the SRL model (i.e., https://storage.googleapis.com/allennlp-public-models/YOUR-PREFERRED-MODEL)
            cuda_device: The cuda device to use. If -1, the CPU is used.
            max_batch_char_length: Maximum number of characters in a batch. If None, the maximum number of characters is set to Inf.
            batch_size: Maximum number of sentences in a batch. If None, the maximum number of sentences is set to Inf.
            max_sentence_length: Maximum number of characters in a sentence. If None, the maximum number of characters is set to Inf.
            max_number_words: Maximum number of words in a sentence. If None, the maximum number of words is set to Inf.
            cuda_empty_cache: If True, the cache is emptied after each batch.
            cuda_sleep: Time to wait after emptying the cache.
        """

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
        output_path: Optional[str] = None,
        progress_bar: bool = False,
    ):
        """
        Run the SRL model on a list of sentences.

        Args:
            sentences: List of sentences.
            output_path: Path to save the output (saved as a json file).
            progress_bar: If True, a progress bar is displayed.
        """

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

        res: List[Dict[str, List]] = []

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

        if output_path is not None:
            save_roles(res, output_path)

        return res


def _extract_role_per_sentence(
    sentence_dict: dict,
    used_roles: List[str],
    only_triplets: bool = True,
) -> List[Dict[str, Union[str, bool]]]:
    """

    Extract the semantic roles for a given sentence.

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
        for role in ["ARG0", "B-V", "B-ARGM-MOD", "ARG1", "ARG2"]:
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
            if not value or value == "":
                key_to_delete.append(key)
        for key in key_to_delete:
            del statement_role_dict[key]

        if only_triplets:
            if is_subsequence(
                ["ARG0", "B-V", "ARG1"], list(statement_role_dict.keys())
            ) or is_subsequence(
                ["ARG0", "B-V", "ARG2"], list(statement_role_dict.keys())
            ):
                sentence_role_list.append(statement_role_dict)
        else:
            sentence_role_list.append(statement_role_dict)

    return sentence_role_list


def extract_roles(
    srl: List[Dict[str, Any]],
    used_roles: List[str],
    only_triplets: bool = False,
    progress_bar: bool = False,
) -> Tuple[List[Dict[str, Union[str, bool]]], List[int]]:
    """

    Extract semantic roles from the SRL output.

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
        print("Extracting semantic roles...")
        time.sleep(1)
        srl = tqdm(srl)

    for i, sentence_dict in enumerate(srl):
        role_per_sentence = _extract_role_per_sentence(
            sentence_dict, used_roles, only_triplets
        )
        if role_per_sentence:
            sentence_index.extend([i] * len(role_per_sentence))
            statements_role_list.extend(role_per_sentence)

    return statements_role_list, np.asarray(sentence_index, dtype=np.uint32)
