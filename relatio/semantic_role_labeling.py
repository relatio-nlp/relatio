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
from allennlp_models.structured_prediction.predictors import (
    SemanticRoleLabelerPredictor as Predictor,
)
from tqdm import tqdm

from .utils import group_sentences_in_batches, replace_sentences


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
        return res
