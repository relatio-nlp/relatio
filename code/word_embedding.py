from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm

import tensorflow_hub as hub
from gensim.models import KeyedVectors, Word2Vec
from utils import UsedRoles


class USE:
    def __init__(self, path: str):
        self._embed = hub.load(path)

    def __call__(self, tokens: List[str]) -> np.ndarray:
        return self._embed([" ".join(tokens)]).numpy()[0]


class SIF_Word2Vec:
    def __init__(
        self, path: str, alpha: Optional[float] = 0.001, normalize: bool = True
    ):

        self._model = Word2Vec.load(path)
        # TODO : Load just the  KeyedVectors
        # Create a word count dictionary based on the trained model
        word_count_dict = {}
        for word, vocab_obj in self._model.wv.vocab.items():
            word_count_dict[word] = vocab_obj.count

        # Create a dictionary that maps from a word to its frequency and then use the frequency of
        # a word to compute its sif-weight (saved in sif_dict)
        self._sif_dict = {}
        for word, count in word_count_dict.items():
            self._sif_dict[word] = alpha / (alpha + count)

        self._normalize = normalize

    def __call__(self, tokens: List[str]):
        res = np.mean(
            [self._sif_dict[token] * self._model.wv[token] for token in tokens], axis=0
        )
        if self._normalize:
            res = res / norm(res)  # normalize
        return res

    def most_similar(self, v):
        return self._model.wv.most_similar(positive=[v], topn=1)[0]


def run_word2vec(
    sentences: List[str],
    model: Word2Vec,
    pretrained_path: str,
    save_path: Union[None, str] = None,
) -> Word2Vec:
    w2v_sentences = [sent.split() for sent in sentences]
    model.build_vocab(w2v_sentences)
    total_examples = model.corpus_count
    pretrained_model = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
    model.build_vocab([list(pretrained_model.vocab.keys())], update=True)
    model.intersect_word2vec_format(pretrained_path, binary=False, lockf=1.0)
    model.train(w2v_sentences, total_examples=total_examples, epochs=model.iter)

    if save_path is not None:
        model.save(save_path)

    return model


def compute_embedding(
    # TODO Refactor (weights, etc)
    model: Union[USE, SIF_Word2Vec],
    statements: List[Dict[str, List]],
    used_roles: UsedRoles,
    start: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    # normalize is applied only to Word2Vec
    if not isinstance(model, (USE, SIF_Word2Vec)):
        raise TypeError("Union[USE, Word2Vec]")

    embed_roles = used_roles.embeddable
    not_embed_roles = used_roles.not_embeddable
    statements_index = {el: [] for el in embed_roles}
    roles_vectors = {el: [] for el in embed_roles}
    not_found_or_empty_index = {el: [] for el in embed_roles}

    for i, statement in enumerate(statements, start=start):
        for role_name, tokens in statement.items():
            if (role_name in embed_roles) and (role_name not in not_embed_roles):
                if isinstance(model, SIF_Word2Vec):
                    if not tokens:
                        not_found_or_empty_index[role_name].append(i)
                        continue
                    if any(token not in model._sif_dict for token in tokens):
                        not_found_or_empty_index[role_name].append(i)
                        continue
                statements_index[role_name].append(i)
                roles_vectors[role_name].append(model(tokens))

    for role_name in embed_roles:
        roles_vectors[role_name] = np.asarray(
            roles_vectors[role_name], dtype=np.float32
        )
        for el in [statements_index, not_found_or_empty_index]:
            el[role_name] = np.asarray(el[role_name], dtype=np.uint32)

    return roles_vectors, statements_index, not_found_or_empty_index
