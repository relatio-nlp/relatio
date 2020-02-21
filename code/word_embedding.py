from typing import List, Union, Dict, Any, Optional, Callable, Tuple

import numpy as np
import tensorflow_hub as hub

from gensim.models import KeyedVectors, Word2Vec

from utils import UsedRoles


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


def encode_role_sif(
    model: Word2Vec, tokens: List[str], sif_dict: Dict[str, int]
) -> np.ndarray:
    return np.mean([sif_dict[token] * model.wv[token] for token in tokens], axis=0)


def encode_role_USE(embed: Any, tokens: List[str]) -> np.ndarray:
    return (embed([" ".join(tokens)]))[0].numpy()


def compute_embedding(
    # TODO Refactor (weights, etc)
    model: Union[Callable, Word2Vec],
    statements: List[Dict[str, List]],
    used_roles: UsedRoles,
    alpha: Optional[float] = 0.001,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    if isinstance(model, Word2Vec):
        # TODO : Load just the  KeyedVectors
        # Create a word count dictionary based on the trained model
        word_count_dict = {}
        for word, vocab_obj in model.wv.vocab.items():
            word_count_dict[word] = vocab_obj.count

        # Create a dictionary that maps from a word to its frequency and then use the frequency of
        # a word to compute its sif-weight (saved in sif_dict)
        sif_dict = {}
        for word, count in word_count_dict.items():
            sif_dict[word] = alpha / (alpha + count)
        compute_role_embedding = encode_role_sif
        kwargs = {"sif_dict": sif_dict}
    elif callable(model):
        compute_role_embedding = encode_role_USE
        kwargs = {}

    role_names = []
    for el in used_roles.keys():
        if used_roles[el] and (el not in ["B-ARGM-NEG", "B-ARGM-MOD"]):
            role_names.append(el)
    statements_index = {el: [] for el in role_names}
    roles_vectors = {el: [] for el in role_names}
    not_found_or_empty_index = {el: [] for el in role_names}

    for i, statement in enumerate(statements):
        for role_name, tokens in statement.items():
            if (role_name in role_names) and (
                role_name not in ["B-ARGM-NEG", "B-ARGM-MOD"]
            ):
                if isinstance(model, Word2Vec):
                    if not tokens:
                        not_found_or_empty_index[role_name].append(i)
                        continue
                    if any(token not in sif_dict for token in tokens):
                        not_found_or_empty_index[role_name].append(i)
                        continue
                statements_index[role_name].append(i)
                roles_vectors[role_name].append(
                    compute_role_embedding(model, tokens, **kwargs)
                )

    for role_name in role_names:
        roles_vectors[role_name] = np.asarray(roles_vectors[role_name])
        statements_index[role_name] = np.asarray(statements_index[role_name])
    return roles_vectors, statements_index, not_found_or_empty_index
