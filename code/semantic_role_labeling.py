from copy import deepcopy
from typing import List, Dict, Any, Tuple

from allennlp.predictors.predictor import Predictor
import numpy as np

from utils import preprocess


class SRL:
    def __init__(self, path: str):
        self._predictor = Predictor.from_path(path)

    def __call__(self, sentences: List[str]):
        sentences_json = [{"sentence": sent} for sent in sentences]
        res = self._predictor.predict_batch_json(sentences_json)
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


def postprocess_roles(statements: List[Dict[str, List]]) -> List[Dict[str, List]]:
    roles_copy = deepcopy(statements)
    for i, statement in enumerate(statements):
        for role, tokens in statements[i].items():
            if isinstance(tokens, list):
                res = [preprocess([" ".join(tokens)], lemmatize=True)[0].split()][0]
                roles_copy[i][role] = res
            elif isinstance(tokens, bool):
                pass
            else:
                raise ValueError(f"{tokens}")
    return roles_copy
