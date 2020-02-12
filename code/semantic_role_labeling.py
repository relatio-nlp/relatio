import json
from typing import List, Union, Dict, Any

from allennlp.predictors.predictor import Predictor

from utils import preprocess


def run_srl(
    sentences: List[str], predictor_path: str, save_path: Union[None, str] = None
) -> List[Dict[str, Any]]:
    predictor = Predictor.from_path(predictor_path)
    sentences_json = [{"sentence": sent} for sent in sentences]
    res = predictor.predict_batch_json(sentences_json)

    if save_path is not None:
        with open(save_path, "w+") as outfile:
            json.dump(res, outfile)

    return res


def extract_roles(
    srl: List[Dict[str, Any]], modals=True, negation=False
) -> List[List[Dict[str, List]]]:
    sentences_role_list = []
    for sentence_dict in srl:
        sentences_role_list.append(extract_role_per_sentence(sentence_dict, modals))
    return sentences_role_list


def extract_role_per_sentence(sentence_dict, modals=True):
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


def postprocess_roles(roles: List[List[Dict[str, List]]]):
    for i, sent in enumerate(roles):
        for j, statement in enumerate(sent):
            for role, tokens in roles[i][j].items():
                if isinstance(tokens, list):
                    roles[i][j][role] = [
                        preprocess([" ".join(tokens)], lemmatize=True)[0].split()
                    ][0]
                elif isinstance(tokens, bool):
                    pass
                else:
                    raise ValueError(f"{tokens}")
    return roles
