import json
from typing import List, Union, Dict, Any

from allennlp.predictors.predictor import Predictor


def run_srl(
    sentences: List[str], predictor_path: str, save_path: Union[None, str] = None
) -> Union[None, List[Dict[str, Any]]]:
    predictor = Predictor.from_path(predictor_path)
    sentences_json = [{"sentence": sent} for sent in sentences]
    res = predictor.predict_batch_json(sentences_json)

    if save_path is not None:
        with open(save_path, "w+") as outfile:
            json.dump(res, outfile)
        return None
    else:
        return res
