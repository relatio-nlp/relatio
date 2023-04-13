import json
from ast import literal_eval
from io import StringIO

import pandas as pd
import requests

datasets = """
{
    "trump_tweet_archive": 
    {
        "description": "Tweets from the Trump Tweet Archives (https://www.thetrumparchive.com/)",
        "language": "english",
        "srl_model": "allennlp v0.9 -- srl-model-2018.05.25.tar.gz",
        "links": 
        {
            "raw": "https://www.dropbox.com/s/lxqz454n29iqktn/trump_archive.csv?dl=1",
            "sentences": "https://www.dropbox.com/s/coh4ergyrjeolen/split_sentences.json?dl=1",
            "srl_res": "https://www.dropbox.com/s/54lloy84ka8mycp/srl_res.json?dl=1"
        }
    },
    "tweets_candidates_french_elections": 
    {
        "description": "Tweets of candidates at the French presidential elections (2022)",
        "language": "french",
        "srl_model": "",
        "links": 
        {
            "raw": "https://www.dropbox.com/s/qqlq8xn9x645f79/tweets_candidates_french_elections.csv?dl=1"
        }
    }
}
"""


def list_data():
    print(datasets)


def load_data(dataset: str, content: str):
    """
    Load a dataset from the list of available datasets.

    Args:
        - dataset: one of the available datasets
        - content: either 'raw', 'sentences' or 'srl_res'
    """

    json_of_datasets = json.loads(datasets)

    if content == "raw":
        r = requests.get(json_of_datasets[dataset]["links"]["raw"])
        r.encoding = "utf8"
        r = pd.read_csv(StringIO(r.text))
    else:
        r = requests.get(json_of_datasets[dataset]["links"][content])
        r = literal_eval(r.text)

    return r
