from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Any
import requests
import pandas as pd
from io import StringIO


def list_datasets():
    s = """
    List of available datasets:

    Trump Tweet Archive
    - function call: load_trump_data()
    - format: 'raw', 'split_sentences', 'srl_res'
    - allennlp version: 0.9
    - srl model: srl-model-2018.05.25.tar.gz
    """

    return s


def load_trump_data(format: str):
    """

    Load processed tweets from the Trump Tweet Archives (https://www.thetrumparchive.com/).

    Args:
        file: either 'raw' (i.e. dataframe with raw text), 'split_sentences' (i.e. result from split_sentences()) or 'srl_res' (i.e. result from run_srl())

    Returns:
        The desired object.
    """

    if format == "raw":
        r = requests.get(
            "https://www.dropbox.com/s/lxqz454n29iqktn/trump_archive.csv?dl=1"
        )
        r = pd.read_csv(StringIO(r.text))
    elif format == "split_sentences":
        r = requests.get(
            "https://www.dropbox.com/s/coh4ergyrjeolen/split_sentences.json?dl=1"
        )
        r = eval(r.text)
    elif format == "srl_res":
        r = requests.get("https://www.dropbox.com/s/54lloy84ka8mycp/srl_res.json?dl=1")
        r = eval(r.text)
    else:
        raise ValueError(
            "file argument should either be raw, split_sentences or srl_res"
        )

    return r
