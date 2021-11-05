# MIT License

# Copyright (c) 2020-2021 ETH Zurich, Andrei V. Plamada
# Copyright (c) 2020-2021 ETH Zurich, Elliott Ash
# Copyright (c) 2020-2021 University of St.Gallen, Philine Widmer
# Copyright (c) 2020-2021 Ecole Polytechnique, Germain Gauthier

from ast import literal_eval
from io import StringIO

import pandas as pd
import requests


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
        r = literal_eval(r.text)
    elif format == "srl_res":
        r = requests.get("https://www.dropbox.com/s/54lloy84ka8mycp/srl_res.json?dl=1")
        r = literal_eval(r.text)
    else:
        raise ValueError(
            "file argument should either be raw, split_sentences or srl_res"
        )

    return r
