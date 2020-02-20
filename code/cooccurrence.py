from collections import Counter
import itertools
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np
import pandas as pd

from utils import UsedRoles


def vectors_as_tuples(vectors: List[List[Dict[str, Any]]], used_roles: UsedRoles):
    # Chain list of lists to get only list of dicts
    list_of_all_dicts = list(itertools.chain.from_iterable(vectors))

    # Now write list of dicts to df
    df = pd.DataFrame(list_of_all_dicts)
    subset = [el for el in used_roles.keys() if used_roles[el]]
    df_ = df[subset]
    df_ = df_.dropna()
    # SET ALL TO INTEGER!!!
    tuples = list(df_.itertuples(index=False, name=None))
    # Group verb and negation or modals into one tuple
    if "B-ARGM-MOD" in subset or "B-ARGM-NEG" in subset:
        BV_index = subset.index("B-V")
        tuples = [tup[:BV_index] + (tup[BV_index:],) for tup in tuples]
    return tuples


def unique_counts(input: List[Tuple[Any]]) -> Dict[Tuple[Any], int]:
    """
    Count the unique elements of the List.
    
    Examples:
    >>> unique_counts([])
    {}
    >>> unique_counts([(),()])
    {(): 2}
    >>> unique_counts(((1,None),(1,None),(1,2)))
    {(1, None): 2, (1, 2): 1}
    """
    res = dict(Counter(input))
    return res


def unique_tuple_values_counts(input: List[Tuple[Any]]) -> List[Dict[Any, int]]:
    """
    Count the unique elements of the tuples inside the list on the same position.

    Examples:
    >>> unique_tuple_values_counts([])
    []
    >>> unique_tuple_values_counts([(),()])
    []
    >>> unique_tuple_values_counts([(1,None),(1,None),(1,2)])
    [{1: 3}, {None: 2, 2: 1}]
    """
    res: List[Dict[Any, int]] = []
    if not input:
        return res

    for i in range(0, len(input[0])):
        res.append(dict(Counter([t[i] for t in input])))
    return res


def compute_pmi(
    vectors_or_tuples: List[Union[List[Dict[str, Any]], Tuple[Any]]],
    used_roles: Optional[UsedRoles] = None,
) -> Dict[Tuple[Any], float]:
    if not vectors_or_tuples:
        return {}
    if isinstance(vectors_or_tuples[0], list):
        assert isinstance(used_roles, UsedRoles)
        tuples = vectors_as_tuples(vectors_or_tuples, used_roles)
    else:
        tuples = vectors_or_tuples
    counts_narratives = unique_counts(tuples)
    counts_individual = unique_tuple_values_counts(tuples)

    results_dic = {}
    for t in counts_narratives:
        frequency_narrative = counts_narratives[t]
        product = 1
        for j in range(0, len(counts_individual)):
            product = product * counts_individual[j][t[j]]
        pmi = np.log(frequency_narrative / product)
        results_dic[t] = pmi
    return results_dic
