from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set

import numpy as np
import pandas as pd

from utils import UsedRoles


def build_df(
    postproc_roles, clustering_res, statement_index, used_roles
) -> pd.DataFrame:
    series = []
    for role in used_roles.used:
        if role == "B-ARGM-NEG":
            serie = pd.Series(
                data=[statement.get(role) for statement in postproc_roles],
                dtype="boolean",
                name=role,
            )
        elif role == "B-ARGM-MOD":
            b_arg_mod_res = []
            b_arg_mod_index = []
            for i, statement in enumerate(postproc_roles):
                if statement.get(role) is not None:
                    _res = statement[role]
                    if len(_res) > 1:
                        raise ValueError(f"Expect one element:{_res}")
                    else:
                        b_arg_mod_index.append(i)
                        b_arg_mod_res.append(_res[0])
            serie = pd.Series(data=b_arg_mod_res, index=b_arg_mod_index, name=role)

        elif role in used_roles.embeddable:
            serie = pd.Series(
                data=clustering_res[role],
                index=statement_index[role],
                dtype="UInt16",
                name=role,
            )
        series.append(serie)
    return pd.concat(series, axis=1)


def subset_as_tuples(
    df: pd.DataFrame, used_roles: UsedRoles, roles_subset: Optional[Set[str]] = None
):
    if roles_subset is None:
        sublist = used_roles.used
    else:
        if not set(roles_subset).issubset(used_roles.used):
            raise ValueError(f"{roles_subset} not in {used_roles.used}")
        sublist = [el for el in used_roles.used if el in roles_subset]

    df = df.loc[:, sublist].dropna()

    tuples = list(df.itertuples(index=False, name=None))
    # Group verb and negation or modals into one tuple
    BV_index = sublist.index("B-V")
    if "B-ARGM-MOD" in sublist or "B-ARGM-NEG" in sublist:
        tuples = [tup[:BV_index] + (tup[BV_index:],) for tup in tuples]
    return tuples


def unique_counts(tuples: List[Tuple[Any]]) -> Dict[Tuple[Any], int]:
    """
    Count the unique elements of the List.
    
    Examples:
    >>> unique_counts([])
    {}
    >>> unique_counts([(),()])
    {(): 2}
    >>> unique_counts([(1,None),(1,None),(1,2)])
    {(1, None): 2, (1, 2): 1}
    """
    res = dict(Counter(tuples))
    return res


def unique_tuple_values_counts(tuples: List[Tuple[Any]]) -> List[Dict[Any, int]]:
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
    if not tuples:
        return res

    for i in range(0, len(tuples[0])):
        res.append(dict(Counter([t[i] for t in tuples])))
    return res


def compute_pmi(tuples: List[Tuple[Any]]) -> Dict[Tuple[Any], float]:
    if not tuples:
        return {}
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
