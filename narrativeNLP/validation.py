# Analysis
# ..................................................................................................................
# ..................................................................................................................

import pandas as pd


def inspect_label(final_statements, label: str, role: str):

    """

    A function to inspect the content of a label for a user-specified role
    (i.e. to check the 'quality' of the clusters and named entity recognition).

    Args:
        final_statements: dataframe with the output of the pipeline
        label: label to inspect
        role: role to inspect

    Returns:
        A pandas series sorted by frequency of raw roles contained in this label

    """

    res = final_statements.loc[
        final_statements[role] == label, str(role + "-RAW")
    ].value_counts()

    return res


def inspect_narrative(final_statements, narrative: str):

    """

    A function to inspect the raw statements represented by a narrative
    (i.e. to check the 'quality' of the final narratives).

    Args:
        final_statements: dataframe with the output of the pipeline
        narrative: cleaned narrative to inspect

    Returns:
        A pandas series sorted by frequency of raw narratives contained in this label

    """

    res = final_statements.loc[
        final_statements["narrative-CLEANED"] == narrative, "narrative-RAW"
    ].value_counts()

    return res
