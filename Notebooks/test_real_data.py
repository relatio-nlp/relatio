# Short notebook for test samples.

# NYT SAMPLE:
# TEXT AS DATA CONFERENCE - OCT 2019
# The SRL output to 'reproduce' our results is in the following folder:
# /cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/srl_output_tax_2sents/

# CONGRESS RECORDS (GPO) SAMPLES

# %%
import glob
import json
import typing

import numpy as np
import pandas as pd

# %%
import sys

sys.path.append("../code")

from utils import (
    tokenize_into_sentences,
    filter_sentences,
    preprocess,
    UsedRoles,
    Document,
    dict_concatenate,
)
from word_embedding import run_word2vec, compute_embedding, USE, SIF_Word2Vec
from semantic_role_labeling import SRL, extract_roles, postprocess_roles
from clustering import Clustering
from sklearn.cluster import KMeans
from cooccurrence import build_df_and_labels, CoOccurence

used_roles = UsedRoles()
used_roles["ARG2"] = True

sif_w2v = SIF_Word2Vec("./test/w2v/cong_gpo_word2vec.model")

kmeans = KMeans(random_state=0)

clustering = Clustering(
    cluster=kmeans,
    n_clusters={"ARGO": 2, "ARG1": 1, "ARG2": 1, "B-V": 2},
    used_roles=used_roles,
)

# %%
filenames = glob.glob("./test/srl*")

# %%
documents_all = []
postproc_roles_all = []
sentence_index_all = np.array([], dtype=np.uint32)
vectors_all = None
statement_index_all = {}
funny_index_all = {}

start_index = 0
for filename in filenames:
    with open(filename) as json_file:
        srl_res = json.load(json_file)

    roles, sentence_index = extract_roles(srl_res, start=start_index)

    postproc_roles = postprocess_roles(roles)

    sif_vectors, sif_statements_index, sif_funny_index = compute_embedding(
        sif_w2v, statements=postproc_roles, used_roles=used_roles, start=start_index
    )

    documents_all.append(Document(filename, start_index))
    postproc_roles_all.extend(postproc_roles)
    sentence_index_all = np.append(sentence_index_all, sentence_index)
    vectors_all = dict_concatenate(vectors_all, sif_vectors)
    statement_index_all = dict_concatenate(statement_index_all, sif_statements_index)
    funny_index_all = dict_concatenate(funny_index_all, sif_funny_index)
    start_index += sentence_index.size


# %%
# Clustering and Labelling all the data
clustering.fit(vectors=vectors_all, sample_size=None)
clustering_res = clustering.predict(vectors=vectors_all)

df, labels = build_df_and_labels(
    postproc_roles_all, clustering_res, statement_index_all, used_roles
)
# %%
# Write df, labels and previously used roles to files for future work

df.to_pickle("./df.pkl")
with open("labels.json", "w") as f:
    json.dump(labels, f, indent=4)
with open("used_roles.json", "w") as f:
    json.dump(used_roles.as_dict(), f, indent=4)

