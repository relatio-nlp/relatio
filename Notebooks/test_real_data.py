# Short notebook for test samples.

# NYT SAMPLE:
# TEXT AS DATA CONFERENCE - OCT 2019
# The SRL output to 'reproduce' our results is in the following folder:
# /cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/srl_output_tax_2sents/

# CONGRESS RECORDS (GPO) SAMPLES

# %%
import glob
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

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
from clustering import Clustering, label_clusters, label_clusters_most_freq
from sklearn.cluster import KMeans
from cooccurrence import build_df, CoOccurrence

used_roles = UsedRoles()
used_roles["ARG2"] = True
folder = "../myNotebooks/test/"
sif_w2v = SIF_Word2Vec(folder + "w2v/cong_gpo_word2vec.model")

# %%
filenames = glob.glob(folder + "srl*")

documents_all = []
postproc_roles_all = []
sentence_index_all = []  # np.array([], dtype=np.uint32)
vectors_all = []  # None
statement_index_all = []  # {}
funny_index_all = []  # {}


def do_all(filenames):
    global documents_all, postproc_roles_all, sentence_index_all, vectors_all, statement_index_all, funny_index_all

    start_index = 0

    for filename in tqdm(filenames):
        with open(filename) as json_file:
            srl_res = json.load(json_file)

        roles, sentence_index = extract_roles(srl_res, start=start_index)

        postproc_roles = postprocess_roles(roles)

        sif_vectors, sif_statements_index, sif_funny_index = compute_embedding(
            sif_w2v, statements=postproc_roles, used_roles=used_roles, start=start_index
        )
        documents_all.append(Document(filename, start_index))
        postproc_roles_all.extend(postproc_roles)
        sentence_index_all.append(sentence_index)
        vectors_all.append(sif_vectors)
        statement_index_all.append(sif_statements_index)
        funny_index_all.append(sif_funny_index)

        start_index += sentence_index.size


do_all(filenames)
sentence_index_all = np.concatenate(sentence_index_all)
vectors_all = dict_concatenate(vectors_all)
statement_index_all = dict_concatenate(statement_index_all)
funny_index_all = dict_concatenate(funny_index_all)

# %%
# Clustering and Labelling all the data
kmeans = KMeans(random_state=0)

clustering = Clustering(
    cluster=kmeans,
    n_clusters={"ARGO": 4, "ARG1": 2, "ARG2": 1, "B-V": 4},
    used_roles=used_roles,
)

sample_vectors = clustering.resample(vectors=vectors_all, sample_size=0.9)
clustering.fit(vectors=sample_vectors)
clustering_res = clustering.predict(vectors=vectors_all)
distance = clustering.compute_distance(vectors_all, clustering_res)

# %% Use the most similar based on embedding (word2vec)
labels = clustering.label_most_similar_in_w2v(sif_w2v)
labels
# %% Use the labeling based on Euclidean distance. If you want to use it in CoOccurrence make sure you use top=1 (default value)
labels = label_clusters(
    clustering_res=clustering_res,
    distance=distance,
    postproc_roles=postproc_roles_all,
    statement_index=statement_index_all,
    top=1,
)
labels
# %% Use the labeling based on most frequent
clustering_mask = clustering.distance_mask(distance, threshold=1.5)
labels = label_clusters_most_freq(
    clustering_res=clustering_res,
    postproc_roles=postproc_roles_all,
    statement_index=statement_index_all,
    clustering_mask=clustering_mask,
)
labels
# %%
df = build_df(
    clustering_res=clustering_res,
    postproc_roles=postproc_roles_all,
    statement_index=statement_index_all,
    used_roles=used_roles,
    clustering_mask=clustering_mask,
)
df
# %%
# Write df, labels and previously used roles to files for future work

df.to_pickle(folder + "res/df.pkl")
with open(folder + "res/labels.json", "w") as f:
    json.dump(labels, f, indent=4)
with open(folder + "res/used_roles.json", "w") as f:
    json.dump(used_roles.as_dict(), f, indent=4)


# %%
# Read df, labels and previously used roles
df = pd.read_pickle(folder + "res/df.pkl")
with open(folder + "res/labels.json", "r") as f:
    labels = json.load(f)
    for role, value in labels.items():
        labels[role] = {int(k): v for k, v in value.items()}
with open(folder + "res/used_roles.json", "r") as f:
    used_roles = UsedRoles(json.load(f))
labels
# %%
# Run CoOccurrence

cooc = CoOccurrence(df, labels, used_roles)
cooc.subset = {"ARGO", "ARG1", "B-V", "B-ARGM-NEG"}
print(cooc.normal_order)
print(cooc.display_order)

cooc.narratives_counts
# %%
cooc.narratives_pmi

# %%
