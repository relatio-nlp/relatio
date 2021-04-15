# Browse list of available datasets

from narrativeNLP.datasets import list_datasets

print(list_datasets())

# Load an available dataset

from narrativeNLP.datasets import load_trump_data

df = load_trump_data("raw")

# Split into sentences (example on 100 tweets)

from narrativeNLP.utils import split_into_sentences

split_sentences = split_into_sentences(
    df.iloc[0:100], save_to_disk=None, progress_bar=True
)

# Run SRL (example on a 100 tweets)

from narrativeNLP.wrappers import run_srl

srl_res = run_srl(
    path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
    sentences=split_sentences[1],
    save_to_disk=None,
    batch_size=20,
    progress_bar=True,
)

# As sentence splitting and SRL is time-consuming, we download the results from the datasets module.

split_sentences = load_trump_data("split_sentences")
srl_res = load_trump_data("srl_res")

# Build the narrative model
# This will take several minutes to run. You might want to grab a coffee.

from narrativeNLP.wrappers import build_narrative_model

narrative_model = build_narrative_model(
    srl_res=srl_res[0:1000],
    sentences=split_sentences[1][0:1000],  # list of sentences
    roles_considered=["ARGO", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
    roles_with_embeddings=[["ARGO", "ARG1", "ARG2"]],
    embeddings_type="gensim_keyed_vectors",  # see documentation for a list of supported types
    embeddings_path="glove-wiki-gigaword-300",
    n_clusters=[[10, 20]],  # try different cluster numbers
    verbose=1,
    roles_with_entities=["ARGO", "ARG1", "ARG2"],
    top_n_entities=10,
    dimension_reduce_verbs=True,
    save_to_disk=None,
    max_length=4,
    remove_punctuation=True,
    remove_digits=True,
    remove_chars="",
    stop_words=["the"],
    lowercase=True,
    strip=True,
    remove_whitespaces=True,
    lemmatize=True,
    stem=False,
    tags_to_keep=None,
    remove_n_letter_words=1,
    progress_bar=True,
)

# Get narrative blocks based on the narrative_model and srl_res.

from narrativeNLP.wrappers import get_narratives

final_statements = get_narratives(
    srl_res=srl_res[0:1000],
    doc_index=split_sentences[0][0:1000],  # doc names
    narrative_model=narrative_model,
    save_to_disk=None,
    n_clusters=[0],  # pick model with 10 clusters
    cluster_labeling="most_frequent",
    progress_bar=True,
)

# Build narrative statements (preliminary)

from narrativeNLP.wrappers import prettify_narratives

final_statements = prettify_narratives(
    final_statements, narrative_model, filter_complete_narratives=True
)

# Validation (preliminary)

from narrativeNLP.validation import inspect_label, inspect_narrative

inspect_label(final_statements, label="democrat", role="ARGO")

inspect_narrative(final_statements, narrative="democrat steal election")

# Plot network (preliminary)

from narrativeNLP.graphs import build_graph, draw_graph

temp = final_statements[["ARGO", "ARG1", "B-V-RAW"]]
temp = temp.groupby(["ARGO", "ARG1", "B-V-RAW"]).size().reset_index(name="weight")
temp = temp.sort_values(by="weight", ascending=False).iloc[
    0:100
]  # pick top 100 most frequent narratives
temp = temp.to_dict(orient="records")

for l in temp:
    l["color"] = None

G = build_graph(
    dict_edges=temp, dict_args={}, edge_size=None, node_size=10, prune_network=True
)

draw_graph(G, notebook=False, output_filename="graph_test.html")
