# Browse list of available datasets

from relatio.datasets import list_datasets

print(list_datasets())

# Load an available dataset

from relatio.datasets import load_trump_data

df = load_trump_data("raw")

# Split into sentences (example on 100 tweets)

from relatio.utils import split_into_sentences

split_sentences = split_into_sentences(
    df.iloc[0:100], output_path=None, progress_bar=True
)

# Run SRL (example on a 100 tweets)

from relatio.wrappers import run_srl

srl_res = run_srl(
    path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
    sentences=split_sentences[1],
    output_path=None,
    progress_bar=True,
)

# As sentence splitting and SRL is time-consuming, we download the results from the datasets module.

split_sentences = load_trump_data("split_sentences")
srl_res = load_trump_data("srl_res")

# Build the narrative model
# This will take several minutes to run. You might want to grab a coffee.

from relatio.wrappers import build_narrative_model

narrative_model = build_narrative_model(
    srl_res=srl_res,
    sentences=split_sentences[1],  # list of sentences
    roles_considered=["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
    roles_with_embeddings=[["ARG0", "ARG1", "ARG2"]],
    embeddings_type="gensim_keyed_vectors",  # see documentation for a list of supported types
    embeddings_path="glove-wiki-gigaword-300",
    n_clusters=[[50, 100]],  # try different cluster numbers
    verbose=0,
    roles_with_entities=["ARG0", "ARG1", "ARG2"],
    top_n_entities=50,
    dimension_reduce_verbs=True,
    output_path=None,
    max_length=None,
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

from relatio.wrappers import get_narratives

final_statements = get_narratives(
    srl_res=srl_res,
    doc_index=split_sentences[0],  # doc names
    narrative_model=narrative_model,
    output_path=None,
    n_clusters=[0],  # pick model with 5O clusters
    cluster_labeling="most_frequent",
    progress_bar=True,
)

# Plot network (preliminary)

from relatio.graphs import build_graph, draw_graph

temp = final_statements[["ARG0_lowdim", "ARG1_lowdim", "B-V_lowdim"]]
temp.columns = ["ARG0", "ARG1", "B-V"]
temp = temp[(temp["ARG0"] != "") & (temp["ARG1"] != "") & (temp["B-V"] != "")]
temp = temp.groupby(["ARG0", "ARG1", "B-V"]).size().reset_index(name="weight")
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
