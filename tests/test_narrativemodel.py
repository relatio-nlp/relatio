# test for  NarrativeModel
import pytest
import spacy

from relatio.datasets import load_data
from relatio.narrative_models import NarrativeModel
from relatio.preprocessing import Preprocessor
from relatio.semantic_role_labeling import SRL, extract_roles

nlp = spacy.load("en_core_web_sm")
stop_words = list(nlp.Defaults.stop_words)


# prepare data for test, 500 documents for test
df_test = load_data(dataset="trump_tweet_archive", content="raw")[:500]

p = Preprocessor(
    spacy_model="en_core_web_sm",
    remove_punctuation=True,
    remove_digits=True,
    lowercase=True,
    lemmatize=True,
    remove_chars=[
        '"',
        "-",
        "^",
        ".",
        "?",
        "!",
        ";",
        "(",
        ")",
        ",",
        ":",
        "'",
        "+",
        "&",
        "|",
        "/",
        "{",
        "}",
        "~",
        "_",
        "`",
        "[",
        "]",
        ">",
        "<",
        "=",
        "*",
        "%",
        "$",
        "@",
        "#",
        "’",
    ],
    stop_words=stop_words,
    n_process=-1,
    batch_size=100,
)

df_split = p.split_into_sentences(df_test, output_path=None, progress_bar=False)

SRL_model = SRL(
    path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
    batch_size=10,
    cuda_device=-1,
)

srl_res = SRL_model(df_split["sentence"], progress_bar=False)

roles, sentence_index = extract_roles(
    srl_res,
    used_roles=["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
    progress_bar=True,
)

postproc_roles = p.process_roles(
    roles,
    dict_of_pos_tags_to_keep={
        "ARG0": ["PRON", "NOUN", "PROPN"],
        "B-V": ["VERB"],
        "ARG1": ["NOUN", "PROPN", "PRON"],
    },
    max_length=50,
    progress_bar=False,
    output_path=None,
)

known_entities = p.mine_entities(
    df_split["sentence"],
    clean_entities=True,
    progress_bar=False,
    output_path=None,
)

top_known_entities = [e[0] for e in list(known_entities.most_common(20)) if e[0] != ""]


# NarrativeModel(kmeans+MiniLM)
model_kmeans_MiniLM = NarrativeModel(
    clustering="kmeans",
    PCA=True,
    UMAP=True,
    roles_considered=["ARG0", "B-V", "B-ARGM-NEG", "ARG1", "ARG2"],
    roles_with_known_entities=["ARG0", "ARG1", "ARG2"],
    known_entities=top_known_entities,
    assignment_to_known_entities="character_matching",
    roles_with_unknown_entities=["ARG0", "ARG1", "ARG2"],
    threshold=0.3,
    embeddings_type="SentenceTransformer",
    embeddings_model="all-MiniLM-L6-v2",
)
model_kmeans_MiniLM.fit(
    postproc_roles,
    progress_bar=False,
    pca_args={"n_components": 15, "svd_solver": "full"},
)

# NarrativeModel(kmeans+m_BERT)
model_kmeans_m_BERT = NarrativeModel(
    clustering="kmeans",
    PCA=True,
    UMAP=True,
    roles_considered=["ARG0", "B-V", "B-ARGM-NEG", "ARG1", "ARG2"],
    roles_with_known_entities=["ARG0", "ARG1", "ARG2"],
    known_entities=top_known_entities,
    assignment_to_known_entities="character_matching",
    roles_with_unknown_entities=["ARG0", "ARG1", "ARG2"],
    threshold=0.3,
    embeddings_type="SentenceTransformer",
    embeddings_model="distiluse-base-multilingual-cased-v2",
)
model_kmeans_m_BERT.fit(
    postproc_roles,
    progress_bar=False,
    pca_args={"n_components": 15, "svd_solver": "full"},
)

# NarrativeModel(kmeans+globe)
model_kmeans_gensim = NarrativeModel(
    clustering="kmeans",
    PCA=True,
    UMAP=True,
    roles_considered=["ARG0", "B-V", "B-ARGM-NEG", "ARG1", "ARG2"],
    roles_with_known_entities=["ARG0", "ARG1", "ARG2"],
    known_entities=top_known_entities,
    assignment_to_known_entities="character_matching",
    roles_with_unknown_entities=["ARG0", "ARG1", "ARG2"],
    threshold=0.3,
    embeddings_type="GensimPretrained",
    embeddings_model="glove-twitter-25",
)
model_kmeans_gensim.fit(
    postproc_roles,
    progress_bar=False,
    pca_args={"n_components": 15, "svd_solver": "full"},
)

# NarrativeModel(kmeans+spacy)
model_kmeans_spacy = NarrativeModel(
    clustering="kmeans",
    PCA=True,
    UMAP=True,
    roles_considered=["ARG0", "B-V", "B-ARGM-NEG", "ARG1", "ARG2"],
    roles_with_known_entities=["ARG0", "ARG1", "ARG2"],
    known_entities=top_known_entities,
    assignment_to_known_entities="character_matching",
    roles_with_unknown_entities=["ARG0", "ARG1", "ARG2"],
    threshold=0.3,
    embeddings_type="spaCy",
    embeddings_model="en_core_web_md",
)
model_kmeans_spacy.fit(
    postproc_roles,
    progress_bar=False,
    pca_args={"n_components": 15, "svd_solver": "full"},
)

# NarrativeModel(kmeans+p_BERT)
model_kmeans_p_BERT = NarrativeModel(
    clustering="kmeans",
    PCA=True,
    UMAP=True,
    roles_considered=["ARG0", "B-V", "B-ARGM-NEG", "ARG1", "ARG2"],
    roles_with_known_entities=["ARG0", "ARG1", "ARG2"],
    known_entities=top_known_entities,
    assignment_to_known_entities="character_matching",
    roles_with_unknown_entities=["ARG0", "ARG1", "ARG2"],
    threshold=0.3,
    embeddings_type="SentenceTransformer",
    embeddings_model="whaleloops/phrase-bert",
)
model_kmeans_p_BERT.fit(
    postproc_roles,
    progress_bar=False,
    pca_args={"n_components": 15, "svd_solver": "full"},
)

# NarrativeModel(hdbscan+USE)
model_hdbscan_MiniLM = NarrativeModel(
    clustering="hdbscan",
    PCA=True,
    UMAP=True,
    roles_considered=["ARG0", "B-V", "B-ARGM-NEG", "ARG1", "ARG2"],
    roles_with_known_entities=["ARG0", "ARG1", "ARG2"],
    known_entities=top_known_entities,
    assignment_to_known_entities="character_matching",
    roles_with_unknown_entities=["ARG0", "ARG1", "ARG2"],
    threshold=0.3,
    embeddings_type="SentenceTransformer",
    embeddings_model="all-MiniLM-L6-v2",
)
model_hdbscan_MiniLM.fit(
    postproc_roles,
    progress_bar=False,
    pca_args={"n_components": 15, "svd_solver": "full"},
)


models = [
    model_kmeans_MiniLM,
    model_kmeans_m_BERT,
    model_kmeans_gensim,
    model_kmeans_p_BERT,
    model_hdbscan_MiniLM,
]


# test function for checking whether the narrrative outputs are consistent
@pytest.mark.parametrize("narrative_model", models)
def test_narrative_outputs(narrative_model):
    narratives_output = []
    for i in range(3):
        narratives = narrative_model.predict(postproc_roles, progress_bar=True)
        narratives_output.append(narratives)
    assert narratives_output[0] == narratives_output[1] == narratives_output[2]
