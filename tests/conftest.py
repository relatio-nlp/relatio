"""
    Dummy conftest.py for demo_dsproject.
    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

# import pandas as pd
# import pytest
# import spacy

# from relatio.datasets import load_data
# from relatio.preprocessing import Preprocessor
# from relatio.semantic_role_labeling import SRL, extract_roles
# from relatio.utils import load_roles

# nlp = spacy.load("en_core_web_sm")
# stop_words = list(nlp.Defaults.stop_words)


# # 2000 documents for test
# @pytest.fixture(scope="session")
# def df_test():
#     return load_data(dataset="trump_tweet_archive", content="raw")[:2000]


# @pytest.fixture(scope="session")
# def df_split(p, df_test) -> pd.DataFrame:
#     df_split = p.split_into_sentences(df_test, output_path=None, progress_bar=False)
#     return df_split


# @pytest.fixture(scope="session")
# def p():
#     p = Preprocessor(
#         spacy_model="en_core_web_sm",
#         remove_punctuation=True,
#         remove_digits=True,
#         lowercase=True,
#         lemmatize=True,
#         remove_chars=[
#             '"',
#             "-",
#             "^",
#             ".",
#             "?",
#             "!",
#             ";",
#             "(",
#             ")",
#             ",",
#             ":",
#             "'",
#             "+",
#             "&",
#             "|",
#             "/",
#             "{",
#             "}",
#             "~",
#             "_",
#             "`",
#             "[",
#             "]",
#             ">",
#             "<",
#             "=",
#             "*",
#             "%",
#             "$",
#             "@",
#             "#",
#             "’",
#         ],
#         stop_words=stop_words,
#         n_process=-1,
#         batch_size=100,
#     )
#     return p


# @pytest.fixture(scope="session")
# def SRL_model():
#     return SRL(
#         path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
#         batch_size=10,
#         cuda_device=-1,
#     )


# @pytest.fixture(scope="session")
# def srl_res(df_split, SRL_model):
#     return SRL_model(df_split["sentence"], progress_bar=False)


# @pytest.fixture(scope="session")
# def roles(srl_res):
#     roles, sentence_index = extract_roles(
#         srl_res, used_roles=["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"], progress_bar=True
#     )
#     return roles


# @pytest.fixture(scope="session")
# def postproc_roles(p, roles):
#     postproc_roles = p.process_roles(
#         roles,
#         dict_of_pos_tags_to_keep={
#             "ARG0": ["PRON", "NOUN", "PROPN"],
#             "B-V": ["VERB"],
#             "ARG1": ["NOUN", "PROPN", "PRON"],
#         },
#         max_length=50,
#         progress_bar=False,
#         output_path=None,
#     )
#     return postproc_roles


# @pytest.fixture(scope="session")
# def known_entities(p, df_split):
#     known_entities = p.mine_entities(
#         df_split["sentence"],
#         clean_entities=True,
#         progress_bar=False,
#         output_path=None,
#     )
#     return known_entities
