from collections import Counter
import numpy as np
import spacy
import time
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any
from spacy.cli import download as spacy_download

from relatio.utils import save_sentences, save_roles, save_entities


class Preprocessor:
    """

    A class to preprocess a given corpus
    (e.g., split it into sentences, annotate semantic roles, clean the text, mine named entities)
    
    Args:
        spacy_model: One of the available spacy models for the English language (default: en_core_web_sm). For a complete list, see: https://spacy.io/models/en#en_core_web_trf
        remove_punctuation: whether to remove string.punctuation
        remove_digits: whether to remove string.digits
        stop_words: list of stopwords to remove
        lowercase: whether to lower the case
        lemmatize: whether to lemmatize 
        n_process: Number of processes to user in nlp.pipe() for parallel computing (default: 1). Set to -1 to use all cores on the machine.
        batch_size: Size of the batches for parallel computing (default: 1000)

    """

    def __init__(
        self,
        spacy_model="en_core_web_sm",
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        stop_words: List[str] = [],
        lowercase: bool = True,
        lemmatize: bool = True,
        n_process: int = 1,
        batch_size: int = 1000,
    ):

        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)

        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe("sentencizer")
        self.n_process = n_process
        self.batch_size = batch_size
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.lemmatize = lemmatize

    def split_into_sentences(
        self,
        dataframe: pd.DataFrame,
        output_path: Optional[str] = None,
        progress_bar: bool = False,
    ) -> Tuple[List[str], List[str]]:

        """

        Split a list of documents into sentences (using the SpaCy sentence splitter).

        Args:
            dataframe: a pandas dataframe with one column "id" and one column "doc"
            output_path: path to save the output
            progress_bar: print a progress bar (default is False)

        Returns:
            Tuple with the list of document indices and list of sentences

        """

        sentences: List[str] = []
        doc_indices: List[str] = []

        length = len(dataframe["doc"])

        spacy_docs = self.nlp.pipe(
            dataframe["doc"],
            disable=["tagger", "ner", "parser", "lemmatizer"],
            batch_size=self.batch_size,
            n_process=self.n_process,
        )

        if progress_bar:
            print("Splitting into sentences...")
            time.sleep(1)
            spacy_docs = tqdm(spacy_docs, total=length)

        for i, doc in enumerate(spacy_docs):
            for sent in doc.sents:
                sentences.append(str(sent))
                doc_indices = doc_indices + [dataframe["id"].iloc[i]]

        if output_path is not None:
            save_sentences(doc_indices, sentences, output_path)

        return (doc_indices, sentences)

    def clean_text(self, s, pos_tags_to_keep: Optional[List[str]] = None) -> List[str]:

        """

        Clean a string of text.

        """

        if self.remove_punctuation:
            s = [t for t in s if t.is_punct == False]

        if self.remove_digits:
            s = [t for t in s if t.is_digit == False]

        if pos_tags_to_keep:
            s = [t for t in s if t.pos_ in pos_tags_to_keep]

        if self.lowercase and not self.lemmatize:
            s = [t.lower_ for t in s]

        if self.lowercase and self.lemmatize:
            s = [t.lemma_.lower() for t in s]

        if not self.lowercase and not self.lemmatize:
            s = [t.text for t in s]

        s = [t for t in s if t not in self.stop_words]

        s = [t.strip() for t in s if t not in self.stop_words]

        s = " ".join(s)

        return s

    def mine_entities(
        self,
        sentences: List[str],
        ent_labels: List[str] = ["PERSON", "NORP", "ORG", "GPE", "EVENT"],
        clean_entities: bool = True,
        output_path: Optional[str] = None,
        progress_bar: bool = False,
    ) -> Counter:

        """

        Go through sentences and counts named entities found in the corpus.

        Args:
            sentences: list of sentences
            ent_labels: list of entity labels to be considered (see SpaCy documentation)
            progress_bar: print a progress bar (default is False)
            For other arguments see utils.clean_text.

        Returns:
            Counter with the named entity and its associated frequency on the corpus

        """

        entities_all = []

        spacy_sentences = self.nlp.pipe(
            sentences, batch_size=self.batch_size, n_process=self.n_process
        )

        length = len(sentences)

        if progress_bar:
            print("Mining named entities...")
            time.sleep(1)
            spacy_sentences = tqdm(spacy_sentences, total=length)

        for sentence in spacy_sentences:
            for ent in sentence.ents:
                if ent.label_ in ent_labels:
                    entity = ent.text
                    if clean_entities:
                        entity = self.clean_text(ent)
                    entities_all.append(entity)

        entity_counts = Counter(entities_all)

        if output_path is not None:
            save_entities(entity_counts, output_path)

        return entity_counts

    def _extract_role_per_sentence(
        self, sentence_dict: dict, used_roles: List[str]
    ) -> List[Dict[str, Union[str, bool]]]:

        """

        Extract the semantic roles for a given sentence.

        Args:
            srl: srl output
            used_roles: list of semantic roles to extract

        Returns:
            List of statements with their associated roles for a given sentence

        """

        word_list = sentence_dict["words"]
        sentence_role_list = []

        for statement_dict in sentence_dict["verbs"]:
            tag_list = statement_dict["tags"]

            statement_role_dict: Dict[str, Union[str, bool]] = {}
            for role in ["ARG0", "ARG1", "ARG2", "B-V", "B-ARGM-MOD"]:
                if role in used_roles:
                    indices_role = [i for i, tok in enumerate(tag_list) if role in tok]
                    toks_role = [
                        tok for i, tok in enumerate(word_list) if i in indices_role
                    ]
                    statement_role_dict[role] = " ".join(toks_role)

            if "B-ARGM-NEG" in used_roles:
                role_negation_value = any("B-ARGM-NEG" in tag for tag in tag_list)
                statement_role_dict["B-ARGM-NEG"] = role_negation_value

            key_to_delete = []
            for key, value in statement_role_dict.items():
                if not value:
                    key_to_delete.append(key)
            for key in key_to_delete:
                del statement_role_dict[key]
            sentence_role_list.append(statement_role_dict)

        if not sentence_role_list:
            sentence_role_list = [{}]

        return sentence_role_list

    def extract_roles(
        self,
        srl: List[Dict[str, Any]],
        used_roles: List[str],
        progress_bar: bool = False,
    ) -> Tuple[List[Dict[str, Union[str, bool]]], List[int]]:

        """

        Extract semantic roles from the SRL output.

        Args:
            srl: srl output
            used_roles: list of semantic roles to extract
            progress_bar: print a progress bar (default is False)

        Returns:
            List of statements and numpy array of sentence indices (to keep track of sentences)

        """

        statements_role_list: List[Dict[str, Union[str, bool]]] = []
        sentence_index: List[int] = []

        if progress_bar:
            print("Extracting semantic roles...")
            time.sleep(1)
            srl = tqdm(srl)

        for i, sentence_dict in enumerate(srl):
            role_per_sentence = self._extract_role_per_sentence(
                sentence_dict, used_roles
            )
            sentence_index.extend([i] * len(role_per_sentence))
            statements_role_list.extend(role_per_sentence)

        return statements_role_list, np.asarray(sentence_index, dtype=np.uint32)

    def _make_list_of_roles(self, used_role, statements):

        list_of_roles = []
        indices = []

        for i, statement in enumerate(statements):
            role_content = statement.get(used_role)
            if role_content is not None:
                list_of_roles.append(role_content)
                indices.append(i)

        return indices, list_of_roles

    def _process_list_of_roles(
        self, list_of_roles, max_length, pos_tags_to_keep, progress_bar
    ):

        list_of_spacy_roles = self.nlp.pipe(
            list_of_roles, batch_size=self.batch_size, n_process=self.n_process
        )

        length = len(list_of_roles)

        if progress_bar:
            list_of_spacy_roles = tqdm(list_of_spacy_roles, total=length)

        list_of_clean_roles = []

        for phrase in list_of_spacy_roles:
            clean_phrase = self.clean_text(phrase, pos_tags_to_keep=pos_tags_to_keep)
            if max_length is not None:
                if len(clean_phrase) > max_length:
                    clean_phrase = ""
            list_of_clean_roles.append(clean_phrase)

        return list_of_clean_roles

    def process_roles(
        self,
        statements: List[Dict[str, List]],
        max_length: Optional[int] = None,
        dict_of_pos_tags_to_keep: Optional[dict] = None,
        output_path: Optional[str] = None,
        progress_bar: bool = False,
    ) -> List[Dict[str, List]]:

        """

        Takes a list of raw extracted semantic roles and cleans the text.

        Args:
            max_length = remove roles of more than n characters (NB: very long roles tend to be uninformative)
            progress_bar: print a progress bar (default is False)
            For other arguments see utils.clean_text.

        Returns:
            List of processed statements

        """

        pos_tags_to_keep = {
            "ARG0": None,
            "ARG1": None,
            "ARG2": None,
            "B-ARGM-MOD": None,
            "B-V": None,
        }
        if dict_of_pos_tags_to_keep is not None:
            for role in dict_of_pos_tags_to_keep.keys():
                pos_tags_to_keep[role] = dict_of_pos_tags_to_keep[role]

        length = len(statements)
        clean_statements = [{} for i in range(length)]

        for role in ["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"]:

            indices, list_of_roles = self._make_list_of_roles(role, statements)

            if role != "B-ARGM-NEG":
                print("Cleaning roles %s..." % role)
                list_of_roles = self._process_list_of_roles(
                    list_of_roles,
                    max_length=max_length,
                    pos_tags_to_keep=pos_tags_to_keep[role],
                    progress_bar=progress_bar,
                )

            for i, role_content in enumerate(list_of_roles):
                clean_statements[indices[i]][role] = role_content

        if output_path is not None:
            save_roles(clean_statements, output_path)

        return clean_statements

    def _clean_text_not_optimized(
        self, s, pos_tags_to_keep: Optional[List[str]] = None
    ) -> List[str]:

        """

        Clean a string of text.

        """

        s = self.nlp(s)

        if self.remove_punctuation:
            s = [t for t in s if t.is_punct == False]

        if self.remove_digits:
            s = [t for t in s if t.is_digit == False]

        if pos_tags_to_keep:
            s = [t for t in s if t.pos_ in pos_tags_to_keep]

        if self.lowercase and not self.lemmatize:
            s = [t.lower_ for t in s]

        if self.lowercase and self.lemmatize:
            s = [t.lemma_.lower() for t in s]

        if not self.lowercase and not self.lemmatize:
            s = [t.text for t in s]

        s = [t for t in s if t not in self.stop_words]

        s = [t.strip() for t in s if t not in self.stop_words]

        s = " ".join(s)

        return s

    def _process_roles_not_optimized(
        self,
        statements: List[Dict[str, List]],
        max_length: Optional[int] = None,
        dict_of_pos_tags_to_keep: Optional[dict] = None,
        output_path: Optional[str] = None,
        progress_bar: bool = False,
    ) -> List[Dict[str, List]]:

        """

        Takes a list of raw extracted semantic roles and cleans the text.

        Args:
            max_length = remove roles of more than n characters (NB: very long roles tend to be uninformative)
            progress_bar: print a progress bar (default is False)
            For other arguments see utils.clean_text.

        Returns:
            List of processed statements

        """

        roles_copy = deepcopy(statements)

        if progress_bar:
            print("Cleaning semantic roles...")
            time.sleep(1)
            statements = tqdm(statements)

        pos_tags_to_keep = {
            "ARG0": None,
            "ARG1": None,
            "ARG2": None,
            "B-ARGM-MOD": None,
            "B-V": None,
        }
        if dict_of_pos_tags_to_keep is not None:
            for role in dict_of_pos_tags_to_keep.keys():
                pos_tags_to_keep[role] = dict_of_pos_tags_to_keep[role]

        for i, statement in enumerate(statements):
            for role, role_content in statement.items():
                if isinstance(role_content, str):
                    res = self._clean_text_not_optimized(
                        role_content, pos_tags_to_keep=pos_tags_to_keep[role]
                    )
                    if max_length is not None:
                        if len(res) <= max_length:
                            roles_copy[i][role] = res
                        else:
                            roles_copy[i][role] = ""
                    else:
                        roles_copy[i][role] = res
                elif isinstance(role_content, bool):
                    pass
                else:
                    raise ValueError(f"{role_content}")

        if output_path is not None:
            save_roles(roles_copy, output_path)

        return roles_copy
