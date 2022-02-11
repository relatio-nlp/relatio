from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import numpy as np
import spacy
import time
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any


class Preprocessor:
    """

    A class to preprocess a given corpus

    (e.g., split it into sentences, annotate semantic roles, clean the text, mine named entities)

    """

    def __init__(self, spacy_model):
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)

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

        docs = dataframe.to_dict(orient="records")

        sentences: List[str] = []
        doc_indices: List[str] = []

        if progress_bar:
            print("Splitting into sentences...")
            time.sleep(1)
            docs = tqdm(docs)

        for doc in docs:
            for sent in self.nlp(doc["doc"], disable=["tagger", "ner"]).sents:
                sentences.append(str(sent))
                doc_indices = doc_indices + [doc["id"]]

        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump((doc_indices, sentences), f)

        return (doc_indices, sentences)

    def mine_entities(
        self,
        sentences: List[str],
        ent_labels: Optional[List[str]] = ["PERSON", "NORP", "ORG", "GPE", "EVENT"],
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

        if progress_bar:
            print("Mining named entities...")
            time.sleep(1)
            sentences = tqdm(sentences)

        for sentence in sentences:
            sentence = self.nlp(sentence)
            for ent in sentence.ents:
                if ent.label_ in ent_labels:
                    entities_all.append(ent.text)

        entity_counts = Counter(entities_all)

        return entity_counts

    def clean_text(
        self,
        sentence: str,
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        stop_words: List[str] = [],
        lowercase: bool = True,
        lemmatize: bool = True,
        pos_tags_to_keep: Optional[List[str]] = None,
    ) -> List[str]:

        """

        Clean a string of text.

        """

        s = self.nlp(sentence)

        if remove_punctuation:
            s = [t for t in s if t.is_punct == False]

        if remove_digits:
            s = [t for t in s if t.is_digit == False]

        if pos_tags_to_keep:
            s = [t for t in s if t.pos_ in pos_tags_to_keep]

        if lowercase and not lemmatize:
            s = [t.lower_ for t in s]

        if lowercase and lemmatize:
            s = [t.lemma_.lower() for t in s]

        if not lowercase and not lemmatize:
            s = [t.text for t in s]

        s = [t for t in s if t not in stop_words]

        s = [t.strip() for t in s if t not in stop_words]

        s = " ".join(s)

        return s

    def extract_role_per_sentence(
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
            role_per_sentence = self.extract_role_per_sentence(
                sentence_dict, used_roles
            )
            sentence_index.extend([i] * len(role_per_sentence))
            statements_role_list.extend(role_per_sentence)

        return statements_role_list, np.asarray(sentence_index, dtype=np.uint32)

    def process_roles(
        self,
        statements: List[Dict[str, List]],
        max_length: Optional[int] = None,
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        stop_words: List[str] = [],
        lowercase: bool = True,
        lemmatize: bool = True,
        dict_of_pos_tags_to_keep: Optional[dict] = None,
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
            for role, role_content in roles_copy[i].items():
                if isinstance(role_content, str):
                    res = self.clean_text(
                        role_content,
                        remove_punctuation=remove_punctuation,
                        remove_digits=remove_digits,
                        stop_words=stop_words,
                        lowercase=lowercase,
                        lemmatize=lemmatize,
                        pos_tags_to_keep=pos_tags_to_keep[role],
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

        return roles_copy
