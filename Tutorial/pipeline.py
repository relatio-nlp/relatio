import pandas as pd

import spacy
nlp = spacy.load("en_core_web_sm")

from collections import Counter
from sklearn.cluster import KMeans

import numpy as np
from numpy.linalg import norm

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Any
import numpy as np
from copy import deepcopy

import gensim.downloader as api

import warnings
import torch
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

import pickle as pk
import json
import time

import re
import string
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from copy import deepcopy
from tqdm import tqdm

# Utils
#..................................................................................................................
#..................................................................................................................


def split_into_sentences(
    dataframe,
    save_to_disk: Optional[str] = None,
    progress_bar: Optional[bool] = False,
):

    """

    A function that splits a list of documents into sentences (using the SpaCy sentence splitter).

    Args:
        dataframe: a pandas dataframe with one column "id" and one column "doc"
        progress_bar: print a progress bar (default is False)

    Returns:
        List of document ids and list of sentences

    """

    list_of_docs = dataframe.to_dict(orient = 'records')

    sentences = []
    doc_indices = []

    if progress_bar == True:
        print('Splitting into sentences...')
        time.sleep(1)
        list_of_docs = tqdm(list_of_docs)

    for doc_info in list_of_docs:
        for sent in nlp(doc_info['doc']).sents:
            sent = str(sent)
            sentences = sentences + [sent]
            doc_indices = doc_indices + [doc_info['id']]

    if save_to_disk is not None:
        with open(save_to_disk, 'w') as f:
            json.dump((doc_indices, sentences), f)

    return (doc_indices, sentences)


def remove_extra_whitespaces(
    s: str
) -> str:

    res = " ".join(s.split())

    return res


def _get_wordnet_pos(word):
    """Get POS tag"""
    tag = pos_tag([word])[0][1][0].upper()

    return tag


wnl = WordNetLemmatizer()
f_lemmatize = wnl.lemmatize


def preprocess(
    sentences: List[str],
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    stop_words: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_whitespaces: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
) -> List[str]:
    """
    Preprocess a list of sentences for word embedding.
    Args:
        sentence: list of sentences
        remove_punctuation: whether to remove string.punctuation
        remove_digits: whether to remove string.digits
        remove_chars: remove the given characters
        stop_words: list of stopwords to remove
        lowercase: whether to lower the case
        strip: whether to strip
        remove_whitespaces: whether to remove superfluous whitespaceing by " ".join(str.split(())
        lemmatize: whether to lemmatize using nltk.WordNetLemmatizer
        stem: whether to stem using nltk.SnowballStemmer("english")
        tags_to_keep: list of grammatical tags to keep (common tags: ['V', 'N', 'J'])
        remove_n_letter_words: drop words lesser or equal to n letters (default is None)
    Returns:
        Processed list of sentences
    Examples:
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'])
        ['return the factorial of n an exact integer']
        >>> preprocess(['Learning is usefull.'])
        ['learning is usefull']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], stop_words=['factorial'])
        ['return the of n an exact integer']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], lemmatize=True)
        ['return the factorial of n an exact integer']
        >>> preprocess(['Learning is usefull.'],lemmatize=True)
        ['learn be usefull']
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'], stem=True)
        ['return the factori of n an exact integ']
        >>> preprocess(['Learning is usefull.'],stem=True)
        ['learn is useful']
        >>> preprocess(['A1b c\\n\\nde \\t fg\\rkl\\r\\n m+n'])
        ['ab c de fg kl mn']
        >>> preprocess(['This is a sentence with verbs and nice adjectives.'], tags_to_keep = ['V', 'J'])
        ['is nice']
        >>> preprocess(['This is a sentence with one and two letter words.'], remove_n_letter_words = 2)
        ['this sentence with one and two letter words']
    """
    if lemmatize is True and stem is True:
        raise ValueError("lemmatize and stemming cannot be both True")
    if stop_words is not None and lowercase is False:
        raise ValueError("remove stop words make sense only for lowercase")

    # remove chars
    if remove_punctuation:
        remove_chars += string.punctuation
    if remove_digits:
        remove_chars += string.digits
    if remove_chars:
        sentences = [re.sub(f"[{remove_chars}]", "", str(sent)) for sent in sentences]

    # lowercase, strip and remove superfluous white spaces
    if lowercase:
        sentences = [sent.lower() for sent in sentences]
    if strip:
        sentences = [sent.strip() for sent in sentences]
    if remove_whitespaces:
        sentences = [" ".join(sent.split()) for sent in sentences]

    # lemmatize
    if lemmatize:

        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

        sentences = [
            " ".join(
                [
                    f_lemmatize(
                        word, tag_dict.get(_get_wordnet_pos(word), wordnet.NOUN)
                    )
                    for word in sent.split()
                ]
            )
            for sent in sentences
        ]

    # keep specific nltk tags
    # this step should be performed before stemming, but may be performed after lemmatization
    if tags_to_keep is not None:
        sentences = [
            " ".join(
                [
                    word
                    for word in sent.split()
                    if _get_wordnet_pos(word) in tags_to_keep
                ]
            )
            for sent in sentences
        ]

    # stem
    if stem:
        stemmer = SnowballStemmer("english")
        f_stem = stemmer.stem

        sentences = [
            " ".join([f_stem(word) for word in sent.split()]) for sent in sentences
        ]

    # drop stopwords
    # stopwords are dropped after the bulk of preprocessing steps, so they should also be preprocessed with the same standards
    if stop_words is not None:
        sentences = [
            " ".join([word for word in sent.split() if word not in stop_words])
            for sent in sentences
        ]

    # remove short words < n
    if remove_n_letter_words is not None:
        sentences = [
            " ".join(
                [word for word in sent.split() if len(word) > remove_n_letter_words]
            )
            for sent in sentences
        ]

    return sentences


def get_role_counts(
    statements: List[dict],
    roles: Optional[list] = ["B-V", "ARGO", "ARG1", "ARG2"],
    progress_bar: Optional[bool] = False
) -> dict:

    """

    Get role frequency within the corpus from preprocessed semantic roles. Roles considered are specified by the user.
    Args:
        statements: list of dictionaries of postprocessed semantic roles
        roles: list of roles considered
        progress_bar: print a progress bar (default is False)

    Returns:
        Dictionary in which postprocessed semantic roles are keys and their frequency within the corpus are values
        (e.g. d['verb'] = count)

    Example:
        >>> test = [{'B-V': ['increase'], 'B-ARGM-NEG': True},{'B-V': ['decrease']},{'B-V': ['decrease']}]\n
        ... verb_counts = get_role_counts(test, roles = ['B-V'])
        {'increase': 1, 'decrease': 2}

    """

    counts = {}

    if progress_bar == True:
        print('Computing role frequencies...')
        time.sleep(1)
        statements = tqdm(statements)

    for statement in statements:
        for key in statement.keys():
            if key in roles:
                temp = " ".join(statement[key])
                if temp in counts:
                    counts[temp] += 1
                else:
                    counts[temp] = 1

    return counts


# Semantic Role Labeling
#..................................................................................................................
#..................................................................................................................

# link to choose the SRL model
# https://storage.googleapis.com/allennlp-public-models/YOUR-PREFERRED-MODEL


def filter_sentences(
    sentences: List[str],
    max_sentence_length: Optional[int] = None,
    max_number_words: Optional[int] = None,
) -> List[str]:

    """

    Filter list of sentences based on the number of characters length.
    Args:
        max_sentence_length: Keep only sentences with a a number of character lower or equal to max_sentence_length. For max_number_words = max_sentence_length = -1 all sentences are kept.
        max_number_words: Keep only sentences with a a number of words lower or equal to max_number_words. For max_number_words = max_sentence_length = -1 all sentences are kept.
    Returns:
        Filtered list of sentences.
    Examples:
        >>> filter_sentences(['This is a house'])
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_sentence_length=15)
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_sentence_length=14)
        []
        >>> filter_sentences(['This is a house'], max_number_words=4)
        ['This is a house']
        >>> filter_sentences(['This is a house'], max_number_words=3)
        []
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=18)
        ['This is a house', 'It is a nice house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=4, max_sentence_length=18)
        ['This is a house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=17)
        ['This is a house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=0, max_sentence_length=18)
        []
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=0)
        []
        >>> filter_sentences(['This is a house', 'It is a nice house'])
        ['This is a house', 'It is a nice house']
        >>> filter_sentences(['This is a house', 'It is a nice house'], max_number_words=4)
        ['This is a house']

    """

    if max_sentence_length is None and max_number_words is None:
        pass
    elif max_sentence_length == 0 or max_number_words == 0:
        sentences = []
    else:
        if max_sentence_length is not None:
            sentences = [sent for sent in sentences if len(sent) <= max_sentence_length]

            def filter_funct(sent):
                return len(sent) <= max_sentence_length

        if max_number_words is not None:
            sentences = [
                sent for sent in sentences if len(sent.split()) <= max_number_words
            ]

    return sentences


def group_sentences_in_batches(
    sentences: List[str],
    max_batch_char_length: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> List[List[str]]:

    """

    Group sentences in batches of given total character length.
    Args:
        sentences: List of sentences
        max_batch_char_length: maximum char length for a batch
    Returns:
        List of batches (list) of sentences.
    Examples:
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=15)
        [['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=14)
        []
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=29)
        [['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=30)
        [['This is a house', 'This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'])
        [['This is a house', 'This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], max_batch_char_length=29)
        [['This is a house'], ['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], batch_size=2)
        [['This is a house', 'This is a house'], ['This is a house']]

    """

    batches: List[List[str]] = []

    if max_batch_char_length is None and batch_size is None:
        batches = [sentences]
    elif max_batch_char_length is not None and batch_size is not None:
        raise ValueError("max_batch_char_length and batch_size are mutual exclusive.")
    elif batch_size is not None:
        batches = [
            sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)
        ]
    else:
        batch_char_length = 0
        batch: List[str] = []

        for el in sentences:
            length = len(el)
            batch_char_length += length
            if length > max_batch_char_length:
                warnings.warn(
                    f"The length of the sentence = {length} > max_batch_length={max_batch_char_length}. The following sentence is skipped: \n > {el}",
                    RuntimeWarning,
                )
                continue
            if batch_char_length > max_batch_char_length:
                batches.append(batch)
                batch = [el]
                batch_char_length = length
            else:
                batch.append(el)

        if batch:
            batches.append(batch)

    return batches


class SRL:
    def __init__(
        self,
        path: str,
        cuda_device: int = -1,
        max_batch_char_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_sentence_length: Optional[int] = None,
        max_number_words: Optional[int] = None,
        cuda_empty_cache: bool = True,
        cuda_sleep: float = 0.0,
    ):
        self._predictor = Predictor.from_path(path, cuda_device=cuda_device)
        self._max_batch_char_length = max_batch_char_length
        self._batch_size = batch_size
        self._max_sentence_length = max_sentence_length
        self._max_number_words = max_number_words
        self._cuda_empty_cache = cuda_empty_cache
        self._cuda_device = cuda_device
        self._cuda_sleep = cuda_sleep

    def _clean_cache(self, cuda_sleep, cuda_empty_cache):
        if self._cuda_device > -1 and cuda_empty_cache:
            with torch.cuda.device(self._cuda_device):
                torch.cuda.empty_cache()
                time.sleep(cuda_sleep)

    def __call__(
        self,
        sentences: List[str],
        max_batch_char_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_sentence_length: Optional[int] = None,
        max_number_words: Optional[int] = None,
        cuda_empty_cache: bool = None,
        cuda_sleep: float = None,
        progress_bar: Optional[bool] = False
    ):
        max_batch_char_length = (
            max_batch_char_length
            if max_batch_char_length is not None
            else self._max_batch_char_length
        )

        batch_size = batch_size if batch_size is not None else self._batch_size

        max_sentence_length = (
            max_sentence_length
            if max_sentence_length is not None
            else self._max_sentence_length
        )

        max_number_words = (
            max_number_words if max_number_words is not None else self._max_number_words
        )

        cuda_empty_cache = (
            cuda_empty_cache if cuda_empty_cache is not None else self._cuda_empty_cache
        )

        cuda_sleep = cuda_sleep if cuda_sleep is not None else self._cuda_sleep

        sentences = filter_sentences(
            sentences,
            max_sentence_length=max_sentence_length,
            max_number_words=max_number_words,
        )

        batches = group_sentences_in_batches(
            sentences,
            max_batch_char_length=max_batch_char_length,
            batch_size=batch_size,
        )

        res = []

        if progress_bar == True:
            print('Running SRL...')
            time.sleep(1)
            batches = tqdm(batches)

        for batch in batches:
            sentences_json = [{"sentence": sent} for sent in batch]
            try:
                res_batch = self._predictor.predict_batch_json(sentences_json)
            except RuntimeError as err:
                warnings.warn(
                    f"empty result {err}",
                    RuntimeWarning,
                )
                res = [None]
                break
            except:
                raise
            finally:
                self._clean_cache(cuda_sleep, cuda_empty_cache)

            res.extend(res_batch)
        return res


def extract_roles(
    srl: List[Dict[str, Any]],
    UsedRoles: List[str],
    progress_bar: Optional[bool] = False
) -> Tuple[List[Dict[str, List]], List[int]]:

    """

    A function that extracts semantic roles from the SRL output.

    Args:
        srl: srl output
        UsedRoles: list of roles
        progress_bar: print a progress bar (default is False)

    Returns:
        List of statements and numpy array of sentence indices (to keep track of sentences)

    """

    statements_role_list: List[Dict[str, List]] = []
    sentence_index: List[int] = []

    if progress_bar == True:
        print('Processing SRL...')
        time.sleep(1)
        srl = tqdm(srl)

    for i, sentence_dict in enumerate(srl):
        role_per_sentence = extract_role_per_sentence(sentence_dict, UsedRoles)
        sentence_index.extend([i] * len(role_per_sentence))
        statements_role_list.extend(role_per_sentence)

    return statements_role_list, np.asarray(sentence_index, dtype=np.uint32)


def extract_role_per_sentence(
    sentence_dict: dict,
    UsedRoles: List[str]
) -> List[dict]:

    """

    A function that extracts the semantic roles for a given sentence.

    Args:
        srl: srl output
        UsedRoles: list of roles

    Returns:
        List of statements with their associated roles for a given sentence

    """

    word_list = sentence_dict["words"]
    sentence_role_list = []

    for statement_dict in sentence_dict["verbs"]:
        tag_list = statement_dict["tags"]

        statement_role_dict = {}

        if 'ARGO' in UsedRoles:
            indices_agent = [i for i, tok in enumerate(tag_list) if "ARG0" in tok]
            agent = [tok for i, tok in enumerate(word_list) if i in indices_agent]
            statement_role_dict["ARGO"] = agent

        if 'ARG1' in UsedRoles:
            indices_patient = [i for i, tok in enumerate(tag_list) if "ARG1" in tok]
            patient = [tok for i, tok in enumerate(word_list) if i in indices_patient]
            statement_role_dict["ARG1"] = patient

        if 'ARG2' in UsedRoles:
            indices_attribute = [i for i, tok in enumerate(tag_list) if "ARG2" in tok]
            attribute = [tok for i, tok in enumerate(word_list) if i in indices_attribute]
            statement_role_dict["ARG2"] = attribute

        if 'B-V' in UsedRoles:
            indices_verb = [i for i, tok in enumerate(tag_list) if "B-V" in tok]
            verb = [tok for i, tok in enumerate(word_list) if i in indices_verb]
            statement_role_dict["B-V"] = verb

        if 'B-ARGM-MOD' in UsedRoles:
            indices_modal = [i for i, tok in enumerate(tag_list) if "B-ARGM-MOD" in tok]
            modal = [tok for i, tok in enumerate(word_list) if i in indices_modal]
            statement_role_dict["B-ARGM-MOD"] = modal

        if 'B-ARGM-NEG' in UsedRoles:
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


def postprocess_roles(
    statements: List[Dict[str, List]],
    max_length: Optional[int] = None,
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    stop_words: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_whitespaces: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
    progress_bar: Optional[bool] = False
) -> List[Dict[str, List]]:

    """

    max_length = remove roles of more than n tokens (NB: very long roles tend to be uninformative in our context)
    progress_bar: print a progress bar (default is False)
    For other arguments see utils.preprocess.

    """

    roles_copy = deepcopy(statements)

    if progress_bar == True:
        print('Cleaning SRL...')
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, tokens in roles_copy[i].items():
            if isinstance(tokens, list):
                res = [
                    preprocess(
                        [" ".join(tokens)],
                        remove_punctuation=remove_punctuation,
                        remove_digits=remove_digits,
                        remove_chars=remove_chars,
                        stop_words=stop_words,
                        lowercase=lowercase,
                        strip=strip,
                        remove_whitespaces=remove_whitespaces,
                        lemmatize=lemmatize,
                        stem=stem,
                        tags_to_keep=tags_to_keep,
                        remove_n_letter_words=remove_n_letter_words,
                    )[0].split()
                ][0]
                if max_length is not None:
                    if len(res) <= max_length:
                        roles_copy[i][role] = res
                    else:
                        roles_copy[i][role] = []
                else:
                    roles_copy[i][role] = res
            elif isinstance(tokens, bool):
                pass
            else:
                raise ValueError(f"{tokens}")
    return roles_copy


# Named Entity Recognition
#..................................................................................................................
#..................................................................................................................


def mine_entities(
    sentences: List[str],
    ent_labels: Optional[List[str]] = ['PERSON', 'NORP', 'ORG', 'GPE', 'EVENT'],
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    stop_words: Optional[List[str]] = None,
    lowercase: bool = True,
    strip: bool = True,
    remove_whitespaces: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
    progress_bar: Optional[bool] = True
) -> List[Tuple[str, int]]:

    """

    A function that goes through sentences and counts named entities found in the corpus.

    Args:
        sentences: list of sentences
        ent_labels: list of entity labels to be considered (see SPaCy documentation)
        progress_bar: print a progress bar (default is False)
        For other arguments see utils.preprocess.

    Returns:
        List of tuples with the named entity and its associated frequency on the corpus

    """

    entities_all = []

    if progress_bar == True:
        print('Mining named entities...')
        time.sleep(1)
        sentences = tqdm(sentences)

    for sentence in sentences:
        sentence = nlp(sentence)
        for ent in sentence.ents:
            if ent.label_ in ent_labels:
                entity = [ent.text]
                entities_all = entity + entities_all

    entities_all = preprocess(entities_all,
                              remove_punctuation,
                              remove_digits,
                              remove_chars,
                              stop_words,
                              lowercase,
                              strip,
                              remove_whitespaces,
                              lemmatize,
                              stem,
                              tags_to_keep,
                              remove_n_letter_words)

    entity_counts = Counter(entities_all)
    entities_sorted = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)

    # forgetting to remove those will break the pipeline
    entities_sorted = [entity for entity in entities_sorted if entity[0] != ''] 
    
    return entities_sorted


def pick_top_entities(
    entities_sorted: List[Tuple[str,int]],
    top_n_entities: Optional[int] = 0
) -> List[str]:

    """

    A function that returns the top n most frequent named entities in the corpus.

    Args:
        entities_sorted: list of tuples (named_entity, frequency)
        top_n_entities: number of named entities to keep (default is all and is specified with top_n = 0)

    Returns:
        List of most frequent named entities

    """

    entities = []

    for entity in entities_sorted:
        entities = entities + [entity[0]]

    if top_n_entities == 0:
        top_n_entities = len(entities_sorted)

    return entities[0:top_n_entities]


def is_subsequence(
    v2: list,
    v1: list
) -> bool:

    """

    Check whether v2 is a subsequence of v1.

    Args:
        v2/v1: lists of elements

    Returns:
        a boolean

    Example:
        >>> v1 = ['the', 'united', 'states', 'of', 'america']\n
        ... v2 = ['united', 'states', 'of', 'europe']\n
        ... is_subsequence(v2,v1)
        False

    """
    it = iter(v1)
    return all(c in it for c in v2)


def map_entities(
    statements: List[dict],
    entities: list,
    UsedRoles: List[str],
    progress_bar: Optional[bool] = False
) -> Tuple[dict, List[dict]]:

    """

    A function that goes through statements and identifies pre-defined named entities within postprocessed semantic roles.

    Args:
        statements: list of dictionaries of postprocessed semantic roles
        entities: user-defined list of named entities
        roles: a list of roles with named entities (default = ARG0 and ARG1)
        UsedRoles: list of roles for named entity recognition
        progress_bar: print a progress bar (default is False)

    Returns:
        entity_index: dictionary containing statements indices with entities for each role
        roles_copy: new list of postprocessed semantic roles (without the named entities mined since they will not be embedded)

    """

    entity_index = {role:{entity:np.asarray([], dtype=int) for entity in entities} for role in UsedRoles}

    roles_copy = deepcopy(statements)

    if progress_bar == True:
        print('Mapping named entities...')
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, tokens in roles_copy[i].items():
            if role in UsedRoles:
                for entity in entities:
                    if is_subsequence(entity.split(), tokens) == True:
                        entity_index[role][entity] = np.append(entity_index[role][entity], [i])
                        roles_copy[i][role] = []

    return entity_index, roles_copy


# Clean Verbs
#..................................................................................................................
#..................................................................................................................


def find_synonyms(verb: str) -> List[str]:

    """

    Find synonyms of a given word based on wordnet.
    Args:
        verb: a verb
    Returns:
        a list of synonyms
    Example:
        >>> find_synonyms('fight')
        ['contend', 'fight', 'struggle', 'fight', 'oppose', 'fight_back', 'fight_down', 'defend', 'fight', 'struggle', 'crusade', 'fight', 'press', 'campaign', 'push', 'agitate']

    """

    synonyms = []

    for syn in wordnet.synsets(verb, pos=wordnet.VERB):
        for l in syn.lemmas():
            synonyms.append(l.name())

    return synonyms


def find_antonyms(verb: str) -> List[str]:

    """

    Find antonyms of a given word based on wordnet.
    Args:
        verb: a verb
    Returns:
        a list of antonyms
    Example:
        >>> find_antonyms('break')
        ['repair', 'keep', 'conform_to', 'make', 'promote']

    """

    antonyms = []

    for syn in wordnet.synsets(verb, pos=wordnet.VERB):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    return antonyms


def get_most_frequent(tokens: List[str], token_counts: dict) -> str:

    """

    Find most frequent token in a list of tokens.

    Args:
        tokens: a list of tokens
        token_counts: a dictionary of token frequencies
    Returns:
        the most frequent token in the list of tokens

    """

    freq = 0
    most_freq_token = None

    for candidate in tokens:
        if candidate in token_counts:
            if token_counts[candidate] > freq:
                freq = token_counts[candidate]
                most_freq_token = candidate

    return most_freq_token


def clean_verbs(statements: List[dict], verb_counts: dict) -> List[dict]:

    """

    Replace verbs by their most frequent synonym or antonym.
    If a verb is combined with a negation in the statement (e.g. 'not increase'),
    it is replaced by its most frequent antonym and the negation is removed (e.g. "decrease").
    Args:
        statements: a list of dictionaries of postprocessed semantic roles
        verb_counts: a dictionary of verb counts (e.g. d['verb'] = count)
    Returns:
        a list of dictionaries of postprocessed semantic roles with replaced verbs (same format as statements)

    Example:
        >>> test = [{'B-V': ['increase'], 'B-ARGM-NEG': True},{'B-V': ['decrease']},{'B-V': ['decrease']}]\n
        ... verb_counts = get_role_counts(test, roles = ['B-V'])\n
        ... clean_verbs(test, verb_counts = verb_counts)
        [{'B-V-CLEANED': 'decrease'}, {'B-V-CLEANED': 'decrease'}, {'B-V-CLEANED': 'decrease'}]

    """

    new_roles_all = []

    for roles in statements:
        new_roles = deepcopy(roles)
        new_roles = {str(k + '-CLEANED'): v for k, v in new_roles.items() if k in ['B-V', 'B-ARGM-NEG']}
        if "B-V" in roles:
            verb = " ".join(new_roles["B-V-CLEANED"])
            new_roles["B-V-CLEANED"] = verb
            if "B-ARGM-NEG" in roles:
                verbs = find_antonyms(verb)
                most_freq_verb = get_most_frequent(tokens = verbs, token_counts = verb_counts)
                if most_freq_verb is not None:
                    new_roles["B-V-CLEANED"] = most_freq_verb
                    del new_roles["B-ARGM-NEG-CLEANED"]
            else:
                verbs = find_synonyms(verb) + [verb]
                most_freq_verb = get_most_frequent(tokens = verbs, token_counts = verb_counts)
                if most_freq_verb is not None:
                    new_roles["B-V-CLEANED"] = most_freq_verb
        new_roles_all.append(new_roles)

    return new_roles_all


# Vectors and Clustering
#..................................................................................................................
#..................................................................................................................

def count_words(
    sentences: List[str]
) -> dict:

    """

    A function that computes word frequencies in a list of sentences.

    Args:
        sentences: list of sentences

    Returns:
        A dictionary {"word": frequency}

    """

    words = []

    for sentence in sentences:
        words = words + sentence.split()

    word_count_dict = dict(Counter(words))

    return word_count_dict


def compute_sif_weights(
    word_count_dict: dict,
    alpha: Optional[float] = 0.001
) -> dict:

    """

    A function that computes SIF weights based on word frequencies.

    Args:
        word_count_dict: a dictionary {"word": frequency}
        alpha: regularization parameter (see original paper)

    Returns:
        A dictionary {"word": SIF weight}

    """

    sif_dict = {}

    for word, count in word_count_dict.items():
        sif_dict[word] = alpha / (alpha + count)

    return sif_dict


class USE:
    def __init__(self, path: str):
        self._embed = hub.load(path)

    def __call__(self, tokens: List[str]) -> np.ndarray:
        return self._embed([" ".join(tokens)]).numpy()[0]


class SIF_word2vec:
    def __init__(
        self, path: str, sentences = List[str], alpha: Optional[float] = 0.001, normalize: bool = True
    ):

        self._model = Word2Vec.load(path)

        self._word_count_dict = count_words(sentences)

        self._sif_dict = compute_sif_weights(self._word_count_dict, alpha)

        self._vocab = self._model.wv.vocab

        self._normalize = normalize

    def __call__(self, tokens: List[str]):
        res = np.mean(
            [self._sif_dict[token] * self._model.wv[token] for token in tokens], axis=0
        )
        if self._normalize:
            res = res / norm(res)
        return res

    def most_similar(self, v):
        return self._model.wv.most_similar(positive=[v], topn=1)[0]


class SIF_keyed_vectors:
    def __init__(
        self, path: str, sentences = List[str], alpha: Optional[float] = 0.001, normalize: bool = True
    ):

        self._model = api.load(path)

        self._word_count_dict = count_words(sentences)

        self._sif_dict = compute_sif_weights(self._word_count_dict, alpha)

        self._vocab = self._model.vocab

        self._normalize = normalize

    def __call__(self, tokens: List[str]):
        res = np.mean(
            [self._sif_dict[token] * self._model[token] for token in tokens], axis=0
        )
        if self._normalize:
            res = res / norm(res)
        return res

    def most_similar(self, v):
        return self._model.most_similar(positive=[v], topn=1)[0]


def get_vector(
    tokens: List[str],
    model: Union[USE, SIF_word2vec, SIF_keyed_vectors]
):

    """

    A function that computes an embedding vector for a list of tokens.

    Args:
        tokens: list of string tokens to embed
        model: trained embedding model
        (e.g. either Universal Sentence Encoders, a full gensim Word2Vec model or gensim Keyed Vectors)

    Returns:
        A two-dimensional numpy array (1,dimension of the embedding space)

    """

    if not isinstance(model, (USE, SIF_word2vec, SIF_keyed_vectors)):
        raise TypeError("Union[USE, SIF_Word2Vec, SIF_keyed_vectors]")

    if isinstance(model, SIF_word2vec) or isinstance(model, SIF_keyed_vectors):
        if not tokens:
            res = None
        elif any(token not in model._sif_dict for token in tokens):
            res = None
        elif any(token not in model._vocab for token in tokens):
            res = None
        else:
            res = model(tokens)
            res = np.array([res]) # correct format to feed the vectors to sklearn clustering methods
    else:
            res = model(tokens)
            res = np.array([res]) # correct format to feed the vectors to sklearn clustering methods

    return res


def train_cluster_model(
    postproc_roles,
    model: Union[USE, SIF_word2vec, SIF_keyed_vectors],
    n_clusters,
    UsedRoles = List[str],
    random_state: Optional[int] = 0,
    verbose: Optional[int] = 0
):

    """

    A function to train a kmeans model on the corpus.

    Args:
        postproc_roles: list of statements
        model: trained embedding model
        (e.g. either Universal Sentence Encoders, a full gensim Word2Vec model or gensim Keyed Vectors)
        n_clusters: number of clusters
        UsedRoles: list of roles
        random_state: seed for replication (default is 0)
        verbose: see Scikit-learn documentation for details

    Returns:
        A sklearn kmeans model

    """

    role_counts = get_role_counts(postproc_roles, roles = UsedRoles)

    role_counts = [role.split() for role in list(role_counts)]

    vecs = None
    for role in role_counts:
        if vecs is None:
            vecs = get_vector(role, model)
        else:
            temp = get_vector(role, model)
            if temp is not None:
                vecs = np.concatenate((vecs, temp), axis=0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, verbose = verbose).fit(vecs)

    return kmeans


def get_clusters(
    postproc_roles: List[dict],
    model: Union[USE, SIF_word2vec, SIF_keyed_vectors],
    kmeans,
    UsedRoles = List[str],
    progress_bar: Optional[bool] = False
) -> List[dict]:

    """

    A function which predicts clusters based on a pre-trained kmeans model.

    Args:
        postproc_roles: list of statements
        model: trained embedding model
        (e.g. either Universal Sentence Encoders, a full gensim Word2Vec model or gensim Keyed Vectors)
        kmeans = a pre-trained sklearn kmeans model
        UsedRoles: list of roles

    Returns:
        A list of dictionaries with the predicted cluster for each role

    """

    clustering_res = []

    if progress_bar == True:
        print('Assigning clusters to roles...')
        time.sleep(1)
        postproc_roles = tqdm(postproc_roles)

    for statement in postproc_roles:
        temp = {}
        for role, tokens in statement.items():
            if role in UsedRoles:
                vec = get_vector(tokens, model)
                if vec is not None:
                    clu = kmeans.predict(vec)
                    temp[role] = int(clu)
        clustering_res = clustering_res + [temp]

    return clustering_res


def label_clusters_most_freq(
    clustering_res: List[dict],
    postproc_roles: List[dict]
) -> dict:

    """

    A function which labels clusters by their most frequent term.

    Args:
        clustering_res: list of dictionaries with the predicted cluster for each role
        postproc_roles: list of statements

    Returns:
        A dictionary associating to each cluster number a label (e.g. the most frequent term in this cluster)

    """

    temp = {}
    labels = {}

    for i,statement in enumerate(clustering_res):
        for role, cluster in statement.items():
            tokens = ' '.join(postproc_roles[i][role])
            cluster_num = cluster
            if cluster_num not in temp:
                temp[cluster_num] = [tokens]
            else:
                temp[cluster_num] = temp[cluster_num] + [tokens]

    for cluster_num, tokens in temp.items():
        token_counts = Counter(tokens)
        token_freq = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        most_freq_token = token_freq[0][0]
        labels[cluster_num] = most_freq_token

    return labels


def label_clusters_most_similar(
    kmeans,
    model
) -> dict:

    """

    A function which labels clusters by the term closest to the centroid in the embedding
    (i.e. distance is cosine similarity)

    Args:
        kmeans: the trained kmeans model
        model: trained embedding model
        (e.g. a full gensim Word2Vec model or gensim Keyed Vectors)

    Returns:
        A dictionary associating to each cluster number a label
        (e.g. the most similar term to cluster's centroid)

    """

    labels = {}

    for i, vec in enumerate(kmeans.cluster_centers_):
        most_similar_term = model.most_similar(vec)
        labels[i] = most_similar_term[0]

    return labels


# Wrappers
#..................................................................................................................
#..................................................................................................................


def build_narratives(
    final_statements,
    narrative_model: dict,
    filter_complete_narratives: Optional[bool] = True
):

    """

    A function to make columns of 'raw' and 'cleaned' narratives.

    Args:
        final_statements: dataframe with the output of the pipeline
        narrative_model: dict with the specifics of the narrative model
        filter_complete_narratives: keep only narratives with at least an agent, a verb and a patient
        (default is True)

    Returns:
        A pandas dataframe with the resulting narratives and two additional columns:
        narrative-RAW and narrative-CLEANED

    """

    narrative_format = [str(role + '-RAW') for role in narrative_model['roles_considered']]

    final_statements = final_statements.replace({'': np.NaN})

    if filter_complete_narratives:
        list_for_filter = [
            arg for arg in narrative_format if arg not in [
                'ARG2-RAW',
                'B-ARGM-NEG-RAW',
                'B-ARGM-MOD-RAW'
            ]
        ]
        final_statements = final_statements.dropna(subset=list_for_filter)

    final_statements = final_statements.replace({np.NaN: ''})
    final_statements = final_statements.replace({True: 'not'})

    final_statements['narrative-RAW'] = final_statements[narrative_format].agg(' '.join, axis=1)
    final_statements['narrative-RAW'] = final_statements['narrative-RAW'].apply(remove_extra_whitespaces)

    narrative_format = []
    for role in narrative_model['roles_considered']:
        if role == 'B-V':
            if narrative_model['dimension_reduce_verbs'] == True:
                narrative_format = narrative_format + ['B-V-CLEANED']
                narrative_format = narrative_format + ['B-ARGM-NEG-CLEANED']
            else:
                narrative_format = narrative_format + ['B-V-RAW']
                narrative_format = narrative_format + ['B-ARGM-NEG-RAW']

        elif role == 'B-ARGM-NEG':
            continue

        elif role == 'B-ARGM-MOD':
            narrative_format = narrative_format + ['B-ARGM-MOD-RAW']

        else:
            if narrative_model['roles_with_embeddings'] is not None or narrative_model['roles_with_entities'] is not None:
                narrative_format = narrative_format + [role]
            else:
                narrative_format = narrative_format + [str(role + '-RAW')]

    final_statements['narrative-CLEANED'] = final_statements[narrative_format].agg(' '.join, axis=1)
    final_statements['narrative-CLEANED'] = final_statements['narrative-CLEANED'].apply(remove_extra_whitespaces)

    # Re-ordering columns
    columns = ['doc', 'sentence', 'statement', 'narrative-CLEANED', 'narrative-RAW']
    for role in narrative_model['roles_considered']:
        if role in ['ARGO', 'ARG1', 'ARG2']:
            columns = columns + [str(role +'-RAW')]
            columns = columns + [role]
        elif role == 'B-ARGM-MOD':
            columns = columns + [str(role +'-RAW')]
        else:
            columns = columns + [str(role +'-RAW')]
            columns = columns + [str(role + '-CLEANED')]

    final_statements = final_statements[columns]

    return final_statements


def run_srl(
    path: str,
    sentences: List[str],
    max_batch_char_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_sentence_length: Optional[int] = None,
    max_number_words: Optional[int] = None,
    cuda_empty_cache: bool = None,
    cuda_sleep: float = None,
    save_to_disk: Optional[str] = None,
    progress_bar: Optional[bool] = False
):

    """

    A wrapper function to run semantic role labeling on a corpus.

    Args:
        path: location of the SRL model to be used
        sentences: list of sentences
        SRL_options: see class SRL()
        save_to_disk: path to save the narrative model (default is None, which means no saving to disk)
        progress_bar: print a progress bar (default is False)

    Returns:
        A list of dictionaries with the SRL output

    """

    srl = SRL(path=path)

    srl_res = srl(sentences=sentences, batch_size = batch_size, progress_bar = progress_bar)

    if save_to_disk is not None:
        with open(save_to_disk, 'w') as json_file:
            json.dump(srl_res, json_file)

    return srl_res


def build_narrative_model(
    srl_res: List[dict],
    sentences: List[str],
    roles_considered: Optional[List[str]] = ['ARGO', 'B-V', 'B-ARGM-NEG', 'B-ARGM-MOD', 'ARG1', 'ARG2'],
    save_to_disk: Optional[str] = None,
    max_length: Optional[int] = None,
    remove_punctuation: Optional[bool] = True,
    remove_digits: Optional[bool] = True,
    remove_chars: Optional[str] = "",
    stop_words: Optional[List[str]] = None,
    lowercase: Optional[bool] = True,
    strip: Optional[bool] = True,
    remove_whitespaces: Optional[bool] = True,
    lemmatize: Optional[bool] = False,
    stem: Optional[bool] = False,
    tags_to_keep: Optional[List[str]] = None,
    remove_n_letter_words: Optional[int] = None,
    roles_with_embeddings: Optional[List[List[str]]] = [['ARGO','ARG1', 'ARG2']],
    embeddings_type: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    n_clusters: Optional[int] = [0],
    verbose: Optional[int] = 0,
    roles_with_entities: Optional[List[str]] = ['ARGO', 'ARG1', 'ARG2'],
    ent_labels: Optional[List[str]] = ['PERSON', 'NORP', 'ORG', 'GPE', 'EVENT'],
    top_n_entities: Optional[int] = 0,
    dimension_reduce_verbs: Optional[bool] = True,
    progress_bar: Optional[bool] = False
):

    """

    A wrapper function to build the narrative model from a sample of of the corpus.

    Args:
        srl_res: sentences labeled with their semantic roles
        sentences: list of sentences
        roles_considered: list of semantic roles to consider
        save_to_disk: path to save the narrative model (default is None, which means no saving to disk)
        roles_with_embeddings: list of lists of semantic roles to embed and cluster
        (i.e. each list represents semantic roles that should be clustered together)
        embeddings_type: whether the user wants to use USE / Keyed Vectors or a custom pre-trained Word2Vec
        (e.g. "USE" / "gensim_keyed_vectors" / "gesim_full_model")
        embeddings_path: path for the trained embeddings model
        n_clusters: number of clusters for the clustering model
        preprocessing_options: see preprocess() function
        roles_with_entities: list of semantic roles with named entities
        ent_labels: list of entity labels to be considered (see SPaCy documentation)
        top_n_entities: number of named entities to keep (default is all and is specified with top_n = 0)
        progress_bar: print a progress bar (default is False)
        verbose: see Scikit-learn documentation for details


    Returns:
        A dictionary with the details of the pipeline to extract narratives from text

    """

    # Sanity checks
    if is_subsequence(roles_considered, ['ARGO', 'B-V', 'B-ARGM-NEG', 'B-ARGM-MOD', 'ARG1', 'ARG2']) == False:
        raise ValueError("Some roles_considered do not exist.")

    if is_subsequence(['ARGO', 'B-V', 'B-ARGM-NEG', 'ARG1'], roles_considered) == False:
        raise ValueError("Minimum roles to consider: ['ARGO', 'B-V', 'B-ARGM-NEG', 'ARG1']")

    if roles_with_entities is not None:
        if is_subsequence(roles_with_entities, roles_considered) == False:
            raise ValueError("roles_with_entities should be in roles_considered.")

    if roles_with_embeddings is not None:
        for roles in roles_with_embeddings:
            if is_subsequence(roles, roles_considered) == False:
                raise ValueError("each list in roles_with_embeddings should be a subset of roles_considered.")
            if ['B-ARGM-NEG', 'B-ARGM-MOD'] in roles:
                raise ValueError("Negations and modals cannot be embedded and clustered.")

    if roles_with_embeddings is not None:
        if embeddings_type not in ['gensim_keyed_vectors', 'gensim_full_model', 'USE']:
            raise TypeError("Only three types of embeddings accepted: gensim_keyed_vectors, gensim_full_model, USE")

    if is_subsequence(ent_labels, ['PERSON', 'NORP', 'ORG', 'GPE', 'EVENT']) == False:
        raise ValueError("Some ent_labels do not exist.")

    if lemmatize is True and stem is True:
        raise ValueError("lemmatize and stemming cannot be both True")

    # Narrative model dictionary
    narrative_model = {}

    narrative_model['roles_considered'] = roles_considered
    narrative_model['roles_with_entities'] = roles_with_entities
    narrative_model['roles_with_embeddings'] = roles_with_embeddings
    narrative_model['dimension_reduce_verbs'] = dimension_reduce_verbs
    narrative_model['clean_text_options'] = {
        'max_length': max_length,
        'remove_punctuation': remove_punctuation,
        'remove_digits': remove_digits,
        'remove_chars': remove_chars,
        'stop_words': stop_words,
        'lowercase': lowercase,
        'strip': strip,
        'remove_whitespaces': remove_whitespaces,
        'lemmatize': lemmatize,
        'stem': stem,
        'tags_to_keep': tags_to_keep,
        'remove_n_letter_words': remove_n_letter_words
    }

    # Process SRL
    roles, sentence_index = extract_roles(srl_res, 
                                          UsedRoles = roles_considered, 
                                          progress_bar = progress_bar)

    postproc_roles = postprocess_roles(roles,
                                       max_length,
                                       remove_punctuation,
                                       remove_digits,
                                       remove_chars,
                                       stop_words,
                                       lowercase,
                                       strip,
                                       remove_whitespaces,
                                       lemmatize,
                                       stem,
                                       tags_to_keep,
                                       remove_n_letter_words,
                                       progress_bar = progress_bar)

    # Verb Counts
    if dimension_reduce_verbs:
        verb_counts = get_role_counts(postproc_roles, 
                                      roles = ['B-V'], 
                                      progress_bar = progress_bar)
        
        narrative_model['verb_counts'] = verb_counts

    # Named Entities
    if roles_with_entities is not None:
        entities_sorted = mine_entities(sentences = sentences, 
                                        ent_labels = ent_labels, 
                                        progress_bar = progress_bar)
        
        entities = pick_top_entities(entities_sorted, top_n_entities = top_n_entities)
        
        entity_index, postproc_roles = map_entities(statements = postproc_roles,
                                                    entities = entities,
                                                    UsedRoles = roles_with_entities,
                                                    progress_bar = progress_bar)
        narrative_model['entities'] = entities
        
    # Embeddings and clustering
    if roles_with_embeddings is not None:
        sentences = preprocess(sentences,
                               remove_punctuation,
                               remove_digits,
                               remove_chars,
                               stop_words,
                               lowercase,
                               strip,
                               remove_whitespaces,
                               lemmatize,
                               stem,
                               tags_to_keep,
                               remove_n_letter_words)

        if embeddings_type == 'gensim_keyed_vectors':
            model = SIF_keyed_vectors(path = embeddings_path, sentences = sentences)
        if embeddings_type == 'gensim_full_model':
            model = SIF_word2vec(path = embeddings_path, sentences = sentences)
        if embeddings_type == 'USE':
            model = USE(path = embeddings_path)

        narrative_model['embeddings_model'] = model

        narrative_model['cluster_labels_most_similar'] = []
        narrative_model['cluster_model'] = []
        narrative_model['cluster_labels_most_freq'] = []

        for i, roles in enumerate(roles_with_embeddings):

            if n_clusters[i] == 0:
                test = list(get_role_counts(postproc_roles, roles = roles))
                n_clusters[i] = round(len(test)/100) # what should we do when there are not enough roles to cluster?

            kmeans = train_cluster_model(postproc_roles,
                                         model,
                                         n_clusters = n_clusters[i],
                                         UsedRoles=roles,
                                         verbose = verbose)

            clustering_res = get_clusters(postproc_roles,
                                          model,
                                          kmeans,
                                          UsedRoles=roles)

            labels_most_freq = label_clusters_most_freq(clustering_res=clustering_res,
                                                        postproc_roles=postproc_roles)

            if isinstance(model, (USE)) == False:
                labels_most_similar = label_clusters_most_similar(kmeans, model)
                narrative_model['cluster_labels_most_similar'].append(labels_most_similar)

            narrative_model['cluster_model'].append(kmeans)
            narrative_model['cluster_labels_most_freq'].append(labels_most_freq)

    if save_to_disk is not None:
        with open(save_to_disk, 'wb') as f:
            pk.dump(narrative_model, f)

    return narrative_model


def get_narratives(
    srl_res: List[dict],
    doc_index: List[int],
    narrative_model: dict,
    save_to_disk: Optional[str] = None,
    filter_complete_narratives: Optional[bool] = True,
    cluster_labeling: Optional[str] = 'most_frequent',
    progress_bar: Optional[bool] = False
):

    """

    A wrapper function to obtain the final mined narratives.

    Args:
        srl_res: sentences labeled with their semantic roles
        doc_index: list of indices to keep track of original documents
        narrative_model: dict with the specifics of the narrative model
        save_to_disk: path to save the narrative model (default is None, which means no saving to disk)
        filter_complete_narratives: keep only narratives with at least an agent, a verb and a patient
        (default is True)
        cluster_labeling: either 'most_frequent' or 'most_similar'
        progress_bar: print a progress bar (default is False)

    Returns:
        A pandas dataframe with the resulting narratives

    """

    final_statements = []

    # Sanity checks
    if cluster_labeling not in ['most_similar', 'most_frequent']:
        raise ValueError("cluster_labeling is either most_similar or most_frequent.")


    if cluster_labeling == 'most_similar' and isinstance(narrative_model['embeddings_model'], USE):
        raise ValueError("most_similar option is not implemented for Universal Sentence Encoders.Consider switching to other ebedding types.")

    # Process SRL
    roles, sentence_index = extract_roles(srl_res, 
                                          UsedRoles = narrative_model['roles_considered'], 
                                          progress_bar = progress_bar)

    postproc_roles = postprocess_roles(roles,
                                       narrative_model['clean_text_options']['max_length'],
                                       narrative_model['clean_text_options']['remove_punctuation'],
                                       narrative_model['clean_text_options']['remove_digits'],
                                       narrative_model['clean_text_options']['remove_chars'],
                                       narrative_model['clean_text_options']['stop_words'],
                                       narrative_model['clean_text_options']['lowercase'],
                                       narrative_model['clean_text_options']['strip'],
                                       narrative_model['clean_text_options']['remove_whitespaces'],
                                       narrative_model['clean_text_options']['lemmatize'],
                                       narrative_model['clean_text_options']['stem'],
                                       narrative_model['clean_text_options']['tags_to_keep'],
                                       narrative_model['clean_text_options']['remove_n_letter_words'],
                                       progress_bar = progress_bar)

    for statement in postproc_roles:
        temp = {}
        for role, tokens in statement.items():
            name = role + '-RAW'
            if type(tokens)!=bool:
                temp[name] = ' '.join(tokens)
            else:
                temp[name] = tokens
        final_statements = final_statements + [temp]

    # Dimension reduction of verbs
    if narrative_model['dimension_reduce_verbs']:
        cleaned_verbs = clean_verbs(postproc_roles,
                                    narrative_model['verb_counts'])

        for i,statement in enumerate(cleaned_verbs):
            for role, value in statement.items():
                final_statements[i][role] = value

    # Named Entities
    if narrative_model['roles_with_entities'] is not None:
        entity_index, postproc_roles = map_entities(statements = postproc_roles,
                                                    entities = narrative_model['entities'],
                                                    UsedRoles = narrative_model['roles_with_entities'],
                                                    progress_bar = progress_bar)

        for role in narrative_model['roles_with_entities']:
            for token, indices in entity_index[role].items():
                for index in indices:
                    final_statements[index][role] = token

    # Embeddings
    if narrative_model['roles_with_embeddings'] is not None:

        for l, roles in enumerate(narrative_model['roles_with_embeddings']):
            clustering_res = get_clusters(postproc_roles,
                                          narrative_model['embeddings_model'],
                                          narrative_model['cluster_model'][l],
                                          UsedRoles=roles,
                                          progress_bar = progress_bar)

            if cluster_labeling == 'most_frequent':
                for i,statement in enumerate(clustering_res):
                    for role, cluster in statement.items():
                        final_statements[i][role] = narrative_model['cluster_labels_most_freq'][l][cluster]

            if cluster_labeling == 'most_similar':
                for i,statement in enumerate(clustering_res):
                    for role, cluster in statement.items():
                        final_statements[i][role] = narrative_model['cluster_labels_most_similar'][l][cluster]

    # Original sentence and document
    for i,index in enumerate(sentence_index):
        final_statements[i]['sentence'] = index
        final_statements[i]['doc'] = doc_index[index]

    final_statements = pd.DataFrame(final_statements)
    final_statements['statement'] = final_statements.index

    final_statements = build_narratives(final_statements, 
                                        narrative_model, 
                                        filter_complete_narratives)

    if save_to_disk is not None:
        final_statements.to_csv(save_to_disk, index = False)

    return final_statements


# Analysis
#..................................................................................................................
#..................................................................................................................


def inspect_label(
    final_statements,
    label: str,
    role: str
):

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

    res = final_statements.loc[final_statements[role] == label, str(role + '-RAW')].value_counts()

    return res


def inspect_narrative(
    final_statements,
    narrative: str
):

    """

    A function to inspect the raw statements represented by a narrative
    (i.e. to check the 'quality' of the final narratives).

    Args:
        final_statements: dataframe with the output of the pipeline
        narrative: cleaned narrative to inspect

    Returns:
        A pandas series sorted by frequency of raw narratives contained in this label

    """

    res = final_statements.loc[final_statements['narrative-CLEANED'] == narrative, 'narrative-RAW'].value_counts()

    return res
