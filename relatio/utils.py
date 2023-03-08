# MIT License

# Copyright (c) 2020-2021 ETH Zurich, Andrei V. Plamada
# Copyright (c) 2020-2021 ETH Zurich, Elliott Ash
# Copyright (c) 2020-2021 University of St.Gallen, Philine Widmer
# Copyright (c) 2020-2021 Ecole Polytechnique, Germain Gauthier

# Utils
# ..................................................................................................................
# ..................................................................................................................

import json
import pickle as pk
import time
from collections import Counter
from typing import Dict, List, Optional

from torch.cuda import is_available
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer


def replace_sentences(
    sentences: List[str],
    max_sentence_length: Optional[int] = None,
    max_number_words: Optional[int] = None,
) -> List[str]:

    """

    Replace long sentences in list of sentences by empty strings.

    Args:
        sentences: list of sentences
        max_sentence_length: Keep only sentences with a a number of character lower or equal to max_sentence_length. For max_number_words = max_sentence_length = -1 all sentences are kept.
        max_number_words: Keep only sentences with a a number of words lower or equal to max_number_words. For max_number_words = max_sentence_length = -1 all sentences are kept.

    Returns:
        Replaced list of sentences.

    Examples:
        >>> replace_sentences(['This is a house'])
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_sentence_length=15)
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_sentence_length=14)
        ['']
        >>> replace_sentences(['This is a house'], max_number_words=4)
        ['This is a house']
        >>> replace_sentences(['This is a house'], max_number_words=3)
        ['']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=18)
        ['This is a house', 'It is a nice house']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=4, max_sentence_length=18)
        ['This is a house', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=17)
        ['This is a house', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=0, max_sentence_length=18)
        ['', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=5, max_sentence_length=0)
        ['', '']
        >>> replace_sentences(['This is a house', 'It is a nice house'])
        ['This is a house', 'It is a nice house']
        >>> replace_sentences(['This is a house', 'It is a nice house'], max_number_words=4)
        ['This is a house', '']

    """

    if max_sentence_length is not None:
        sentences = [
            "" if (len(sent) > max_sentence_length) else sent for sent in sentences
        ]

    if max_number_words is not None:
        sentences = [
            "" if (len(sent.split()) > max_number_words) else sent for sent in sentences
        ]

    return sentences


def group_sentences_in_batches(
    sentences: List[str],
    max_batch_char_length: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> List[List[str]]:

    """

    Group sentences in batches of given total character length or size (number of sentences).

    In case a sentence is longer than max_batch_char_length it is replaced with an empty string.

    Args:
        sentences: List of sentences
        max_batch_char_length: maximum char length for a batch
        batch_size: number of sentences

    Returns:
        List of batches (list) of sentences.

    Examples:
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=15)
        [['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=14)
        [['', '']]
        >>> group_sentences_in_batches(['This is a house','This is a house', 'This is not a house'], max_batch_char_length=15)
        [['This is a house'], ['This is a house', '']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=29)
        [['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'], max_batch_char_length=30)
        [['This is a house', 'This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house'])
        [['This is a house', 'This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], max_batch_char_length=29)
        [['This is a house'], ['This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], max_batch_char_length=30)
        [['This is a house', 'This is a house'], ['This is a house']]
        >>> group_sentences_in_batches(['This is a house','This is a house','This is a house'], batch_size=2)
        [['This is a house', 'This is a house'], ['This is a house']]

    """

    batches: List[List[str]] = []

    if max_batch_char_length is not None and batch_size is not None:
        raise ValueError("max_batch_char_length and batch_size are mutually exclusive.")
    elif max_batch_char_length is not None:

        # longer sentences are replaced with an empty string
        sentences = replace_sentences(
            sentences, max_sentence_length=max_batch_char_length
        )
        batch_char_length = 0
        batch: List[str] = []

        for el in sentences:
            length = len(el)
            batch_char_length += length
            if batch_char_length > max_batch_char_length:
                batches.append(batch)
                batch = [el]
                batch_char_length = length
            else:
                batch.append(el)

        if batch:
            batches.append(batch)

    elif batch_size is not None:
        batches = [
            sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)
        ]
    else:
        batches = [sentences]

    return batches


def is_subsequence(v1: list, v2: list) -> bool:

    """

    Check whether v1 is a subset of v2.

    Args:
        v1: lists of elements
        v2: list of elements

    Returns:
        a boolean

    Example:
        >>> is_subsequence(['united', 'states', 'of', 'europe'],['the', 'united', 'states', 'of', 'america'])
        False
        >>> is_subsequence(['united', 'states', 'of'],['the', 'united', 'states', 'of', 'america'])
        True

    """
    # TODO: Check whether the order of elements matter, e.g. is_subsequence(["A","B"],["B","A"])
    return set(v1).issubset(set(v2))


def count_values(
    dicts: List[Dict], keys: Optional[list] = None, progress_bar: bool = False
) -> Counter:

    """

    Get a counter with the values of a list of dictionaries, with the conssidered keys given as argument.

    Args:
        dicts: list of dictionaries
        keys: keys to consider
        progress_bar: print a progress bar (default is False)

    Returns:
        Counter

    Example:
        >>> count_values([{'B-V': 'increase', 'B-ARGM-NEG': True},{'B-V': 'decrease'},{'B-V': 'decrease'}],keys = ['B-V'])
        Counter({'decrease': 2, 'increase': 1})
        >>> count_values([{'B-V': 'increase', 'B-ARGM-NEG': True},{'B-V': 'decrease'},{'B-V': 'decrease'}])
        Counter()

    """

    counts: Dict[str, int] = {}

    if progress_bar:
        print("Computing role frequencies...")
        time.sleep(1)
        dicts = dicts

    if keys is None:
        return Counter()

    for el in dicts:
        for key, value in el.items():
            if key in keys:
                if value in counts:
                    counts[value] += 1
                else:
                    counts[value] = 1

    return Counter(counts)


def count_words(sentences: List[str]) -> Counter:

    """

    A function that computes word frequencies in a list of sentences.

    Args:
        sentences: list of sentences

    Returns:
        Counter {"word": frequency}

    Example:
    >>> count_words(["this is a house"])
    Counter({'this': 1, 'is': 1, 'a': 1, 'house': 1})
    >>> count_words(["this is a house", "this is a house"])
    Counter({'this': 2, 'is': 2, 'a': 2, 'house': 2})
    >>> count_words([])
    Counter()
    """

    words: List[str] = []

    for sentence in sentences:
        words.extend(sentence.split())

    words_counter = Counter(words)

    return words_counter


def make_list_from_key(key, list_of_dicts):

    """

    Extract the content of a specific key in a list of dictionaries.
    Returns a list and the corresponding indices.

    """

    list_from_key = []
    indices = []

    for i, statement in enumerate(list_of_dicts):
        content = statement.get(key)
        if content is not None:
            list_from_key.append(content)
            indices.append(i)

    return indices, list_from_key

class FixGrammarDataset(Dataset):
    """ torch.utils.data.Dataset subclass which holds the narratives to be grammar-fixed. """
    
    def __init__(self, 
                 narratives: List[str], 
                 tokenizer):
        """ 
        
        Initializer for the Dataset object.
        
        Tokenizes all narratives so that they can be fed to the language model.

        Args:
            narratives (List[str]): a list of narratives to be grammar-fixed.
            tokenizer: the appropriate tokenizer from the Huggingface library.
        
        Returns:
            the dataset object.
        """

        self.max_length = 100
        self.padding = 'max_length'
        self.truncation = True
        self.return_tensors = 'pt'

        # Encode inputs
        self.input_ids = []
        self.attn_masks = []
        for narrative in narratives:
            inp_encoded = tokenizer(narrative, 
                                    max_length=self.max_length, 
                                    padding=self.padding, 
                                    truncation=self.truncation, 
                                    return_tensors=self.return_tensors)
            self.input_ids.append(inp_encoded['input_ids'][0])
            self.attn_masks.append(inp_encoded['attention_mask'][0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

def fix_grammars(narratives: List[str]):
    """ 

    TODO: should I delete the ML stuff when the fct is done (to make space in memory)?

    Fixes the grammars of the given list of narratives.
    
    Args:
        narratives: a list of narratives to be grammar-fixed. 

    Returns:
        outputs: a list with the same narratives as in the input, but their grammars are fixed.

    """
    narratives = [handle_negation(n) for n in narratives] # handle the '!'s in narratives
    device = 'cuda' if is_available() else 'cpu'
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('egek-7/relatio-fix-grammar').to(device)
    model.eval()
    dataset = FixGrammarDataset(narratives, tokenizer)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    model.eval()
    outputs = []
        
    for sample in tqdm(dataloader):
        input_ids, attn_mask = sample[0], sample[1]
        model_kwargs = {
            'inputs': input_ids.to(device),
            'attention_mask': attn_mask.to(device),
            'num_beams': 4,
            'max_length': 100,
            #'return_dict_in_generate': True # transformers.generation_utils.BeamSearchEncoderDecoderOutput
        }
        
        output_ids = model.generate(**model_kwargs)
        out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs += out

    return outputs

def handle_selection(narrative: Dict[str, str]) -> Dict[str, str]:
    """ 
    
    Modifies the arguments of the given narrative so that all '|'s are removed.
     
    For each argument with multiple options, all options except the first one are removed, 
    regardless of whether | stands for the logical "and" or logical "or".
    
    """
    arguments = ['ARG0', 'ARG1', 'ARG2']
    for arg in arguments:
        value = narrative.get(arg)
        if value is not None and '|' in value:
            narrative[arg] = value[:value.index('|')]
    return narrative

def handle_negation(narrative: str) -> str:
    """ Modifies the given narrative so that all '!'s are replaced with 'not-'. """

    if '!' in narrative:
        split = narrative.split(' ')
        neg_idx = split.index('!')
        split[neg_idx+1] = "not-" + split[neg_idx+1]
        split.pop(neg_idx)
        narrative = ' '.join(split)
    return narrative

def get_element(narrative, role):
    return narrative[role] if role in narrative else ""

def prettify(narrative) -> str:

    ARG0 = get_element(narrative, "ARG0")
    V = get_element(narrative, "B-V")

    NEG = get_element(narrative, "B-ARGM-NEG")
    if NEG is True:
        NEG = "!"
    elif NEG is False:
        NEG = ""

    MOD = get_element(narrative, "B-ARGM-MOD")
    ARG1 = get_element(narrative, "ARG1")
    ARG2 = get_element(narrative, "ARG2")

    pretty_narrative = (ARG0, MOD, NEG, V, ARG1, ARG2)

    pretty_narrative = " ".join([t for t in pretty_narrative if t != ""])

    return pretty_narrative

def prettify_narratives(narratives: List[Dict], fix_grammar: bool = False):
    """ 
    
    'Prettifies' a given list of narratives.
    
    Converts each narrative from a dictionary to a (grammatically incorrect)
    human-readable phrase. The grammar can optionally be corrected. It takes
    roughly one minute to correct the grammars of around 250 narratives.

    Args:
        narratives (List[Dict]): list of narratives.
        fix_grammar (bool): if True, the prettified narratives will be 
            grammatically correct. False by default.

    Returns:
        pretty_narratives (Counter[str]): a counter over the prettified list of narratives.
    
    """

    prettifiable_narratives = []
    for narrative in narratives:
        if narrative.get('ARG0') is not None:
            if narrative.get('B-V') is not None:
                if narrative.get('ARG1') is not None:
                    prettifiable_narratives.append(narrative)
    
    if fix_grammar:
        prettifiable_narratives = [handle_selection(n) for n in prettifiable_narratives] # handle the '|'s in narratives
    
    pretty_narratives = []
    for narrative in prettifiable_narratives:    
        pretty_narratives.append(prettify(narrative))

    if fix_grammar:
        pretty_narratives = fix_grammars(pretty_narratives)
    
    pretty_narratives = Counter(pretty_narratives)
    return pretty_narratives

def save_entities(entity_counts, output_path: str):

    with open(output_path, "wb") as f:
        pk.dump(entity_counts, f)


def load_entities(input_path: str):

    with open(input_path, "rb") as f:
        entity_counts = pk.load(f)

    return entity_counts


def save_roles(roles, output_path):

    with open(output_path, "w") as f:
        json.dump(roles, f)


def load_roles(input_path):

    with open(input_path, "r") as f:
        roles = json.load(f)

    return roles


def is_negation(tok, negs=["pas", "ne", "n'"], neg_deps=["advmod"]):
    """
    Identify if the verb is negated in the sentence.
    """
    flag_negation = False
    l1 = [right for right in tok.rights if right.dep_ in neg_deps]
    l2 = [left for left in tok.lefts if left.dep_ in neg_deps]
    adv_mods = l1 + l2
    adv_mods = get_text(adv_mods)
    for neg in negs:
        if neg in adv_mods:
            flag_negation = True
    return flag_negation


def filter_pos(sent, pos):
    """
    Returns all tokens with specific part of speech tags.
    """
    l = [tok for tok in sent if tok.pos_ in pos]
    return l


def get_deps(verb, deps=None):
    """
    Returns all dependencies of a verb.
    """
    l = []
    if deps is not None:
        l.extend([tok for tok in verb.lefts if tok.dep_ in deps])
        l.extend([tok for tok in verb.rights if tok.dep_ in deps])
    else:
        l.extend([tok for tok in verb.lefts])
        l.extend([tok for tok in verb.rights])
    return l


def get_text(tokens):
    """
    Returns text from list of spacy tokens.
    """
    return [tok.text for tok in tokens]


def extract_svos_fr(sent):
    """
    Get SVOs from a spacy sentence (for french).
    """
    svos = []

    all_verbs = filter_pos(sent, pos=["VERB"])

    for i, verb in enumerate(all_verbs):

        negation = is_negation(verb)

        # subjects
        subjs = []
        subjs.extend(get_deps(verb, deps=["nsubj"]))  # active forms
        subjs.extend(get_deps(verb, deps=["obl:agent"]))  # passive forms

        for k, subj in enumerate(subjs):
            if subj.text in ["qui", "qu'"]:
                for tok in sent:
                    for t in tok.rights:
                        if t == verb:
                            subjs[k] = tok
                    for t in tok.lefts:
                        if t == verb:
                            subjs[k] = tok

        if len(subjs) != 0:
            subjs = [" ".join([t.text for t in subj.subtree]) for subj in subjs]
        elif i > 0 and len(svos) > 0:
            subjs = [svos[i - 1][0]]

        # objects
        objs = []
        objs.extend(get_deps(verb, deps=["obj"]))  # active forms
        objs.extend(get_deps(verb, deps=["nsubj:pass"]))  # passive forms

        for k, obj in enumerate(objs):
            if obj.text in ["que", "qu'"]:
                for tok in sent:
                    for t in tok.rights:
                        if t == verb:
                            objs[k] = tok
                    for t in tok.lefts:
                        if t == verb:
                            objs[k] = tok

        if len(objs) != 0:
            objs = [" ".join([t.text for t in obj.subtree]) for obj in objs]

        # packaging
        subjs = " ".join(subjs)
        objs = " ".join(objs)
        verb = verb.text
        svo = (subjs, negation, verb, objs)

        svos.append(svo)

    return svos


def from_svos_to_srl_res(svos):
    """
    Mapping between SVO triples obtained by dependency parsing and AVP triples obtained by SRL.
    """
    avps = []
    for svo in svos:
        avp = {}
        avp["ARG0"] = svo[0]
        avp["B-ARGM-NEG"] = svo[1]
        avp["B-V"] = svo[2]
        avp["ARG1"] = svo[3]
        avps.append(avp)
    return avps
