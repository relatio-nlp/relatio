from copy import deepcopy


def is_negation(verb, negs=["pas", "ne", "n'"], neg_deps=["advmod"]):
    """
    Identify if the verb is negated in the sentence.

    Args:
        verb: a verb
        negs: list of negation words
        neg_deps: list of dependency labels for negation words

    Returns:
        True if the verb is negated, False otherwise
    """
    flag_negation = False
    l1 = [right for right in verb.rights if right.dep_ in neg_deps]
    l2 = [left for left in verb.lefts if left.dep_ in neg_deps]
    adv_mods = l1 + l2
    adv_mods = get_text(adv_mods)
    for neg in negs:
        if neg in adv_mods:
            flag_negation = True
    return flag_negation


def filter_pos(sent, pos):
    """
    A function that retrieves all tokens with specific part of speech tags.

    Args:
        sent: a spacy sentence
        pos: list of part of speech tags

    Returns:
        a list of tokens with specific part of speech tags
    """
    l = [tok for tok in sent if tok.pos_ in pos]
    return l


def get_deps(verb, deps=None):
    """
    A function that retrieves all dependencies of a verb.

    Args:
        verb: a verb
        deps: list of dependency labels

    Returns:
        a list of tokens that are dependencies of the verb
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

    Args:
        tokens: list of spacy tokens

    Returns:
        list of text

    """
    return [tok.text for tok in tokens]


def extract_svos_fr(sent, expand_nouns: bool = True, only_triplets: bool = True):
    """
    Get SVOs (Subject-Verb-Object triplets) from a spacy sentence (for french).

    Args:
        sent: a spacy sentence
        expand_nouns: get phrase nouns
        only_triplets: only return complete triplets SVO (where the three elements are present)

    Returns:
        a list of SVOs
    """
    svos = []

    all_verbs = filter_pos(sent, pos=["VERB"])

    for i, verb in enumerate(all_verbs):
        # negation
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
            temp = subjs.copy()
            for subj in temp:
                if len(filter_pos(subj.subtree, pos=["VERB"])) == 0:
                    for t in subj.subtree:
                        if t.dep_ == "conj":
                            subjs.append(t)

            if expand_nouns:
                for k, subj in enumerate(subjs):
                    if subj._.noun_chunk:
                        subjs[k] = subj._.noun_chunk.text
                    else:
                        subjs[k] = subj.text
            else:
                subjs = [subj.text for subj in subjs]
        elif not only_triplets:
            subjs = [""]

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
            temp = objs.copy()
            for obj in temp:
                if len(filter_pos(obj.subtree, pos=["VERB"])) == 0:
                    for t in obj.subtree:
                        if t.dep_ == "conj":
                            objs.append(t)

            if expand_nouns:
                for k, obj in enumerate(objs):
                    if obj._.noun_chunk:
                        objs[k] = obj._.noun_chunk.text
                    else:
                        objs[k] = obj.text
            else:
                objs = [obj.text for obj in objs]
        elif not only_triplets:
            objs = [""]

        # packaging
        for subj in subjs:
            for obj in objs:
                svo = (subj, negation, verb.text, obj)
                svos.append(svo)

    return svos


def extract_svos_en(sent, expand_nouns: bool = True, only_triplets: bool = True):
    """
    Get SVOs (Subject-Verb-Object triplets) from a spacy sentence (for english).

    Args:
        sent: a spacy sentence
        expand_nouns: get phrase nouns
        only_triplets: only return complete triplets SVO (where the three elements are present)

    Returns:
        a list of SVOs
    """
    svos = []

    all_verbs = filter_pos(sent, pos=["VERB"])

    for i, verb in enumerate(all_verbs):
        # negation
        negation = is_negation(verb)

        # subjects
        subjs = []
        subjs.extend(get_deps(verb, deps=["nsubj"]))  # active forms
        agents = get_deps(verb, deps=["agent"])  # passive forms
        for a in agents:
            subjs.extend(get_deps(a, deps=["pobj"]))

        for k, subj in enumerate(subjs):
            if subj.text in ["who", "that"]:
                for tok in sent:
                    for t in tok.rights:
                        if t == verb:
                            subjs[k] = tok
                    for t in tok.lefts:
                        if t == verb:
                            subjs[k] = tok

        if len(subjs) != 0:
            temp = subjs.copy()
            for subj in temp:
                if len(filter_pos(subj.subtree, pos=["VERB"])) == 0:
                    for t in subj.subtree:
                        if t.dep_ == "conj":
                            subjs.append(t)

            if expand_nouns:
                for k, subj in enumerate(subjs):
                    if subj._.noun_chunk:
                        subjs[k] = subj._.noun_chunk.text
                    else:
                        subjs[k] = subj.text
            else:
                subjs = [subj.text for subj in subjs]
        elif not only_triplets:
            subjs = [""]

        # objects
        objs = []
        objs.extend(get_deps(verb, deps=["dobj"]))  # active forms
        objs.extend(get_deps(verb, deps=["nsubjpass"]))  # passive forms

        if len(objs) != 0:
            temp = objs.copy()
            for obj in temp:
                if len(filter_pos(obj.subtree, pos=["VERB"])) == 0:
                    for t in obj.subtree:
                        if t.dep_ == "conj":
                            objs.append(t)

            if expand_nouns:
                for k, obj in enumerate(objs):
                    if obj._.noun_chunk:
                        objs[k] = obj._.noun_chunk.text
                    else:
                        objs[k] = obj.text
            else:
                objs = [obj.text for obj in objs]
        elif not only_triplets:
            objs = [""]

        # packaging
        for subj in subjs:
            for obj in objs:
                svo = (subj, negation, verb.text, obj)
                svos.append(svo)

    return svos


def from_svos_to_srl_res(svos):
    """
    Mapping between SVO triples obtained by dependency parsing and AVP triples obtained by SRL.

    Args:
        svos: a list of SVOs

    Returns:
        a list of AVPs
    """
    avps = []
    for svo in svos:
        avp = {}
        avp["ARG0"] = svo[0]
        avp["B-ARGM-NEG"] = svo[1]
        avp["B-V"] = svo[2]
        avp["ARG1"] = svo[3]
        avp_copy = deepcopy(avp)
        for key, value in avp_copy.items():
            if value == "" or value is False:
                del avp[key]
        avps.append(avp)
    return avps
