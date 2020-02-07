import re
import string
from typing import List, Union

from gensim.models import KeyedVectors, Word2Vec


def preprocess(
    sentences: List[str],
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    remove_chars: str = "",
    lowercase: bool = True,
    strip: bool = True,
    remove_whitespaces: bool = True,
) -> List[str]:
    """
    Preprocess a list of sentences for word embedding.

    Args:
        sentence: list of sentences
        remove_punctuation: whether to remove string.punctuation
        remove_digits: whether to remove string.digits
        remove_chars: remove the given characters
        lowercase: whether to lower the case
        strip: whether to strip
        remove_whitespaces: whether to remove superfluous whitespaceing by " ".join(str.split(())
    Returns:
        Processed list of sentences

    Examples:
        >>> preprocess([' Return the factorial of n, an  exact integer >= 0.'])
        ['return the factorial of n an exact integer']
        >>> preprocess(['A1b c\\n\\nde \\t fg\\rkl\\r\\n m+n'])
        ['ab c de fg kl mn']
    """
    # remove chars
    if remove_punctuation is True:
        remove_chars += string.punctuation
    if remove_digits is True:
        remove_chars += string.digits
    if remove_chars:
        sentences = [re.sub(f"[{remove_chars}]", "", sent) for sent in sentences]

    # lowercase, strip and remove superfluous white spaces
    if lowercase:
        sentences = [sent.lower() for sent in sentences]
    if strip:
        sentences = [sent.strip() for sent in sentences]
    if remove_whitespaces:
        sentences = [" ".join(sent.split()) for sent in sentences]
    return sentences


def run_word2vec(
    sentences: List[str],
    model: Word2Vec,
    pretrained_path: str,
    save_path: Union[None, str] = None,
) -> Union[None, Word2Vec]:
    sentences = [sent.split() for sent in sentences]
    model.build_vocab(sentences)
    total_examples = model.corpus_count
    pretrained_model = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
    model.build_vocab([list(pretrained_model.vocab.keys())], update=True)
    model.intersect_word2vec_format(pretrained_path, binary=False, lockf=1.0)
    model.train(sentences, total_examples=total_examples, epochs=model.iter)

    if save_path is not None:
        model.save(save_path)
        return None
    else:
        return model


def phrase_embeddings(model: Union[str, Word2Vec]):
    # TODO
    if isinstance(model, str):
        model = Word2Vec.load(model)
    elif isinstance(model, str):
        pass
    else:
        raise TypeError("model is either the a string or an Word2Vec object")

    # Create a word count dictionary based on the trained model
    word_count_dict = {}
    for word, vocab_obj in model.wv.vocab.items():
        word_count_dict[word] = vocab_obj.count

    # Create a dictionary that maps from a word to its frequency and then use the frequency of
    # a word to compute its sif-weight (saved in sif_dict)
    sif_dict = {}
    for word, count in word_count_dict.items():
        sif_dict[word] = 0.001 / (0.001 + count)
    # pk.dump(sif_dict, open('sif_dict.p', 'wb'))
