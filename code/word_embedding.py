from typing import List, Union

from gensim.models import KeyedVectors, Word2Vec


def run_word2vec(
    sentences: List[str],
    model: Word2Vec,
    pretrained_path: str,
    save_path: Union[None, str] = None,
) -> Union[None, Word2Vec]:
    w2v_sentences = [sent.split() for sent in sentences]
    model.build_vocab(w2v_sentences)
    total_examples = model.corpus_count
    pretrained_model = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
    model.build_vocab([list(pretrained_model.vocab.keys())], update=True)
    model.intersect_word2vec_format(pretrained_path, binary=False, lockf=1.0)
    model.train(w2v_sentences, total_examples=total_examples, epochs=model.iter)

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
