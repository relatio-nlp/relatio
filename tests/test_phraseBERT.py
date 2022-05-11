import time


import pytest
import numpy


from relatio import Embeddings


def test_phraseBERT():
    path = "whaleloops/phrase-bert"
    model = Embeddings("phrase-BERT", path)
    sentence_emb = model.get_vector("Hello world")

    assert type(sentence_emb) == numpy.ndarray
    assert len(sentence_emb) >= 1


def sentences_encode(duration=0.1):
    time.sleep(duration)
    sentences = (
        "Republicans and Democrats have both created our economic problems."
        "The Unsolicited Mail In Ballot Scam is a major threat to our Democracy, "
        "&amp; the Democrats know it. Almost all recent elections using this system, "
        "even though much smaller &amp; with far fewer Ballots to count, have ended up being a disaster. "
        "Large numbers of missing Ballots &amp; Fraud!"
    )
    path = "whaleloops/phrase-bert"
    model = Embeddings("phrase-BERT", path)
    model.get_vectors(sentences)

    return True


def each_sentence_encode(duration=0.1):
    time.sleep(duration)
    sentences = (
        "Republicans and Democrats have both created our economic problems."
        "The Unsolicited Mail In Ballot Scam is a major threat to our Democracy, "
        "&amp; the Democrats know it. Almost all recent elections using this system, "
        "even though much smaller &amp; with far fewer Ballots to count, have ended up being a disaster. "
        "Large numbers of missing Ballots &amp; Fraud!"
    )
    path = "whaleloops/phrase-bert"
    model = Embeddings("phrase-BERT", path)
    senteces_list = sentences.split(".")
    for sentence in senteces_list:
        model.get_vector(sentence)

    return True


def test_phraseBERT_speed(benchmark):
    ret = benchmark(sentences_encode)
    assert ret


def test_phraseBERT_speed_2(benchmark):
    ret = benchmark(each_sentence_encode)
    assert ret


if __name__ == "__main__":
    pytest.main(["-v", __file__])
