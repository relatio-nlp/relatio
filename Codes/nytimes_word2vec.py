import gensim
from gensim.models import KeyedVectors, Word2Vec

from gensim.models.word2vec import LineSentence
sentences = LineSentence('/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/unigram_sentences_nytimes.txt', max_sentence_length=30000)

model = Word2Vec(size = 300, window = 10, min_count = 1, workers = 4)
model.build_vocab(sentences)
total_examples = model.corpus_count

pretrained_model = KeyedVectors.load_word2vec_format("/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/word_embeddings/glove_2_word2vec.6B.300d.txt", binary = False)

model.build_vocab([list(pretrained_model.vocab.keys())], update = True)
model.intersect_word2vec_format("/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/word_embeddings/glove_2_word2vec.6B.300d.txt", binary = False, lockf = 1.0)
model.train(sentences, total_examples = total_examples, epochs = model.iter)
model.save('/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/NYT/nytimes_word2vec.model')

