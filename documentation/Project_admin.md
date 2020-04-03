
# Project
**Narrative extraction system using semantic role labeling and word embeddings**

The project will target English corpora and focus on established word embedding ([gensim - word2vec](https://radimrehurek.com/gensim/models/word2vec.html)) and [semantic role labelling - allennlp](https://allennlp.org/)

## Initial Tasks
- [ ] (Task 1) make the multiple scripts modular and run in a single task.
- [ ] (Task 2) better keep track of original documents and sentences (to combine them later with metadata + compare the mined narrative with the original sentence)
- [ ] (Task 3) we plan to add more roles later, so that should be an option.
- [ ] (Task 4) on the optimization side: CPU or GPU? can we speed up the SRL? mine narratives using sparse matrices instead of numpy arrays (to save some memory)

## Initial Budget
- 20 working days

## Strategy
1. **Phase I: ~10 working days, Deadine 2020-02-29**
Working refactored code to be modular + minimal (unit) testing:
    - [ ] (Task 1) 
    - [ ] (Task 2)
    - [ ] (Task 3)

2. **Phase II** 
    - [ ] Refactoring (to be established after Phase I)
    - [ ] (Task 4)

last update (2020-01-14)

### Implementation:

Object oriented programming similar to [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

## Terminology:
- Corpus -> Documents -> Sentences

## Decisions made:

### sentence tokenizer
- use nltk and add potentially add new rules. See [here](../Notebooks/Benchmarking.ipynb).

|                        | spacy   | nltk   | in-house |
|------------------------|---------|--------|----------|
| small doc (500 chars)  | 7_500 µs | 180 µs | 80 µs  |
| large doc (900K chars) | 11_000 ms  | 260 ms | 90 ms |

-
## TODO
- [ ] mark how to find any warning in the original document 

## Nice to have:

- [ ] utils.preprocess arguments also as Callable when it makes sense, e.g. lemmatize
- [ ] address opposites (e.g. increase and decrease) that could end up in the same cluster via verbnet synsets top dimension-reduce the verbs (see https://www.nltk.org/howto/wordnet.html)

## Open Questions:
## Meetings / Reporting
### 2020-03-30
- the code is tested at scale and the bugs are fixed (excluding SRL and training of gensim.word2vec)
- new features: 
    - compute distance
    - discard vectors based on a distance threshold
    - cluster labeling based on (i) most frequent ngram in the cluster or (ii) the closest vector to the centroid
- TODO:
    - [ ] parallelization of the code (IPython Parallel [`ipyparallel` package](https://ipyparallel.readthedocs.io/en/stable/))
    - [ ] SRL: GPU and parallelization (IPython Parallel - no workflow management system)
    - [x] cluster labeling: closest word in embedding for gensim.word2vec
    - [ ] cluster labeling: add a new parameter min_freq for the closest vector to the centroid
### 2020-02-24
See [Slides](./Slides/2020-02-24%20Updates.pptx)
### 2020-02-06
See [Slides](./Slides/2020-02-06%20Big_Picture.pptx)
