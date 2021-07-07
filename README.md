# narrativeNLP
A Python package to extract narrative statements from text.

## Installation

It is highly recommended to use a virtual environment (or conda environment) for the installation.

```bash
# clone the project
$ git clone git@github.com:elliottash/narrative-nlp.git

# go in the main directory
$ cd narrative-nlp

# upgrade pip, wheel and setuptools
python -m pip install -U pip wheel setuptools

# install the project
python -m pip install -e .

# download SpaCy and NLTK additional resources
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt wordnet stopwords averaged_perceptron_tagger
```

In case you want to use Jupyter make sure that you have it installed in the current environment.

If you are interested in contributing to the project please read the [Development Guide](./doc/Development.md).