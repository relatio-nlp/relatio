# relatio
A Python package to extract narrative statements from text.

## Installation

Runs on Linux and macOS (x86 platform) and it requires Python 3.7 (or 3.8) and pip.  
It is highly recommended to use a virtual environment (or conda environment) for the installation.

```bash
# upgrade pip, wheel and setuptools
python -m pip install -U pip wheel setuptools

# install the package
python -m pip install git+https://github.com/relatio-nlp/relatio

# download SpaCy and NLTK additional resources
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt wordnet stopwords averaged_perceptron_tagger
```

In case you want to use Jupyter make sure that you have it installed in the current environment.

If you are interested in contributing to the project please read the [Development Guide](./doc/Development.md).
