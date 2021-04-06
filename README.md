# narrative-nlp
code base for constructing narrative statements from text

## Installation

It is highly recommended to use a virtual environment (or conda environment) for the installation.

```bash
# clone the project
$ git clone git@github.com:elliottash/narrative-nlp.git

# go in the main directory
$ cd narrative-nlp

# upgrade pip
$ pip install --upgrade pip

# update / install wheels and setuptools
$ pip install -U wheel setuptools

# install it
$ python -m pip install -e .
```

In case you want to use Jupyter make sure that you have it installed in the current environment.

## Development

### Prepare the development environment
You need `python3.7`, `git` and [tox](https://tox.readthedocs.io/en/latest/).  
A recent version of `wheel` and `pip>0.21` are desired for the python versions that you want to use.

```bash
# clone the project
$ git clone git@github.com:elliottash/narrative-nlp.git

# go in the main directory
$ cd narrative-nlp

# create the development environment
$ tox -e dev
# in case you do not have tox use directly .binder/requirements_dev.txt

# activate the development environment
$ source .tox/dev/bin/activate

# install the hooks
(dev)$ pre-commit install

# OPTIONAL: make your IPython kernel in one env available to Jupyter in a different env
(dev)$ python -m ipykernel install --user --name py37_narrativeNLP --display-name "Python 3.7 (narrativeNLP)"
```

### Testing

You can easily test using `tox` on `python3.7`, `python3.8`, and `python3.9`. 

```bash
$ tox -e py37 # python3.7 required
$ tox -e py38 # python3.8 required
$ tox -e py39 # python3.9 required
```

So far it works only with `python3.7` probably due to implications from `allennlp <1` requirement.