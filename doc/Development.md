# Development

## Prepare the development environment
You need `python3.7`, `git` and [tox](https://tox.readthedocs.io/en/latest/).  
A recent version of `wheel` and `pip>21.0` are desired for the python versions that you want to use.

```bash
# clone the project
$ git clone git@github.com:elliottash/narrative-nlp.git

# go in the main directory
$ cd narrative-nlp

# create the development environment
$ tox -e dev

# activate the development environment
$ source .tox/dev/bin/activate

# install the hooks
(dev)$ pre-commit install

# OPTIONAL: make your IPython kernel in one env available to Jupyter in a different env
(dev)$ python -m ipykernel install --user --name py37_narrativeNLP --display-name "Python 3.7 (narrativeNLP)"
```

## Testing
You can easily test using `tox` on `python3.7`, `python3.8`, and `python3.9`.

```bash
$ tox -e py37 # python3.7 required
$ tox -e py38 # python3.8 required
$ tox -e py39 # python3.9 required
```

So far it works only with `python3.7` probably due to implications from `allennlp <1` requirement.


## Tools
- Auto-formatter:
    - [black](https://black.readthedocs.io/en/stable/) that might break the [79 characters maximum line length](https://www.python.org/dev/peps/pep-0008/#maximum-line-length) from PEP8 (see [here](https://github.com/psf/black#line-length))
- Testing via [pytest](https://docs.pytest.org/en/latest/) with `--doctest-modules` enabled (see [doctest](http://doc.pytest.org/en/latest/doctest.html))
- security issues: [bandit](https://github.com/PyCQA/bandit)
- [mypy](http://mypy-lang.org/)
- [pycodestyle](https://github.com/PyCQA/pycodestyle), [pydocstyle](https://github.com/PyCQA/pydocstyle), [Pylint](https://github.com/PyCQA/pylint), [flake8](https://gitlab.com/pycqa/flake8)

### Guides
- [PEP8](https://www.python.org/dev/peps/pep-0008/)
- [Google Coding Style](http://google.github.io/styleguide/pyguide.html)
- Docstring using Google Style: see [Sphinx 1.3 Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and [pyguide](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
