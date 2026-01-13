# Development

## Prepare the development environment
You need `python3.9` or later (3.9-3.12), `git` and [tox](https://tox.readthedocs.io/en/latest/).
A recent version of `wheel` and `pip>=21.0` are desired for the python versions that you want to use.

```bash
# clone the project
$ git clone https://github.com/relatio-nlp/relatio

# go in the main directory
$ cd relatio

# create the development environment
$ tox -e dev

# activate the development environment
$ source .tox/dev/bin/activate

# install the hooks
(dev)$ pre-commit install

# OPTIONAL: make your IPython kernel available to Jupyter in a different env
(dev)$ python -m ipykernel install --user --name py311_relatio --display-name "Python 3.11 (relatio)"
```

## Testing
You can easily test using `tox` on `python3.9`, `python3.10`, `python3.11`, and `python3.12`.

```bash
$ tox -e py39   # python3.9 required
$ tox -e py310  # python3.10 required
$ tox -e py311  # python3.11 required
$ tox -e py312  # python3.12 required
```


## Tools
- Auto-formatter:
    - [black](https://black.readthedocs.io/en/stable/) that might break the [79 characters maximum line length](https://www.python.org/dev/peps/pep-0008/#maximum-line-length) from PEP8 (see [here](https://github.com/psf/black#line-length))
    - [isort](https://pycqa.github.io/isort/index.html) using [black compatibility](https://pycqa.github.io/isort/docs/configuration/black_compatibility.html)
- Testing via [pytest](https://docs.pytest.org/en/latest/) with `--doctest-modules` enabled (see [doctest](http://doc.pytest.org/en/latest/doctest.html))
- security issues: [bandit](https://github.com/PyCQA/bandit)
- [mypy](http://mypy-lang.org/)
- [pycodestyle](https://github.com/PyCQA/pycodestyle), [pydocstyle](https://github.com/PyCQA/pydocstyle), [Pylint](https://github.com/PyCQA/pylint), [flake8](https://gitlab.com/pycqa/flake8)

### Guides
- [PEP8](https://www.python.org/dev/peps/pep-0008/)
- [Google Coding Style](http://google.github.io/styleguide/pyguide.html)
- Docstring using Google Style: see [Sphinx 1.3 Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and [pyguide](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
