# narrative-nlp
code base for constructing narrative statements from text

## Installation
Clone the project and move to the branch of interest and go to the parent of your repo
```bash
git checkout existingbranch
cd ..
```

### Via Conda on macOS or Linux
Make sure that you have [conda](https://docs.conda.io/projects/conda/en/latest/) installed. Use command line and run

```bash
conda env create -f narrative-nlp/environment.yml
conda activate narrative-nlp-dev
python -m spacy download en_core_web_sm
python -c 'import nltk; nltk.download("punkt")'
```
### Via Containers - repo2docker (technology used in in https://mybinder.org/)
Make sure you have [repo2docker](https://repo2docker.readthedocs.io/en/latest/install.html) - Docker is needed. In case you want to mount a volume just use `--volume`. It might take up to 10 minutes to build the container
```bash
jupyter-repo2docker --volume ./narrative-nlp/Notebooks:Notebooks narrative-nlp/
```

## Usage
In your python script or Jupyter Notebook add to the path `narrative-nlp/code` and use any module as desired
```python
import sys

# update "narrative-nlp/code" with the appropriate path
sys.path.append("narrative-nlp/code")

from utils import preprocess
```
## Development

### Tools
- Auto-formatter: [black](https://black.readthedocs.io/en/stable/) that might break the [79 characters maximum line length](https://www.python.org/dev/peps/pep-0008/#maximum-line-length) from PEP8 (see [here](https://github.com/psf/black#line-length))
- Testing via [pytest](https://docs.pytest.org/en/latest/) with `--doctest-modules` enabled (see [doctest](http://doc.pytest.org/en/latest/doctest.html))
- security issues: [bandit](https://github.com/PyCQA/bandit)
- [mypy](http://mypy-lang.org/)
- [pycodestyle](https://github.com/PyCQA/pycodestyle), [pydocstyle](https://github.com/PyCQA/pydocstyle), [Pylint](https://github.com/PyCQA/pylint), [flake8](https://gitlab.com/pycqa/flake8)

### Guides
- [PEP8](https://www.python.org/dev/peps/pep-0008/)
- [Google Coding Style](http://google.github.io/styleguide/pyguide.html)
- Docstring using Google Style: see [Sphinx 1.3 Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and [pyguide](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)