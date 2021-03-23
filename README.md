# narrativeNLP
A Python package to extract narrative statements from text.

## Installation

```bash
# clone the project
git clone git@github.com:elliottash/narrative-nlp.git

# go in the main directory
cd narrative-nlp

# install it
python -m pip install -e .

# download SpaCy and NLTK additional resources
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt wordnet stopwords averaged_perceptron_tagger
```

In case you want to use Jupyter make sure that you have it installed in the current environment.

## Development

### Prepare the development environment
You need `python3.7`, `git` and [tox](https://tox.readthedocs.io/en/latest/).  
A recent version of `wheel` and `pip>21.0` are desired for the python versions that you want to use.

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

# download SpaCy and NLTK additional resources
(dev)$ python -m spacy download en_core_web_sm
(dev)$ python -m nltk.downloader punkt wordnet stopwords averaged_perceptron_tagger

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

## DEPRECATED - Installation using Conda

### Only for Windows10
One can use conda native for Windows10 or a Linux-like experience via Windows Subsystems for Linux. For compatibility reasons the second option is the recommended.

#### Recommended - WSL and Conda
- [Install Windows Subsystems for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10#install-the-windows-subsystem-for-linux)
- Install [Ubuntu distribution](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6?activetab=pivot:overviewtab) **This is NOT 16.04 nor 18.04 LTS!**  following the [documentation](https://docs.microsoft.com/en-us/windows/wsl/install-win10#install-your-linux-distribution-of-choice)
- Go again in Windows Store and lunch the installed distribution
- in the terminal corresponding to the distribution download & install miniconda running `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh` and answer to all the questions as desired (I recommend to follow the suggested option) and `source ~/.bashrc` . The terminal should look like `(base) ... :~$` . You can remove the downloaded package `rm Miniconda3-latest-Linux-x86_64.sh`
- install make and g++ (needed by jsonnet in pip): `sudo apt-get install make g++`

#### Native
- install [git](https://git-scm.com/download/win)

### On Leonhard Cluster
- login to the cluster, and go into your folder in `/cluster/work/...`  
 `cd /cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei`  
 If you don't have your folder create one.
 - make sure that you are in the folder that you expect run `pwd` and it should be `/cluster/work/...`
- download `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- install it `bash Miniconda3-latest-Linux-x86_64.sh -p ${PWD}/miniconda3`
    - after you accept the license pay attention at the **location:** where miniconda is going to be installed. It should be miniconda3 folder inside your folder from `/cluster/work/...`
    - I recommend to run conda init
    - if you use also virtualenv please `conda config --set auto_activate_base false` such that conda is not activated by default. In this case please make sure that you  `conda activate` before using it
    - `source ~/.bashrc`
- remove the script used for installation: `rm Miniconda3-latest-Linux-x86_64.sh`
### Clone the project
Clone the project and move to the branch of interest and go to the parent of your repo
```bash
git clone https://github.com/elliottash/narrative-nlp.git
cd narrative-nlp
git branch # to list all branches
git checkout existingbranch # replace existingbranch with the one that you want
```
### Create conda environment
Make sure that you have [conda / miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Use command line and run

```bash
conda env create -f environment.yml
conda activate narrative-nlp
bash ./postBuild # in windows copy and run each line from the ./postBuild file
```
## *Experimental* - Installation using repo2docker (technology used in https://mybinder.org/)
Make sure you have [repo2docker](https://repo2docker.readthedocs.io/en/latest/install.html) - Docker is needed. In case you want to mount a volume just use `--volume`. It might take up to 10 minutes to build the container
```bash
jupyter-repo2docker --volume ./narrative-nlp/Notebooks:Notebooks narrative-nlp/
```

## Usage
### PC
You can easily start a jupyter notebook `jupyter notebook` .
In Windows10 using WSL you have to copy the url similar to `http://127.0.0.1:8888/?token=...` to the desired browser.
### Leonhard
For this step you need to terminals: one where you connect to leonhard and one for the ssh tunneling. You will start jupyter notebook on the cluster but you will use the browser from your PC.

**On Leonhard**
- make sure that you can connect passwordless. If not please follow the [documentation](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#SSH_keys)
- start an interactive session (1 core for 1 hour): `bsub -n 1 -Is -W 1:00 -R "rusage[mem=10000]" bash`  
In case you need more resources ask for them (see [here](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#Resource_requirements))  
    - `bsub -n 1 -Is -W 2:00 -R "rusage[mem=10000,ngpus_excl_p=1] select[gpu_model0==GeForceRTX2080Ti]" bash`  
    - `bsub -n 10 -Is -W 2:00 -R "rusage[mem=10000,ngpus_excl_p=2] select[gpu_model0==GeForceRTX2080Ti] span[ptile=5]" bash` - we asked for 2 nodes, each with 2 GPUs and 10GB per core.

- load some modules: `module load eth_proxy openmpi/4.0.1 cuda/10.0.130`
    - `eth_proxy` is needed in case you want to download/connect to external services  (see [here](https://scicomp.ethz.ch/wiki/Accessing_the_clusters#Security))
    - `openmpi/4.0.1` is needed in case you want to run on multiple nodes
    - `cuda/10.0.130` might be needed for gpus
- activate the desired conda environment: `conda activate narrative-nlp`
- just once: `pip install pycuda mpi4py`
- only for ipyparallel: `ipcluster start --ip="*" --location=$(hostname) --engines=MPI -n 6 &`
- start jupyter notebook without browser: `jupyter notebook --no-browser --ip $(hostname -i)` .
Pay attention at the remote_ip, remote_port and token, e.g. `http://remote_ip:remote_port/?token=token` . The remote_is is not `127.0.0.1`

**On your PC (WSL for windows)**
- use the remote_ip, remote_port and make a decision regarding a port_local, e.g. 8888 if available.
- `ssh username@hostname -N -L port_local:ip_remote:port_remote`
- In your favourite browser open `http://localhost:port_local`. You might need the token.
- Once you are done press on `Quit` in jupyter notebook and close your terminal (`Ctrl+C`), stop ipcluster `ipcluster stop` and `exit` from the interactive session (**on Leonhard**)
### Example
See [Example.ipynb](./Notebooks/Example.ipynb).
In your python script or Jupyter Notebook add to the path `narrative-nlp/code` and use any module as desired
```python
import sys

# update "narrative-nlp/code" with the appropriate path
sys.path.append("narrative-nlp/code")

from utils import preprocess
```
## Development

In case you want to develop further the code you might find useful to have some packages
```bash
python -m pip install black pytest mypy bandit pylint flake8 pydocstyle line_profiler memory_profiler psutil
```

### Tools
- Auto-formatter:
    - [black](https://black.readthedocs.io/en/stable/) that might break the [79 characters maximum line length](https://www.python.org/dev/peps/pep-0008/#maximum-line-length) from PEP8 (see [here](https://github.com/psf/black#line-length))
    - [isort](https://timothycrosley.github.io/isort/): "isort your imports, so you don't have to." . Used with black line length and trailing comma `-l 88 -tc`
- Testing via [pytest](https://docs.pytest.org/en/latest/) with `--doctest-modules` enabled (see [doctest](http://doc.pytest.org/en/latest/doctest.html))
- security issues: [bandit](https://github.com/PyCQA/bandit)
- [mypy](http://mypy-lang.org/)
- [pycodestyle](https://github.com/PyCQA/pycodestyle), [pydocstyle](https://github.com/PyCQA/pydocstyle), [Pylint](https://github.com/PyCQA/pylint), [flake8](https://gitlab.com/pycqa/flake8)

### Guides
- [PEP8](https://www.python.org/dev/peps/pep-0008/)
- [Google Coding Style](http://google.github.io/styleguide/pyguide.html)
- Docstring using Google Style: see [Sphinx 1.3 Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and [pyguide](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
