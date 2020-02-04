# narrative-nlp
code base for constructing narrative statements from text

## Installation
Clone the project and make sure that you have [conda](https://docs.conda.io/projects/conda/en/latest/) installed. Use command line and run

```bash
conda env create -f narrative-nlp/environment_dev.yml
conda activate narrative-nlp-dev
python -m spacy download en_core_web_sm
```

## Usage
In case you want to use the modules from [`narrative-nlp/code`](./code) in your python script or Jupyter Notebook try
```python
import sys

# update "narrative-nlp/code" with the appropriate path
sys.path.append("narrative-nlp/code")

from word_embedding import preprocess
```