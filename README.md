# `relatio`

A Python package to extract underlying narrative statements from text. 

* "relatio" is Latin for "storytelling" (pronounced _reh-LOTT-ee-oh_).
* Motivated, described, and applied in "[Text Semantics Capture Political and Economic Narratives" (2021)](https://arxiv.org/abs/2108.01720).
* Interactive tutorial notebook is [here](https://colab.research.google.com/github/relatio-nlp/relatio/blob/master/tutorial/tutorial_english.ipynb).
* See [here](https://sites.google.com/view/trump-narratives/trump-tweet-archive) for graphical demo of system outputs.

## What can this package do?

1. Identify Agent-Verb-Patient (AVP) / Subject-Verb-Object (SVO) triplets in the text

    - AVPs are obtained via Semantic Role Labeling.
    - SVOs are obtained via Dependency Parsing.
    - A concrete example of AVP/SVO extraction: 
    
    Original sentence: "Taxes kill jobs and hinder innovation."

    Triplets: [('taxes', 'kill', 'jobs'), ('taxes','hinder','innovation')]

2. Group agents and patients into interpretable entities in two ways:

    - Supervised classification of entities. Simply provide a list of entities and we will filter the triplets for you (e.g., ['Barack Obama', 'government', ...]).
    - Unsupervised classification via clustering of entities. We represent agents and patients as text embeddings and cluster them via KMeans or HDBSCAN. The optimal number of topics is data-driven.
    - A concrete example of a cluster:

    Interpretable entity: "tax"  
    Related phrases: ['income tax', 'the tax rates', 'taxation in this country', etc.]

3. Visualize clusters and resulting narratives.

We currently support French and English out-of-the-box. You can also provide us with a custom SVO-extraction function for any language supported by spaCy.

## Installation

**Python Requirements**: 3.9, 3.10, 3.11, or 3.12
**Platforms**: Linux and macOS

### Important: Semantic Role Labeling (SRL) Limitations

The **Semantic Role Labeling** feature (AVP extraction) uses AllenNLP, which has compatibility constraints:
- **AllenNLP only works with Python 3.9-3.10** (not 3.11+)
- AllenNLP is now in maintenance mode and requires older PyTorch versions

**Choose your installation based on your needs:**

### Option 1: Full Installation (includes SRL)

**Use Python 3.9 or 3.10 only**

```bash
# Create environment with Python 3.10
conda create -n relatio python=3.10 -y
conda activate relatio

# Install relatio with AllenNLP support
python -m pip install -U pip wheel setuptools
python -m pip install -U 'relatio[allennlp]'
```

### Option 2: Without SRL (Dependency Parsing only)

**Can use Python 3.9, 3.10, 3.11, or 3.12**

```bash
# Install relatio without AllenNLP
python -m pip install -U pip wheel setuptools
python -m pip install -U relatio
```

**What you get:**
- ✅ SVO extraction via Dependency Parsing
- ✅ Clustering and visualization
- ✅ All other features
- ❌ AVP extraction via Semantic Role Labeling (requires AllenNLP)

### Legacy Support

For Python 3.7 or 3.8, use relatio version 0.3.0:
```bash
pip install relatio==0.3.0
```

## Quickstart 

Please see our hands-on tutorials:
* [Trump Tweet Archive](./tutorial/tutorial_english.ipynb)
* [Tweets of French Politicians](./tutorial/tutorial_french.ipynb)

## Team

`relatio` is brought to you by

* [Elliott Ash](elliottash.com), ETH Zurich
* [Germain Gauthier](https://pinchofdata.github.io/germaingauthier/), CREST
* [Andrei Plamada](https://www.linkedin.com/in/andreiplamada), ETH Zurich
* [Philine Widmer](https://philinew.github.io/), University of St.Gallen

with a special thanks for support of [ETH Scientific IT Services](https://sis.id.ethz.ch/).

If you are interested in contributing to the project please read the [Development Guide](./doc/Development.md).

## Disclaimer

Remember that this is a research tool :)
