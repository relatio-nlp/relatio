# Changelog

Changes in relatio

----

## 0.4.0

**Breaking Changes**:
- Minimum Python version: 3.9 (dropped support for 3.7 and 3.8)
- AllenNLP is now optional: `pip install 'relatio[allennlp]'`
  - **Important**: AllenNLP only works with Python 3.9-3.10 due to PyTorch version constraints
  - Users on Python 3.11-3.12 can use all features except Semantic Role Labeling

**Updated Dependencies**:
- pandas >=1.3 (was >=1)
- nltk >=3.6 (was >=3)
- spacy >=3.4 (was >=3.3.2)
- gensim >=4.0 (was >=3)
- scikit-learn >=1.0 (was >=0.22)
- matplotlib >=3.4 (was >=3)
- requests >=2.25 (was >=2)
- Added explicit numpy >=1.21 requirement (supports numpy 2.x)

**Reason for Changes**:
AllenNLP requires torch<1.13.0, which is incompatible with Python 3.11+. AllenNLP has been in maintenance mode since December 2022. Making it optional allows users to choose between:
1. Full functionality (SRL + everything else) on Python 3.9-3.10
2. All features except SRL on Python 3.9-3.12

See MIGRATION_GUIDE.md for upgrade instructions.

----

## 0.3.0

- Upload to PyPI
- New API with Preprocessor() and NarrativeModel() classes
- Support for SentenceTransformers and spaCy models
- Dimension reduction via PCA/UMAP
- Automatic selection of the "optimal" number of clusters based on well-known metrics for HDBSCAN / KMeans.
- Visualization functions for the cluster selection metrics (i.e., elbow method, silhouette score, and DBCV).
- Easy cluster visualization and inspection.
- Multi-lingual support via spaCy language models.
- Dependency parsing to extract SVOs (much much times faster than SRL and pretty decent output).
 
----

## 0.2.1
- Upload to PyPI 
- Update documentation
----

## 0.2.0
- Fixed a bug with `top_n_entities` in `build_narrative_model`. It was not saved in the narrative_model and thus did not filter for the top_n_entities when calling `get_narratives`.
- Set `lemmatize = True` as the default text preprocessing in `build_narrative_model`. 
- Updated `tutorial.ipynb`

----

## 0.1.1
- use `allennlp_models.structured_prediction.predictors.SemanticRoleLabelerPredictor` instead of `allennlp.predictors.predictor`

---

## 0.1
- public release

---
