# Changelog

Changes in relatio

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
