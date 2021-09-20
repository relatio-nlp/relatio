# Changelog

Changes in relatio

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
