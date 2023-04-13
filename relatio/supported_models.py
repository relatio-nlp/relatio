LANGUAGE_MODELS = {
    # Gensim
    "fasttext-wiki-news-subwords-300": {"language": "english", "size_vectors": 300},
    "word2vec-google-news-300": {"language": "english", "size_vectors": 300},
    "glove-wiki-gigaword-50": {"language": "english", "size_vectors": 50},
    "glove-wiki-gigaword-100": {"language": "english", "size_vectors": 100},
    "glove-wiki-gigaword-200": {"language": "english", "size_vectors": 200},
    "glove-wiki-gigaword-300": {"language": "english", "size_vectors": 300},
    "glove-twitter-25": {"language": "english", "size_vectors": 25},
    "glove-twitter-50": {"language": "english", "size_vectors": 50},
    "glove-twitter-100": {"language": "english", "size_vectors": 100},
    "glove-twitter-200": {"language": "english", "size_vectors": 200},
    # SentenceTransformer (in practice, all sentence transformers models are supported; https://www.sbert.net/docs/pretrained_models.html)
    "all-MiniLM-L6-v2": {
        "language": "english",
        "size_vectors": 384,
    },
    "distiluse-base-multilingual-cased-v2": {
        "language": "multilingual",
        "size_vectors": 512,
    },
    "whaleloops/phrase-bert": {"language": "english", "size_vectors": 768},
    # spaCy
    "en_core_web_sm": {"language": "english", "size_vectors": 96},
    "en_core_web_md": {"language": "english", "size_vectors": 300},
    "en_core_web_lg": {"language": "english", "size_vectors": 300},
    "fr_core_news_sm": {"language": "french", "size_vectors": 96},
    "fr_core_news_md": {"language": "french", "size_vectors": 300},
    "fr_core_news_lg": {"language": "french", "size_vectors": 300},
}
