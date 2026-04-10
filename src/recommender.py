"""TF-IDF vectors, clustering, and content-based similarity recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MovieCorpus:
    titles: list[str]
    texts: list[str]
    vectorizer: TfidfVectorizer
    matrix: np.ndarray  # sparse in practice; kept flexible

    def similarity_matrix(self) -> np.ndarray:
        x = self.matrix
        if hasattr(x, "toarray"):
            # avoid densifying huge matrices; cosine_similarity accepts sparse
            return cosine_similarity(x)
        return cosine_similarity(x)


def build_corpus(
    titles: list[str],
    texts: list[str],
    max_features: int = 8000,
    ngram_range: tuple[int, int] = (1, 2),
) -> MovieCorpus:
    if len(titles) != len(texts) or not titles:
        raise ValueError("titles and texts must be non-empty and equal length")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return MovieCorpus(titles=titles, texts=texts, vectorizer=vectorizer, matrix=matrix)


def recommend(
    corpus: MovieCorpus,
    query_title: str,
    top_k: int = 5,
) -> pd.DataFrame:
    titles = corpus.titles
    if query_title not in titles:
        raise ValueError(f"Unknown title: {query_title!r}")
    idx = titles.index(query_title)
    sims = cosine_similarity(corpus.matrix[idx], corpus.matrix).ravel()
    order = np.argsort(-sims)
    rows = []
    for j in order:
        if titles[j] == query_title:
            continue
        rows.append({"title": titles[j], "similarity": float(sims[j])})
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def cluster_movies(corpus: MovieCorpus, n_clusters: int | None = None) -> pd.DataFrame:
    n = len(corpus.titles)
    if n < 2:
        return pd.DataFrame({"title": corpus.titles, "cluster": [0] * n})
    k = n_clusters or max(2, min(8, n // 2))
    k = min(k, n - 1)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(corpus.matrix)
    return pd.DataFrame({"title": corpus.titles, "cluster": labels})
