"""
Streamlit app: crawl Wikipedia film pages (HTML), build TF–IDF vectors,
cluster movies, and show content-based recommendations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.crawler import CrawledMovie, crawl_movie_by_search
from src.recommender import build_corpus, cluster_movies, recommend


def init_state() -> None:
    if "movies" not in st.session_state:
        st.session_state.movies: list[CrawledMovie] = []


def corpus_titles_texts(movies: list[CrawledMovie]) -> tuple[list[str], list[str]]:
    titles = [m.title for m in movies]
    texts = [m.combined_text for m in movies]
    return titles, texts


def main() -> None:
    st.set_page_config(page_title="Movie crawl & recommend", layout="wide")
    init_state()

    st.title("Movie recommendations from crawled Wikipedia text")
    st.markdown(
        "Add films by search query. Each title **crawls** the English Wikipedia article "
        "(HTTP GET + HTML parsing) for the lead and **Plot** section, then recommends "
        "similar movies using **TF–IDF** and cosine similarity. Optional **k-means** clusters "
        "group films by vocabulary overlap."
    )

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "- **Crawl**: `requests` → Wikipedia HTML → `BeautifulSoup` extracts text.\n"
            "- **NLP**: TF–IDF on title + intro + plot.\n"
            "- **Recommend**: cosine similarity (content-based).\n"
            "- **Cluster**: k-means on the same vectors."
        )
        st.caption("Be polite: ~1s delay between requests. For production, cache responses.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        query = st.text_input(
            "Search Wikipedia (e.g. `Parasite film`, `The Matrix`)",
            placeholder="Parasite (2019 film)",
        )
    with col_b:
        st.write("")
        st.write("")
        do_crawl = st.button("Crawl & add movie", type="primary")

    if do_crawl and query.strip():
        with st.spinner("Searching and crawling Wikipedia HTML…"):
            try:
                movie, alts = crawl_movie_by_search(query.strip())
            except Exception as e:
                st.error(str(e))
            else:
                existing = {m.title for m in st.session_state.movies}
                if movie.title in existing:
                    st.warning(f"Already in list: **{movie.title}**")
                else:
                    st.session_state.movies.append(movie)
                    st.success(f"Added **{movie.title}**")
                with st.expander("Search hits (first is used)"):
                    st.write(alts)

    if not st.session_state.movies:
        st.info("Add at least **two** movies to enable recommendations and clustering.")
        return

    rows = []
    for m in st.session_state.movies:
        rows.append(
            {
                "title": m.title,
                "url": m.url,
                "chars": len(m.combined_text),
                "preview": (m.plot_text[:280] + "…") if len(m.plot_text) > 280 else m.plot_text,
            }
        )
    st.subheader("Crawled corpus")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    titles, texts = corpus_titles_texts(st.session_state.movies)
    corpus = build_corpus(titles, texts)

    c1, c2, c3 = st.columns(3)
    with c1:
        top_k = st.slider("Top similar movies", 1, min(15, max(1, len(titles) - 1)), 5)
    with c2:
        n_clusters = st.number_input(
            "k-means clusters (k)",
            min_value=2,
            max_value=max(2, min(12, len(titles) - 1)),
            value=max(2, min(5, len(titles) - 1)),
        )
    with c3:
        pick = st.selectbox("Recommend similar to", titles)

    rec_df = recommend(corpus, pick, top_k=top_k)
    st.subheader("Recommendations (cosine similarity on TF–IDF)")
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    if len(titles) >= 2:
        cl_df = cluster_movies(corpus, n_clusters=int(n_clusters))
        st.subheader("Clusters (k-means on TF–IDF)")
        st.dataframe(cl_df.sort_values(["cluster", "title"]), use_container_width=True, hide_index=True)

    with st.expander("Raw crawled text for selected movie"):
        sel = next(m for m in st.session_state.movies if m.title == pick)
        st.markdown(f"**URL:** {sel.url}")
        st.text_area("Combined text used for TF–IDF", sel.combined_text, height=240)

    if st.button("Clear all movies"):
        st.session_state.movies = []
        st.rerun()


if __name__ == "__main__":
    main()
