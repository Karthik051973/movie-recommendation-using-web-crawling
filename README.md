# Movie recommendations from crawled Wikipedia text

**Streamlit app** that crawls English Wikipedia film articles, vectorizes plot and summary text with **TF–IDF**, recommends similar movies via **cosine similarity**, and optionally clusters them with **k-means**.

## Description

This project demonstrates **web crawling**, **NLP**, and **unsupervised learning** on movie data. You enter Wikipedia-style search queries; the app resolves them with the MediaWiki API, then **downloads each article’s HTML** (`requests`) and extracts the lead and **Plot** sections using **BeautifulSoup**. Each film becomes one document (title + intro + plot). **TF–IDF** turns these into sparse vectors. **Content-based recommendations** rank other films in your session by cosine similarity—similarity reflects shared wording and themes in the articles, not user ratings. **K-means** groups films in the same vector space for exploration. Requests use a descriptive **User-Agent** and throttling out of respect for Wikipedia’s servers.

## Requirements

- Python 3.10+ recommended

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`). Add several films via the search box, then pick a film to see recommendations and clusters.

## Input tips

Use specific queries so the first search hit is the film article, e.g. `Parasite (2019 film)`, `Interstellar (film)`.

## Stack

Streamlit · requests · BeautifulSoup · lxml · scikit-learn · pandas · NumPy

## Note

This tool is for **education and research**. Crawl **respectfully** (low rate, caching in production). Recommendations are based on **Wikipedia text only**, not IMDb or collaborative filtering.

## License

Add a license file if you publish the repo (e.g. MIT).
