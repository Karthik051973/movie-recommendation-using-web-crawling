"""
Web crawling: fetch Wikipedia film articles and extract plot / summary text.

Uses a descriptive User-Agent and polite delays (see Wikipedia robots.txt etiquette).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_BASE = "https://en.wikipedia.org/wiki/"
USER_AGENT = (
    "MovieNLPResearchBot/1.0 (educational project; contact: local) "
    "requests/2.x; +https://en.wikipedia.org/api/rest_v1/"
)
MIN_DELAY_SEC = 1.0
_last_request_ts = 0.0


def _throttle() -> None:
    global _last_request_ts
    now = time.monotonic()
    wait = MIN_DELAY_SEC - (now - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.monotonic()


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return s


@dataclass
class CrawledMovie:
    title: str
    url: str
    plot_text: str
    intro_text: str

    @property
    def combined_text(self) -> str:
        parts = [self.title, self.intro_text, self.plot_text]
        return "\n\n".join(p for p in parts if p)


def search_wikipedia_titles(query: str, limit: int = 8) -> list[str]:
    """Use the MediaWiki API (not HTML) to find candidate page titles."""
    _throttle()
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    r = _session().get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hits = data.get("query", {}).get("search", [])
    return [h["title"] for h in hits]


def wiki_article_url(page_title: str) -> str:
    """Build the canonical /wiki/ URL for a Wikipedia page title."""
    path = page_title.replace(" ", "_")
    return WIKI_BASE + quote(path, safe="/():%!,.&'")


def _fetch_article_html(page_title: str) -> str:
    _throttle()
    url = wiki_article_url(page_title)
    r = _session().get(url, timeout=30)
    r.raise_for_status()
    return r.text


def _strip_citations_and_noise(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _main_parser_output(soup: BeautifulSoup) -> Tag | None:
    """Article body lives under #mw-content-text (not sidebar mw-parser-output snippets)."""
    ct = soup.find("div", id="mw-content-text")
    if ct:
        po = ct.find("div", class_="mw-parser-output")
        if po:
            return po
    return soup.find("div", class_="mw-parser-output")


def _heading_block_for_section(h2: Tag) -> Tag:
    """Wikipedia wraps section titles in div.mw-heading; paragraphs follow that div."""
    parent = h2.parent
    if (
        parent
        and parent.name == "div"
        and parent.get("class")
        and "mw-heading" in parent.get("class", [])
    ):
        return parent
    return h2


def _extract_section_after_heading(soup: BeautifulSoup, heading_ids: tuple[str, ...]) -> str:
    """Collect <p> text after the first matching Plot/Synopsis heading until the next section."""
    h2 = None
    for hid in heading_ids:
        h2 = soup.find("h2", id=hid)
        if h2:
            break
        span = soup.find("span", class_="mw-headline", id=hid)
        if span and span.parent and span.parent.name == "h2":
            h2 = span.parent
            break
    if not h2:
        return ""

    start = _heading_block_for_section(h2)
    parts: list[str] = []
    for el in start.next_siblings:
        if isinstance(el, NavigableString):
            continue
        if not isinstance(el, Tag):
            continue
        if el.name == "p":
            t = el.get_text(separator=" ", strip=True)
            if t:
                parts.append(t)
            continue
        if el.name == "div" and el.get("class") and "mw-heading" in el.get("class", []):
            break
        if el.name == "h2":
            break
    return _strip_citations_and_noise(" ".join(parts))


def _extract_intro(soup: BeautifulSoup) -> str:
    content = _main_parser_output(soup)
    if not content:
        return ""
    paras: list[str] = []
    for child in content.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "p":
            t = child.get_text(separator=" ", strip=True)
            if t and not t.startswith("Coordinates:"):
                paras.append(t)
        elif child.name in ("h2", "h3", "table", "div"):
            if paras and child.name in ("h2", "h3"):
                break
            if paras and child.name == "div" and child.get("class"):
                if "mw-heading" in child.get("class", []):
                    break
    return _strip_citations_and_noise(" ".join(paras[:6]))


def crawl_wikipedia_movie(page_title: str) -> CrawledMovie:
    """
    Crawl a Wikipedia film page by title: download HTML and parse Plot + lead.
    """
    html = _fetch_article_html(page_title)
    soup = BeautifulSoup(html, "lxml")
    plot = _extract_section_after_heading(soup, ("Plot", "Synopsis", "Summary"))
    intro = _extract_intro(soup)
    url = wiki_article_url(page_title)

    display_title = page_title.replace("_", " ")
    if not plot and intro:
        plot = intro
    return CrawledMovie(
        title=display_title,
        url=url,
        plot_text=plot or intro,
        intro_text=intro,
    )


def crawl_movie_by_search(query: str) -> tuple[CrawledMovie, list[str]]:
    """
    Search Wikipedia, take the first result, then crawl that page.
    Returns (movie, alternate_titles_from_search).
    """
    titles = search_wikipedia_titles(query, limit=8)
    if not titles:
        raise ValueError(f"No Wikipedia results for: {query!r}")
    chosen = titles[0]
    movie = crawl_wikipedia_movie(chosen)
    return movie, titles
