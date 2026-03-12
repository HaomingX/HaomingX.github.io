#!/usr/bin/env python3
"""
Fetch publications from Semantic Scholar and update _pages/about.md.

Semantic Scholar has a free public API that doesn't require scraping or proxies.
It is robust from CI/CD environments like GitHub Actions.

- Disambiguates your author profile by matching known paper titles.
- Only adds papers not already present (matched by title, case-insensitive).
- Preserves all existing content; inserts new cards grouped by year.
- Newly added cards are marked with <!-- AUTO-ADDED --> for review.
"""

import re
import sys
import time
import requests

# ── Config ────────────────────────────────────────────────────────────────────
MY_NAME   = "Haoming Xu"
ABOUT_FILE = "_pages/about.md"

# Semantic Scholar API base
SS_API = "https://api.semanticscholar.org/graph/v1"

# One or more distinctive title fragments from your known papers.
# Used to pick the right "Haoming Xu" from search results.
KNOWN_TITLE_FRAGMENTS = [
    "Relearn",
    "MLLM Can See",
    "ZJUKLAB",
    "Illusions of Confidence",
]

# Optional: set env var SEMANTIC_SCHOLAR_API_KEY for higher rate limits (free key).
import os
SS_HEADERS = {}
if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
    SS_HEADERS["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]

# ── Boundary markers in about.md ─────────────────────────────────────────────
PUB_START_MARKER = "## 📝 Publications {#publications}"
PUB_END_MARKER   = "\n---\n\n## 🚀 Projects"


# ── Semantic Scholar helpers ──────────────────────────────────────────────────

def _get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=SS_HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate-limited; waiting {wait}s …")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  Request error ({e}), retrying …")
            time.sleep(3)


def find_author_id() -> str:
    """
    Search Semantic Scholar for MY_NAME and return the authorId whose
    paper list contains at least one KNOWN_TITLE_FRAGMENT.
    """
    data = _get(
        f"{SS_API}/author/search",
        params={
            "query": MY_NAME,
            "fields": "authorId,name,papers.title",
            "limit": 10,
        },
    )

    for author in data.get("data", []):
        papers = author.get("papers", [])
        titles = " ".join(p.get("title", "") for p in papers).lower()
        if any(frag.lower() in titles for frag in KNOWN_TITLE_FRAGMENTS):
            print(f"Found author: {author['name']} (id={author['authorId']})")
            return author["authorId"]

    print("ERROR: Could not identify your Semantic Scholar author profile.")
    print("Candidates found:")
    for a in data.get("data", []):
        print(f"  {a.get('name')} ({a.get('authorId')})")
    sys.exit(1)


def fetch_papers(author_id: str) -> list:
    fields = (
        "title,year,venue,authors,externalIds,"
        "abstract,publicationVenue,openAccessPdf"
    )
    data = _get(
        f"{SS_API}/author/{author_id}/papers",
        params={"fields": fields, "limit": 100},
    )
    return data.get("data", [])


# ── Author / card helpers ─────────────────────────────────────────────────────

def format_authors(authors: list, my_name: str = MY_NAME) -> str:
    parts = []
    for a in authors:
        name = a.get("name", "")
        if my_name.lower() in name.lower():
            parts.append(f"<strong>{name}</strong>")
        else:
            parts.append(name)
    return ", ".join(parts)


def _first_sentence(text: str, max_chars: int = 220) -> str:
    text = text.strip()
    m = re.search(r"(?<=[.!?])\s", text)
    snippet = text[: m.start()].strip() if m else text
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "…"
    return snippet


def paper_url(paper: dict) -> str:
    ext = paper.get("externalIds") or {}
    if ext.get("ArXiv"):
        return f"https://arxiv.org/abs/{ext['ArXiv']}"
    if ext.get("DOI"):
        return f"https://doi.org/{ext['DOI']}"
    pdf = paper.get("openAccessPdf") or {}
    if pdf.get("url"):
        return pdf["url"]
    return "#"


def venue_string(paper: dict) -> str:
    pv = paper.get("publicationVenue") or {}
    name = pv.get("name") or paper.get("venue") or ""
    year = paper.get("year") or ""
    return f"{name} {year}".strip()


def make_card(paper: dict) -> str:
    title       = (paper.get("title") or "Untitled").strip()
    url         = paper_url(paper)
    authors_html = format_authors(paper.get("authors") or [])
    venue_str   = venue_string(paper)
    abstract    = paper.get("abstract") or ""
    description = _first_sentence(abstract) if abstract else "TODO: add description."

    return (
        f'<!-- AUTO-ADDED: review and edit as needed -->\n'
        f'<div class="publication-card">\n'
        f'  <div class="card-content">\n'
        f'    <div class="card-body">\n'
        f'      <div class="card-title"><a href="{url}">{title}</a></div>\n'
        f'      <div class="card-meta">\n'
        f'        <span class="card-meta-icons">\n'
        f'          <a href="{url}" class="card-meta-icon" title="Paper">'
        f'<i class="fas fa-file-pdf"></i></a>\n'
        f'        </span>\n'
        f'        <span>{venue_str}</span>\n'
        f'      </div>\n'
        f'      <div class="card-authors">{authors_html}</div>\n'
        f'      <div class="card-description">{description}</div>\n'
        f'    </div>\n'
        f'  </div>\n'
        f'</div>'
    )


# ── about.md helpers ──────────────────────────────────────────────────────────

def existing_titles(content: str) -> set:
    pattern = r'class="card-title"><a [^>]+>([^<]+)</a></div>'
    return {t.lower().strip() for t in re.findall(pattern, content)}


def _pub_section_bounds(content: str):
    start = content.find(PUB_START_MARKER)
    end   = content.find(PUB_END_MARKER, start)
    if start == -1 or end == -1:
        raise ValueError(
            "Could not locate publications section.\n"
            f"Expected: '{PUB_START_MARKER}' … '{PUB_END_MARKER}'"
        )
    return start, end


def insert_new_cards(content: str, new_by_year: dict) -> tuple:
    if not new_by_year:
        return content, 0

    start, end = _pub_section_bounds(content)
    section = content[start:end]
    total_added = 0

    for year in sorted(new_by_year.keys(), reverse=True):
        cards = new_by_year[year]
        year_marker = f'<div class="publication-year">{year}</div>'

        if year_marker in section:
            pos = section.find(year_marker) + len(year_marker)
            section = section[:pos] + "\n\n" + "\n\n".join(cards) + section[pos:]
        else:
            existing_years = [
                (int(m.group(1)), m.start())
                for m in re.finditer(
                    r'<div class="publication-year">(\d+)</div>', section
                )
            ]
            insert_before = None
            for ey, epos in sorted(existing_years, reverse=True):
                if ey < int(year):
                    insert_before = epos
                    break

            year_block = year_marker + "\n\n" + "\n\n".join(cards) + "\n"
            if insert_before is not None:
                section = (
                    section[:insert_before]
                    + year_block + "\n"
                    + section[insert_before:]
                )
            else:
                section = section + "\n\n" + year_block

        total_added += len(cards)

    return content[:start] + section + content[end:], total_added


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Finding author on Semantic Scholar …")
    author_id = find_author_id()

    print("Fetching papers …")
    papers = fetch_papers(author_id)
    print(f"{len(papers)} papers retrieved.")

    with open(ABOUT_FILE, encoding="utf-8") as f:
        content = f.read()

    known = existing_titles(content)
    print(f"{len(known)} publications already on the page.")

    new_by_year: dict = {}
    for paper in papers:
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        if title.lower() in known:
            print(f"  [skip] {title}")
            continue

        year = str(paper.get("year") or "unknown")
        print(f"  [new]  {title} ({year})")
        new_by_year.setdefault(year, []).append(make_card(paper))

    if not new_by_year:
        print("No new publications – nothing to commit.")
        return

    updated, count = insert_new_cards(content, new_by_year)

    with open(ABOUT_FILE, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"\nDone: {count} new card(s) added to {ABOUT_FILE}.")
    print("Review '<!-- AUTO-ADDED -->' entries and fill in missing GitHub links / venue names.")


if __name__ == "__main__":
    main()
