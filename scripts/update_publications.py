#!/usr/bin/env python3
"""
Fetch publications from Google Scholar and update _pages/about.md.

- Only adds papers not already present (matched by title).
- Preserves all existing content; inserts new cards grouped by year.
- Newly added cards are marked with an HTML comment so they can be reviewed.
"""

import re
import sys
import time

SCHOLAR_USER_ID = "I_sHcmgAAAAJ"
MY_NAME = "Haoming Xu"
ABOUT_FILE = "_pages/about.md"

# ── Boundary markers in about.md ─────────────────────────────────────────────
PUB_START_MARKER = "## 📝 Publications {#publications}"
PUB_END_MARKER   = "\n---\n\n## 🚀 Projects"


# ── Scholar helpers ───────────────────────────────────────────────────────────

def get_scholarly():
    try:
        from scholarly import scholarly as _s, ProxyGenerator
        return _s, ProxyGenerator
    except ImportError:
        print("ERROR: 'scholarly' is not installed. Run: pip install scholarly")
        sys.exit(1)


def fetch_publications():
    scholarly, ProxyGenerator = get_scholarly()

    def _fetch(scholar):
        author = scholar.search_author_id(SCHOLAR_USER_ID)
        author = scholar.fill(author, sections=["publications"])
        return author.get("publications", [])

    # 1. Try direct access
    try:
        print("Trying direct Google Scholar access …")
        pubs = _fetch(scholarly)
        print(f"Direct access succeeded: {len(pubs)} publications found.")
        return pubs
    except Exception as e:
        print(f"Direct access failed: {e}")

    # 2. Fall back to free rotating proxies
    try:
        print("Trying with FreeProxies …")
        pg = ProxyGenerator()
        pg.FreeProxies()
        scholarly.use_proxy(pg)
        pubs = _fetch(scholarly)
        print(f"FreeProxies succeeded: {len(pubs)} publications found.")
        return pubs
    except Exception as e:
        print(f"FreeProxies also failed: {e}")
        sys.exit(1)


def fill_pub(scholarly_module, pub):
    """Fill publication details (abstract, url, etc.). Fails silently."""
    try:
        return scholarly_module.fill(pub)
    except Exception as e:
        print(f"  Warning: could not fill details: {e}")
        return pub


# ── Name / author helpers ─────────────────────────────────────────────────────

def _normalize_name(raw: str) -> str:
    """Convert 'Last, First' or 'First Last' → 'First Last'."""
    raw = raw.strip()
    if "," in raw:
        parts = [p.strip() for p in raw.split(",", 1)]
        return f"{parts[1]} {parts[0]}"
    return raw


def format_authors(author_field: str, my_name: str = MY_NAME) -> str:
    """
    scholarly stores authors as 'A and B and C' (BibTeX style).
    Returns an HTML string with MY_NAME bolded.
    """
    if not author_field:
        return ""
    raw_authors = [a.strip() for a in author_field.split(" and ")]
    formatted = []
    for raw in raw_authors:
        name = _normalize_name(raw)
        if my_name.lower() in name.lower():
            formatted.append(f"<strong>{name}</strong>")
        else:
            formatted.append(name)
    return ", ".join(formatted)


# ── Card builder ──────────────────────────────────────────────────────────────

def _first_sentence(text: str, max_chars: int = 220) -> str:
    text = text.strip()
    m = re.search(r"(?<=[.!?])\s", text)
    snippet = text[: m.start()].strip() if m else text
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "…"
    return snippet


def make_card(pub: dict) -> str:
    bib = pub.get("bib", {})

    title   = bib.get("title", "Untitled").strip()
    url     = pub.get("pub_url") or "#"
    year    = str(bib.get("pub_year", "")).strip()
    venue   = (bib.get("venue") or bib.get("journal") or
               bib.get("booktitle") or "").strip()
    authors = format_authors(bib.get("author", ""))
    abstract = bib.get("abstract", "")
    description = _first_sentence(abstract) if abstract else "TODO: add description."

    venue_str = f"{venue} {year}".strip()

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
        f'      <div class="card-authors">{authors}</div>\n'
        f'      <div class="card-description">{description}</div>\n'
        f'    </div>\n'
        f'  </div>\n'
        f'</div>'
    )


# ── about.md helpers ──────────────────────────────────────────────────────────

def existing_titles(content: str) -> set:
    """Return lower-cased set of all paper titles already in about.md."""
    pattern = r'class="card-title"><a [^>]+>([^<]+)</a></div>'
    return {t.lower().strip() for t in re.findall(pattern, content)}


def _pub_section_bounds(content: str):
    start = content.find(PUB_START_MARKER)
    end   = content.find(PUB_END_MARKER, start)
    if start == -1 or end == -1:
        raise ValueError(
            f"Could not locate publications section.\n"
            f"Expected markers:\n  '{PUB_START_MARKER}'\n  '{PUB_END_MARKER}'"
        )
    # end points at the '\n' before '---'; keep that '\n---\n\n## 🚀 Projects'
    return start, end


def insert_new_cards(content: str, new_by_year: dict) -> tuple[str, int]:
    """
    Insert new publication cards into the publication section of content.
    Returns (updated_content, number_of_cards_added).
    """
    if not new_by_year:
        return content, 0

    start, end = _pub_section_bounds(content)
    section = content[start:end]
    total_added = 0

    for year in sorted(new_by_year.keys(), reverse=True):
        cards = new_by_year[year]
        year_marker = f'<div class="publication-year">{year}</div>'

        if year_marker in section:
            # Append cards just after the existing year marker
            pos = section.find(year_marker) + len(year_marker)
            insert_block = "\n\n" + "\n\n".join(cards)
            section = section[:pos] + insert_block + section[pos:]
        else:
            # Find the first existing year-marker that is older (smaller year)
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
                # All existing years are newer → append at the bottom of section
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
    scholarly_module, _ = get_scholarly()
    from scholarly import scholarly

    print("Fetching publications from Google Scholar …")
    publications = fetch_publications()
    print(f"{len(publications)} publications retrieved.")

    with open(ABOUT_FILE, encoding="utf-8") as f:
        content = f.read()

    known = existing_titles(content)
    print(f"{len(known)} publications already on the page.")

    new_by_year: dict[str, list[str]] = {}
    for pub in publications:
        bib = pub.get("bib", {})
        title = bib.get("title", "").strip()
        if not title:
            continue
        if title.lower() in known:
            print(f"  [skip] {title}")
            continue

        print(f"  [new]  {title}")
        pub = fill_pub(scholarly, pub)
        time.sleep(1)          # polite delay

        year = str(pub.get("bib", {}).get("pub_year", "unknown"))
        new_by_year.setdefault(year, []).append(make_card(pub))

    if not new_by_year:
        print("No new publications – nothing to commit.")
        return

    updated, count = insert_new_cards(content, new_by_year)

    with open(ABOUT_FILE, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"\nDone: {count} new card(s) added to {ABOUT_FILE}.")
    print("Please review the '<!-- AUTO-ADDED -->' entries and fill in any missing details.")


if __name__ == "__main__":
    main()
