"""
TEST VERSION: Fetch 50 papers from OpenAlex for pipeline testing.
This is a minimal version to verify the pipeline works before running the full 5000.
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import random

# ============= Configuration =============
TARGET_PAPERS = 50  # Very small test set for quick validation
OUTPUT_DIR = Path("papers/openalex_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POLITE_EMAIL = "alina@jhu.edu"  # Update with your email
BASE_URL = "https://api.openalex.org/works"

# Date range
START_YEAR = 2020  # Narrower range for testing
END_YEAR = 2026

# Focused test queries
TEST_QUERIES = [
    {"title_abstract": "MoS2 doping", "keywords": None},
    {"title_abstract": "graphene doping substitutional", "keywords": None},
    {"title_abstract": "site-selective doping 2D materials", "keywords": None},
    {"title_abstract": "2D materials", "keywords": None},
]


def build_openalex_filter(query_dict: Dict, start_year: int, end_year: int) -> str:
    """Build OpenAlex filter string from query dictionary."""
    filters = []

    if query_dict.get("title_abstract"):
        search_term = query_dict["title_abstract"]
        filters.append(f'title_and_abstract.search:"{search_term}"')

    if query_dict.get("keywords"):
        for kw in query_dict["keywords"]:
            filters.append(f'keywords.keyword:"{kw}"')

    filters.append(f'publication_year:{start_year}-{end_year}')
    filters.append('type:article|preprint')

    return ",".join(filters)


def fetch_papers_openalex(filter_str: str, max_results: int = 50) -> List[Dict]:
    """Fetch papers from OpenAlex API."""
    papers = []
    page = 1
    per_page = min(200, max_results)

    headers = {
        'User-Agent': f'mailto:{POLITE_EMAIL}',
        'Accept': 'application/json'
    }

    while len(papers) < max_results:
        params = {
            'filter': filter_str,
            'per_page': per_page,
            'page': page,
            'select': 'id,doi,title,publication_year,abstract_inverted_index,keywords,open_access'
        }

        try:
            response = requests.get(BASE_URL, params=params, headers=headers, timeout=60)

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 10))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            if not results:
                break

            papers.extend(results)
            print(f"  Fetched page {page}: {len(results)} papers (total: {len(papers)})")

            meta = data.get('meta', {})
            if page >= meta.get('count', 0) // per_page + 1:
                break

            page += 1
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            time.sleep(5)
            continue

    return papers[:max_results]


def invert_abstract_index(inverted_index: Optional[Dict]) -> str:
    """Convert OpenAlex inverted index to plain text."""
    if not inverted_index:
        return ""

    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))

        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
    except Exception as e:
        print(f"Error inverting abstract: {e}")
        return ""


def extract_paper_metadata(paper: Dict) -> Dict:
    """Extract relevant metadata from OpenAlex paper."""
    abstract = invert_abstract_index(paper.get('abstract_inverted_index'))

    keywords = []
    if paper.get('keywords'):
        keywords = [kw.get('keyword', '') for kw in paper['keywords'] if kw.get('keyword')]

    doi = paper.get('doi', '').replace('https://doi.org/', '') if paper.get('doi') else None

    pdf_url = None
    is_oa = False
    if paper.get('open_access'):
        is_oa = paper['open_access'].get('is_oa', False)
        pdf_url = paper['open_access'].get('oa_url')

    return {
        'openalex_id': paper.get('id', ''),
        'doi': doi,
        'title': paper.get('title', ''),
        'abstract': abstract,
        'keywords': keywords,
        'publication_year': paper.get('publication_year'),
        'is_open_access': is_oa,
        'pdf_url': pdf_url,
    }


def main():
    """Fetch test set of papers."""
    all_papers = []
    seen_ids = set()

    output_file = OUTPUT_DIR / "openalex_test_papers.json"

    print(f"\n{'='*60}")
    print(f"Fetching TEST set from OpenAlex (Target: {TARGET_PAPERS})")
    print(f"{'='*60}\n")

    for i, query_dict in enumerate(TEST_QUERIES):
        if len(all_papers) >= TARGET_PAPERS:
            break

        print(f"\n[Query {i+1}/{len(TEST_QUERIES)}]")
        print(f"Search: {query_dict.get('title_abstract', 'N/A')}")

        filter_str = build_openalex_filter(query_dict, START_YEAR, END_YEAR)
        remaining = TARGET_PAPERS - len(all_papers)

        raw_papers = fetch_papers_openalex(filter_str, max_results=remaining * 2)

        new_count = 0
        for paper in raw_papers:
            paper_id = paper.get('id', '')
            if paper_id and paper_id not in seen_ids:
                metadata = extract_paper_metadata(paper)

                if metadata['abstract'] and len(metadata['abstract']) > 100:
                    all_papers.append(metadata)
                    seen_ids.add(paper_id)
                    new_count += 1

                    if len(all_papers) >= TARGET_PAPERS:
                        break

        print(f"  Added {new_count} new papers (total: {len(all_papers)})")

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_papers': len(all_papers),
                'date_fetched': datetime.now().isoformat(),
                'source': 'OpenAlex API (TEST)'
            },
            'papers': all_papers
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Test Fetch Complete: {len(all_papers)} papers")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")

    # Save abstracts as text files
    abstracts_dir = OUTPUT_DIR / "abstracts"
    abstracts_dir.mkdir(exist_ok=True)

    for i, paper in enumerate(all_papers):
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in paper['title'])
        safe_title = safe_title[:80]
        filename = f"{i+1:03d}_{safe_title}.txt"

        content = f"TITLE: {paper['title']}\n\n"
        if paper.get('keywords'):
            content += f"KEYWORDS: {', '.join(paper['keywords'])}\n\n"
        content += f"ABSTRACT:\n{paper['abstract']}\n"

        (abstracts_dir / filename).write_text(content, encoding='utf-8')

    print(f"Abstracts saved to: {abstracts_dir}/")


if __name__ == "__main__":
    main()
