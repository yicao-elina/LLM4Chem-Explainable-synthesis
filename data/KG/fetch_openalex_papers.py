"""
Fetch ~5000 papers related to 2D material doping from OpenAlex API (2015-2026).
Extracts title, abstract, keywords, DOI, and other metadata.
Saves as JSON for further processing.

OpenAlex API docs: https://docs.openalex.org/
No API key required, but polite mode recommended with email in User-Agent.
Rate limit: 100,000 requests/day for polite pool, 10 req/sec.
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import random

# ============= Configuration =============
TARGET_PAPERS = 5000
OUTPUT_DIR = Path("papers/openalex_metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 200  # OpenAlex supports up to 200 per page
POLITE_EMAIL = "alina@jhu.edu"  # IMPORTANT: Replace with your email for polite API pool (10x higher rate limit!)
BASE_URL = "https://api.openalex.org/works"

# Date range
START_YEAR = 2015
END_YEAR = 2026

# ============= Query Design =============
# OpenAlex uses a different query syntax than Semantic Scholar
# We'll create targeted queries for 2D materials doping
OPENALEX_QUERIES = [
    # Core 2D materials doping
    {
        "title_abstract": "2D materials doping",
        "keywords": ["graphene", "MoS2", "transition metal dichalcogenide"]
    },
    {
        "title_abstract": "van der Waals doping",
        "keywords": ["layered materials", "2D materials"]
    },
    {
        "title_abstract": "site-selective doping 2D",
        "keywords": None
    },
    # TMD specific
    {
        "title_abstract": "MoS2 doping",
        "keywords": None
    },
    {
        "title_abstract": "WS2 doping",
        "keywords": None
    },
    {
        "title_abstract": "WSe2 doping",
        "keywords": None
    },
    {
        "title_abstract": "MoSe2 doping",
        "keywords": None
    },
    # Graphene
    {
        "title_abstract": "graphene doping",
        "keywords": ["chemical doping", "substitutional"]
    },
    {
        "title_abstract": "graphene functionalization",
        "keywords": ["adatom", "substitution"]
    },
    # hBN
    {
        "title_abstract": "hexagonal boron nitride doping",
        "keywords": None
    },
    {
        "title_abstract": "hBN doping",
        "keywords": None
    },
    # Topological insulators
    {
        "title_abstract": "Bi2Te3 doping",
        "keywords": None
    },
    {
        "title_abstract": "Bi2Se3 doping",
        "keywords": None
    },
    {
        "title_abstract": "Sb2Te3 doping",
        "keywords": None
    },
    # Intercalation
    {
        "title_abstract": "intercalation layered materials",
        "keywords": ["2D", "van der Waals"]
    },
    {
        "title_abstract": "electrochemical intercalation 2D",
        "keywords": None
    },
    # Site selectivity
    {
        "title_abstract": "substitutional doping 2D materials",
        "keywords": None
    },
    {
        "title_abstract": "interstitial doping layered",
        "keywords": None
    },
    # Processing methods
    {
        "title_abstract": "chemical vapor deposition doping",
        "keywords": ["2D materials", "TMD"]
    },
    {
        "title_abstract": "molecular beam epitaxy doping 2D",
        "keywords": None
    },
    {
        "title_abstract": "ion implantation 2D materials",
        "keywords": None
    },
    # Broader queries for synthesis and processing
    {
        "title_abstract": "2D material synthesis",
        "keywords": ["CVD", "MBE", "exfoliation", "hydrothermal"]
    },
    {
        "title_abstract": "graphene synthesis",
        "keywords": None
    },
    {
        "title_abstract": "MoS2 synthesis",
        "keywords": None
    },
    {
        "title_abstract": "transition metal dichalcogenide synthesis",
        "keywords": None
    },
    {
        "title_abstract": "van der Waals heterostructures",
        "keywords": ["synthesis", "fabrication", "stacking"]
    },
]


def build_openalex_filter(query_dict: Dict, start_year: int, end_year: int) -> str:
    """
    Build OpenAlex filter string from query dictionary.

    OpenAlex filter syntax:
    - title_and_abstract.search: searches title and abstract
    - keywords.keyword: exact keyword match
    - publication_year: year range
    - type: work type (article, preprint, etc.)
    """
    filters = []
    search_terms_combined = []

    # Add title and abstract search term
    if query_dict.get("title_abstract"):
        # Wrap the title_abstract in quotes for phrase search
        search_terms_combined.append(f'"{query_dict["title_abstract"]}"')

    # Add keywords to the combined search terms
    if query_dict.get("keywords"):
        for kw in query_dict["keywords"]:
            if ' ' in kw:
                search_terms_combined.append(f'"{kw}"')
            else:
                search_terms_combined.append(kw)

    if search_terms_combined:
        # Join all search parts with OR, then wrap the whole thing in parentheses
        full_search_string = " OR ".join(search_terms_combined)
        filters.append(f'title_and_abstract.search:({full_search_string})')

    # Year range
    filters.append(f'publication_year:{start_year}-{end_year}')

    # Only journal articles and preprints
    filters.append('type:article|preprint')

    return ",".join(filters)


def fetch_papers_openalex(filter_str: str, max_results: int = 200) -> List[Dict]:
    """
    Fetch papers from OpenAlex API using filter string.

    Args:
        filter_str: OpenAlex filter string
        max_results: Maximum number of results to fetch

    Returns:
        List of paper metadata dictionaries
    """
    papers = []
    page = 1
    per_page = min(BATCH_SIZE, max_results)

    headers = {
        'User-Agent': f'mailto:{POLITE_EMAIL}',
        'Accept': 'application/json'
    }

    while len(papers) < max_results:
        params = {
            'filter': filter_str,
            'per_page': per_page,
            'page': page,
            'select': 'id,doi,title,publication_year,publication_date,abstract_inverted_index,authorships,concepts,keywords,open_access,cited_by_count,primary_location'
        }

        try:
            response = requests.get(BASE_URL, params=params, headers=headers, timeout=60)

            if response.status_code == 429:
                # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
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

            # Check if there are more pages
            meta = data.get('meta', {})
            if page >= meta.get('count', 0) // per_page + 1:
                break

            page += 1

            # Polite delay (10 req/sec max = 100ms between requests)
            time.sleep(0.15 + random.uniform(0, 0.1))

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            time.sleep(5)
            continue

    return papers[:max_results]


def invert_abstract_index(inverted_index: Optional[Dict]) -> str:
    """
    Convert OpenAlex inverted index format to plain text.

    OpenAlex stores abstracts as inverted indexes for efficiency:
    {"the": [0, 5], "quick": [1], "brown": [2], ...}
    """
    if not inverted_index:
        return ""

    try:
        # Create list of (position, word) tuples
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))

        # Sort by position and join
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
    except Exception as e:
        print(f"Error inverting abstract index: {e}")
        return ""


def extract_paper_metadata(paper: Dict) -> Dict:
    """
    Extract relevant metadata from OpenAlex paper object.
    """
    # Convert abstract from inverted index
    abstract = invert_abstract_index(paper.get('abstract_inverted_index'))

    # Extract keywords
    keywords = []
    if paper.get('keywords'):
        keywords = [kw.get('keyword', '') for kw in paper['keywords'] if kw.get('keyword')]

    # Extract concepts (OpenAlex's automatic tagging)
    concepts = []
    if paper.get('concepts'):
        # Only include concepts with score > 0.3
        concepts = [
            c.get('display_name', '')
            for c in paper['concepts']
            if c.get('score', 0) > 0.3
        ]

    # Extract DOI
    doi = paper.get('doi', '').replace('https://doi.org/', '') if paper.get('doi') else None

    # Extract open access status and PDF URL
    pdf_url = None
    is_oa = False
    if paper.get('open_access'):
        is_oa = paper['open_access'].get('is_oa', False)
        pdf_url = paper['open_access'].get('oa_url')

    # Extract primary location (journal/venue)
    venue = None
    if paper.get('primary_location'):
        source = paper['primary_location'].get('source')
        if source:
            venue = source.get('display_name')

    return {
        'openalex_id': paper.get('id', ''),
        'doi': doi,
        'title': paper.get('title', ''),
        'abstract': abstract,
        'keywords': keywords,
        'concepts': concepts,
        'publication_year': paper.get('publication_year'),
        'publication_date': paper.get('publication_date'),
        'venue': venue,
        'cited_by_count': paper.get('cited_by_count', 0),
        'is_open_access': is_oa,
        'pdf_url': pdf_url,
    }


def main():
    """
    Main workflow to fetch ~5000 papers from OpenAlex.
    """
    all_papers = []
    seen_ids = set()

    output_file = OUTPUT_DIR / "openalex_papers.json"
    progress_file = OUTPUT_DIR / "fetch_progress.json"

    # Load existing data if available
    if output_file.exists():
        print(f"Loading existing data from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            all_papers = existing_data.get('papers', [])
            seen_ids = {p['openalex_id'] for p in all_papers if p.get('openalex_id')}
        print(f"Loaded {len(all_papers)} existing papers.")

    print(f"\n{'='*60}")
    print(f"Fetching papers from OpenAlex (Target: {TARGET_PAPERS})")
    print(f"Date range: {START_YEAR}-{END_YEAR}")
    print(f"{'='*60}\n")

    for i, query_dict in enumerate(OPENALEX_QUERIES):
        if len(all_papers) >= TARGET_PAPERS:
            print(f"\n✓ Reached target of {TARGET_PAPERS} papers. Stopping.")
            break

        print(f"\n[Query {i+1}/{len(OPENALEX_QUERIES)}]")
        print(f"Search: {query_dict.get('title_abstract', 'N/A')}")
        if query_dict.get('keywords'):
            print(f"Keywords: {', '.join(query_dict['keywords'])}")

        # Build filter string
        filter_str = build_openalex_filter(query_dict, START_YEAR, END_YEAR)

        # Fetch papers
        remaining = TARGET_PAPERS - len(all_papers)
        fetch_limit = min(500, remaining * 2)  # Fetch extra to account for duplicates

        raw_papers = fetch_papers_openalex(filter_str, max_results=fetch_limit)

        # Process and deduplicate
        new_count = 0
        for paper in raw_papers:
            paper_id = paper.get('id', '')
            if paper_id and paper_id not in seen_ids:
                metadata = extract_paper_metadata(paper)

                # Only include papers with abstracts
                if metadata['abstract'] and len(metadata['abstract']) > 100:
                    all_papers.append(metadata)
                    seen_ids.add(paper_id)
                    new_count += 1

                    if len(all_papers) >= TARGET_PAPERS:
                        break

        print(f"  Added {new_count} new papers (total: {len(all_papers)})")

        # Save progress
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_papers': len(all_papers),
                    'date_fetched': datetime.now().isoformat(),
                    'date_range': f'{START_YEAR}-{END_YEAR}',
                    'source': 'OpenAlex API'
                },
                'papers': all_papers
            }, f, indent=2, ensure_ascii=False)

        print(f"  Progress saved to {output_file}")

    # Final save and summary
    print(f"\n{'='*60}")
    print(f"Fetch Complete!")
    print(f"{'='*60}")
    print(f"Total papers fetched: {len(all_papers)}")
    print(f"Saved to: {output_file}")

    # Generate summary statistics
    years = {}
    open_access_count = 0
    with_keywords_count = 0

    for paper in all_papers:
        year = paper.get('publication_year', 'Unknown')
        years[year] = years.get(year, 0) + 1
        if paper.get('is_open_access'):
            open_access_count += 1
        if paper.get('keywords'):
            with_keywords_count += 1

    print(f"\nStatistics:")
    print(f"  Open Access: {open_access_count} ({open_access_count/len(all_papers)*100:.1f}%)")
    print(f"  With Keywords: {with_keywords_count} ({with_keywords_count/len(all_papers)*100:.1f}%)")
    print(f"\nPapers by Year:")
    for year in sorted(years.keys()):
        print(f"  {year}: {years[year]}")

    # Save abstracts as individual text files for existing pipeline compatibility
    abstracts_dir = OUTPUT_DIR / "abstracts"
    abstracts_dir.mkdir(exist_ok=True)

    print(f"\nSaving abstracts as text files to {abstracts_dir}...")
    for i, paper in enumerate(all_papers):
        # Create filename from title
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in paper['title'])
        safe_title = safe_title[:100]  # Limit length
        filename = f"{i+1:04d}_{safe_title}.txt"

        # Combine title and abstract
        content = f"TITLE: {paper['title']}\n\n"
        if paper.get('keywords'):
            content += f"KEYWORDS: {', '.join(paper['keywords'])}\n\n"
        content += f"ABSTRACT:\n{paper['abstract']}\n"

        (abstracts_dir / filename).write_text(content, encoding='utf-8')

    print(f"✓ Saved {len(all_papers)} abstract files")
    print(f"\nNext steps:")
    print(f"  1. Review papers in: {output_file}")
    print(f"  2. Process abstracts with: python 4-extract_info.py")
    print(f"  3. Build KG with: python build_graph.py")


if __name__ == "__main__":
    main()
