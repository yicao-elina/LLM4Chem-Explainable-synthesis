# clean_txt.py
import re
from pathlib import Path


# ---------- Configurable section name catalogs ----------
# Section headings we will remove as entire sections (from heading to next heading)
SECTION_HEADINGS_REMOVE = [
    r"Acknowledg(?:e)?ments?",
    r"Funding",
    r"Author(?:s)?\s+Contributions?",
    r"Contributions",
    r"Competing\s+Interests?",
    r"Conflict(?:s)?\s+of\s+Interest",
    r"Ethics(?:\s+Statement)?",
    r"Data\s+Availability(?:\s+Statement)?",
    r"Availability\s+of\s+Data(?:\s+and\s+Materials)?",
    r"Publisher[’']s\s+Note",
    r"Additional\s+Information",
    r"Notes",        # journal "Notes" sections (non-technical)
    r"Footnotes",
]

# Section headings that define boundaries (used to detect the end of a removed section)
SECTION_HEADINGS_BOUNDARIES = [
    # Keepable scientific sections (as boundaries)
    r"Abstract",
    r"Introduction",
    r"Background",
    r"Related\s+Work",
    r"Materials\s+and\s+Methods",
    r"Methods?",
    r"Methodology",
    r"Experimental(?:\s+Section)?",
    r"Experimental\s+Details",
    r"Computational\s+Methods?",
    r"Theory\s+and\s+Methods?",
    r"Results(?:\s+and\s+Discussion)?",
    r"Discussion",
    r"Conclusions?",
    r"Outlook",
    r"Future\s+Work",
    r"Appendix(?:es)?",
    r"Supplementary\s+Information",
    # Also include removable ones, so we can stop at them
    *SECTION_HEADINGS_REMOVE,
    # Terminal sections (we’ll remove them separately, but include as boundaries)
    r"References",
    r"Bibliograph(?:y|ies)",
]

# Single-line patterns to drop (NOT entire sections)
LINE_LEVEL_REMOVE_PATTERNS = [
    r"\bCorresponding\s+Author[s]?:",
    r"\bCorrespondence\s+to:?",
    r"\bTo\s+whom\s+correspondence\s+should\s+be\s+addressed\b",
    r"\bE[- ]?mail:?",
    r"\bEmail:?",
    r"\bORCID\b",
    r"©\s?\d{4}",
    r"\bCreative\s+Commons\b",
    r"\bThis\s+article\s+is\s+licensed\b",
    r"\bdoi:\s*10\.\d{4,9}/\S+",
    r"https?://doi\.org/\S+",
    r"\bReceived\b.*\bAccepted\b.*\bPublished\b",  # timeline line
    r"\bPublished\s+online\b",
    r"\bAll\s+rights\s+reserved\b",
]

# Heuristic affiliation lines (only applied near the top of the file)
AFFILIATION_LINE_PATTERNS = [
    r"\bDepartment\b",
    r"\bSchool\b",
    r"\bCollege\b",
    r"\bFaculty\b",
    r"\bInstitute\b",
    r"\bLaborator(y|ies|y)\b|\bLab\.?\b",
    r"\bCenter\b|\bCentre\b",
    r"\bUniversity\b|\bUniv\.?\b",
    r"\bAcademy\b",
    r"\bHospital\b|\bClinic\b",
    r"\bNational\b|\bState\s+Key\b",
    r"\bCampus\b|\bRoom\b|\bBuilding\b",
    r"\bStreet\b|\bSt\.?\b|\bAvenue\b|\bAve\.?\b|\bRoad\b|\bRd\.?\b",
    r"\b[A-Z][a-z]+,\s*[A-Z]{2}\s*\d{5}\b",  # City, ST 12345
    r"\bUSA\b|\bUnited\s*States\b|\bUK\b|\bChina\b|\bGermany\b|\bItaly\b|\bFrance\b|\bJapan\b",
]

# ---------- Helpers ----------
def _compile_heading_union(heading_list):
    return r"(?:%s)" % "|".join(heading_list)

BOUNDARY_REGEX = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s*[\)\.]?\s*)?"  # optional numbered headings like "2." or "3.1)"
    + _compile_heading_union(SECTION_HEADINGS_BOUNDARIES)
    + r"\b.*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

def remove_references_to_end(text: str) -> str:
    """Remove everything from 'References' or 'Bibliography' heading to the end."""
    pattern = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\s*[\)\.]?\s*)?(References|Bibliograph(?:y|ies))\b.*\Z",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return re.sub(pattern, "", text)

def remove_unrelated_sections(text: str) -> str:
    """
    Remove specific sections (acknowledgements, funding, etc.) from their heading
    up to (but not including) the next recognized section heading.
    """
    for heading in SECTION_HEADINGS_REMOVE:
        # Match the start heading
        start_pat = re.compile(
            r"^\s*(?:\d+(?:\.\d+)*\s*[\)\.]?\s*)?(?:" + heading + r")\b.*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        while True:
            m = start_pat.search(text)
            if not m:
                break
            start = m.start()

            # Find the next boundary heading after this start
            next_boundary = None
            for bm in BOUNDARY_REGEX.finditer(text, m.end()):
                next_boundary = bm.start()
                break

            # Cut the section
            if next_boundary is None:
                # To EOF
                text = text[:start]
            else:
                text = text[:start] + text[next_boundary:]
    return text

def remove_line_level_noise(text: str) -> str:
    """Remove only the lines that contain clearly irrelevant info."""
    lines = text.splitlines()
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in LINE_LEVEL_REMOVE_PATTERNS]
    cleaned_lines = []
    for i, line in enumerate(lines):
        if any(p.search(line) for p in compiled):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def strip_top_affiliations(text: str, max_lines_scan: int = 80) -> str:
    """
    Heuristically remove obvious affiliation/address lines near the top
    without touching the scientific content.
    """
    lines = text.splitlines()
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in AFFILIATION_LINE_PATTERNS]

    top = lines[: max_lines_scan if len(lines) > max_lines_scan else len(lines)]
    kept_top = []
    for line in top:
        # Drop if it looks like a pure affiliation/address line (short-ish and dominated by affiliation cues)
        if len(line.strip()) <= 200 and any(p.search(line) for p in compiled):
            continue
        kept_top.append(line)

    new_text = "\n".join(kept_top + lines[len(top):])
    return new_text

def normalize_whitespace_soft(text: str) -> str:
    """
    Soft normalization:
    - collapse >2 blank lines to exactly 1
    - strip trailing spaces
    (Do NOT collapse all whitespace; we must preserve newlines for headings.)
    """
    # Remove trailing spaces
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- Public API ----------
def clean_text(text: str) -> str:
    """
    Keep scientific content. Remove:
      - References/Bibliography (entire tail)
      - Acknowledgements/Funding/etc. sections (only those sections)
      - Single-line noise (emails, ORCID, DOI/license lines, publisher notes)
      - Obvious affiliation lines near the top
    """
    # 0) Preserve structure: DO NOT collapse newlines
    t = text

    # 1) Strip obvious affiliation/header noise near the very top (safe heuristic)
    t = strip_top_affiliations(t)

    # 2) Remove single-line rubbish (safe)
    t = remove_line_level_noise(t)

    # 3) Remove small unrelated sections but keep boundaries intact
    t = remove_unrelated_sections(t)

    # 4) Remove References/Bibliography to end
    t = remove_references_to_end(t)

    # 5) Soft whitespace normalization
    t = normalize_whitespace_soft(t)

    return t


def clean_txt_folder(txt_dir="papers/txt", cleaned_dir="papers/cleaned"):
    """
    Process all .txt files under txt_dir, write cleaned versions to cleaned_dir.
    """
    txt_dir = Path(txt_dir)
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        print("No .txt files found.")
        return

    for txt_file in txt_files:
        raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw_text)
        out_file = cleaned_dir / txt_file.name
        out_file.write_text(cleaned, encoding="utf-8")
        print(f"✓ Cleaned {txt_file.name} → {out_file}")

    print(f"\nDone. Cleaned {len(txt_files)} files into {cleaned_dir}")


if __name__ == "__main__":
    clean_txt_folder()
