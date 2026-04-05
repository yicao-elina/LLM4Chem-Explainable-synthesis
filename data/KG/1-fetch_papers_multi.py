import requests
import os
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import random

# --- Configuration ---
SAVE_DIR = Path("papers/raw")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_PAPER_COUNT = 200 # Set your desired number of papers

# --- Query Design ---
S2_QUERIES = [
    # Core site-selectivity in vdW 2D materials
    '("site-selective" OR "site selectivity" OR "site preference") AND ("van der Waals" OR "vdW" OR "layered") AND (doping OR intercalation) AND ("2D" OR graphene OR hBN OR "transition metal dichalcogenide" OR TMD) AND (experimental OR synthesis OR fabrication) AND NOT (review)',
    # Explicit vdW-gap / interlayer location
    '("vdW gap" OR "van der Waals gap" OR interlayer OR "between layers" OR "quintuple layer" OR "QL") AND (doping OR intercalation) AND ("MoS2" OR "WS2" OR "WSe2" OR "MoSe2" OR "MoTe2" OR "WTe2" OR "Bi2Se3" OR "Bi2Te3" OR "Sb2Te3" OR "TaS2" OR "TiS2" OR "NbSe2") AND (experimental OR synthesis) AND NOT (review)',
    # Substitutional vs intercalation vs surface (comparative)
    '(substitutional OR interstitial OR intercalation OR "surface adsorption") AND (selectivity OR preference OR competition) AND ("2D" OR TMD OR graphene OR hBN OR "layered chalcogenide") AND (doping) AND (experimental OR synthesis) AND NOT (review)',
    # Method-focused: electrochemical intercalation
    '("electrochemical intercalation" OR "galvanostatic intercalation" OR "potentiostatic intercalation") AND ("2D" OR layered OR vdW) AND (dop* OR intercalant) AND (site OR "vdW gap" OR interlayer) AND (STEM OR XPS OR XRD OR SIMS OR EELS OR "Hall")',
    # Method-focused: vapor/solid-state post-anneal
    '("vapor transport" OR "sealed ampoule" OR "chemical vapor transport" OR CVT OR "post-anneal" OR "solid-state anneal") AND (doping OR intercalation) AND ("2D" OR layered OR vdW) AND (site OR substitutional OR interlayer) AND (XPS OR STEM OR HAADF OR "Rietveld" OR "XRD (00l)")',
    # Method-focused: MBE/CVD growth with dopants
    '(MBE OR "molecular beam epitaxy" OR CVD OR "chemical vapor deposition") AND (doping OR delta-doping OR "co-deposition") AND (TMD OR graphene OR hBN OR "layered chalcogenide") AND (site-select* OR substitutional OR intercalation) AND (experimental) AND NOT (review)',
    # Method-focused: plasma/ion implantation with site outcomes
    '("ion implantation" OR plasma OR "remote plasma") AND ("2D" OR vdW OR TMD) AND doping AND (site OR substitutional OR interstitial) AND (anneal OR recrystallization OR healing) AND (TEM OR STEM OR XPS OR Raman)',
    # Alkali/metal intercalation (classic vdW gap occupation)
    '(Li OR Na OR K OR Rb OR Cs OR Cu OR Ag OR Au) AND intercalation AND ("NbSe2" OR "TaS2" OR "TiS2" OR "VSe2" OR graphite OR graphene OR "Bi2Se3" OR "Bi2Te3") AND (site OR "vdW gap" OR staging OR interlayer) AND (experimental) AND NOT (review)',
    # Dopant-site proof via direct microscopy/spectroscopy
    '(STEM OR HAADF OR "atomic-resolution" OR EELS OR "ToF-SIMS" OR SIMS OR "depth profile") AND (doping OR intercalation) AND ("2D" OR vdW OR TMD) AND (site OR interlayer OR substitutional)',
    # Thermodynamic selectivity (formation/segregation energies but with experiments)
    '("formation energy" OR "segregation energy" OR "binding energy") AND (doping OR adatom OR intercalation) AND (TMD OR graphene OR hBN OR "layered chalcogenide") AND (experiment OR synthesis OR "grown") AND NOT (review)',
    # Electric-field/temperature/cooling-rate effects (causal levers)
    '(temperature OR "cooling rate" OR "quench" OR "slow cooling" OR "electric field") AND (controls OR tunes OR modulates) AND (site-select* OR intercalation OR substitutional) AND ("2D" OR vdW) AND (experimental)',
    # Specific outcome keywords that often imply site information
    '("vdW intercalation" OR "gap intercalation" OR "stage-1" OR "stage 1") AND ("2D" OR layered OR graphite OR graphene OR TMD) AND (XRD OR "(00l)" OR "c-axis expansion" OR "interlayer spacing") AND (experimental)',
    # Topical: p-type/n-type control tied to dopant site
    '("p-type" OR "n-type") AND (control OR tuning) AND (doping OR intercalation) AND ("2D" OR TMD OR graphene) AND (site OR interlayer OR substitutional) AND (Hall OR Seebeck OR transport)',
    # Bi2Se3/Bi2Te3/Sb2Te3 quintuple-layer site selectivity (vdW gaps, QL)
    '("Bi2Se3" OR "Bi2Te3" OR "Sb2Te3") AND (doping OR intercalation) AND ("quintuple layer" OR "vdW gap" OR interlayer) AND (site-select* OR preference) AND (XPS OR STEM OR XRD) AND NOT (review)',
    # TMD catalog query (broad net, experimental filters)
    '("MoS2" OR "WS2" OR "WSe2" OR "MoSe2" OR "MoTe2" OR "WTe2") AND (doping OR intercalation) AND (site OR substitutional OR interlayer OR "surface adsorption") AND (experimental OR synthesis OR "grown by") AND NOT (review)',
    # Surface vs bulk selectivity explicitly
    '("surface adsorption" OR "surface functionalization") AND (vs OR compared OR versus) AND (intercalation OR substitutional) AND ("2D" OR vdW) AND (experimental OR TEM OR XPS)',
    # Molten salt / solvothermal routes that often yield intercalation
    '("molten salt" OR solvothermal OR "ionothermal" OR "liquid metal") AND (intercalation OR doping) AND ("2D" OR layered OR vdW) AND (site OR interlayer OR staging) AND (experimental)',
    # Depth-resolved evidence of site (keeps data-rich papers)
    '("depth-resolved" OR "angle-resolved XPS" OR "ARXPS" OR "cross-sectional TEM") AND (doping OR intercalation) AND (TMD OR graphene OR "layered chalcogenide")',
    # Exclude reviews but keep SI-rich experimentals
    '("Supporting Information" OR "Supplementary Information") AND (doping OR intercalation) AND ("2D" OR vdW) AND (site OR interlayer OR substitutional) AND NOT (review)'
]

ARXIV_QUERIES = [
    'ti:"site selective doping" OR ti:"site preference" AND (cat:cond-mat.mes-hall OR cat:cond-mat.mtrl-sci)',
    'abs:(graphene OR MoS2 OR WSe2) AND ti:(dopant OR adatom) AND abs:"binding energy"',
    'all:(DFT OR "first principles") AND ti:(doping AND "2D material")',
    'ti:"defect engineering" AND abs:("transition metal dichalcogenide" OR TMD)',
        # Core site-selectivity in vdW 2D materials
    '("site-selective" OR "site selectivity" OR "site preference") AND ("van der Waals" OR "vdW" OR "layered") AND (doping OR intercalation) AND ("2D" OR graphene OR hBN OR "transition metal dichalcogenide" OR TMD) AND (experimental OR synthesis OR fabrication) AND NOT (review)',
    # Explicit vdW-gap / interlayer location
    '("vdW gap" OR "van der Waals gap" OR interlayer OR "between layers" OR "quintuple layer" OR "QL") AND (doping OR intercalation) AND ("MoS2" OR "WS2" OR "WSe2" OR "MoSe2" OR "MoTe2" OR "WTe2" OR "Bi2Se3" OR "Bi2Te3" OR "Sb2Te3" OR "TaS2" OR "TiS2" OR "NbSe2") AND (experimental OR synthesis) AND NOT (review)',
    # Substitutional vs intercalation vs surface (comparative)
    '(substitutional OR interstitial OR intercalation OR "surface adsorption") AND (selectivity OR preference OR competition) AND ("2D" OR TMD OR graphene OR hBN OR "layered chalcogenide") AND (doping) AND (experimental OR synthesis) AND NOT (review)',
    # Method-focused: electrochemical intercalation
    '("electrochemical intercalation" OR "galvanostatic intercalation" OR "potentiostatic intercalation") AND ("2D" OR layered OR vdW) AND (dop* OR intercalant) AND (site OR "vdW gap" OR interlayer) AND (STEM OR XPS OR XRD OR SIMS OR EELS OR "Hall")',
    # Method-focused: vapor/solid-state post-anneal
    '("vapor transport" OR "sealed ampoule" OR "chemical vapor transport" OR CVT OR "post-anneal" OR "solid-state anneal") AND (doping OR intercalation) AND ("2D" OR layered OR vdW) AND (site OR substitutional OR interlayer) AND (XPS OR STEM OR HAADF OR "Rietveld" OR "XRD (00l)")',
    # Method-focused: MBE/CVD growth with dopants
    '(MBE OR "molecular beam epitaxy" OR CVD OR "chemical vapor deposition") AND (doping OR delta-doping OR "co-deposition") AND (TMD OR graphene OR hBN OR "layered chalcogenide") AND (site-select* OR substitutional OR intercalation) AND (experimental) AND NOT (review)',
    # Method-focused: plasma/ion implantation with site outcomes
    '("ion implantation" OR plasma OR "remote plasma") AND ("2D" OR vdW OR TMD) AND doping AND (site OR substitutional OR interstitial) AND (anneal OR recrystallization OR healing) AND (TEM OR STEM OR XPS OR Raman)',
    # Alkali/metal intercalation (classic vdW gap occupation)
    '(Li OR Na OR K OR Rb OR Cs OR Cu OR Ag OR Au) AND intercalation AND ("NbSe2" OR "TaS2" OR "TiS2" OR "VSe2" OR graphite OR graphene OR "Bi2Se3" OR "Bi2Te3") AND (site OR "vdW gap" OR staging OR interlayer) AND (experimental) AND NOT (review)',
    # Dopant-site proof via direct microscopy/spectroscopy
    '(STEM OR HAADF OR "atomic-resolution" OR EELS OR "ToF-SIMS" OR SIMS OR "depth profile") AND (doping OR intercalation) AND ("2D" OR vdW OR TMD) AND (site OR interlayer OR substitutional)',
    # Thermodynamic selectivity (formation/segregation energies but with experiments)
    '("formation energy" OR "segregation energy" OR "binding energy") AND (doping OR adatom OR intercalation) AND (TMD OR graphene OR hBN OR "layered chalcogenide") AND (experiment OR synthesis OR "grown") AND NOT (review)',
    # Electric-field/temperature/cooling-rate effects (causal levers)
    '(temperature OR "cooling rate" OR "quench" OR "slow cooling" OR "electric field") AND (controls OR tunes OR modulates) AND (site-select* OR intercalation OR substitutional) AND ("2D" OR vdW) AND (experimental)',
    # Specific outcome keywords that often imply site information
    '("vdW intercalation" OR "gap intercalation" OR "stage-1" OR "stage 1") AND ("2D" OR layered OR graphite OR graphene OR TMD) AND (XRD OR "(00l)" OR "c-axis expansion" OR "interlayer spacing") AND (experimental)',
    # Topical: p-type/n-type control tied to dopant site
    '("p-type" OR "n-type") AND (control OR tuning) AND (doping OR intercalation) AND ("2D" OR TMD OR graphene) AND (site OR interlayer OR substitutional) AND (Hall OR Seebeck OR transport)',
    # Bi2Se3/Bi2Te3/Sb2Te3 quintuple-layer site selectivity (vdW gaps, QL)
    '("Bi2Se3" OR "Bi2Te3" OR "Sb2Te3") AND (doping OR intercalation) AND ("quintuple layer" OR "vdW gap" OR interlayer) AND (site-select* OR preference) AND (XPS OR STEM OR XRD) AND NOT (review)',
    # TMD catalog query (broad net, experimental filters)
    '("MoS2" OR "WS2" OR "WSe2" OR "MoSe2" OR "MoTe2" OR "WTe2") AND (doping OR intercalation) AND (site OR substitutional OR interlayer OR "surface adsorption") AND (experimental OR synthesis OR "grown by") AND NOT (review)',
    # Surface vs bulk selectivity explicitly
    '("surface adsorption" OR "surface functionalization") AND (vs OR compared OR versus) AND (intercalation OR substitutional) AND ("2D" OR vdW) AND (experimental OR TEM OR XPS)',
    # Molten salt / solvothermal routes that often yield intercalation
    '("molten salt" OR solvothermal OR "ionothermal" OR "liquid metal") AND (intercalation OR doping) AND ("2D" OR layered OR vdW) AND (site OR interlayer OR staging) AND (experimental)',
    # Depth-resolved evidence of site (keeps data-rich papers)
    '("depth-resolved" OR "angle-resolved XPS" OR "ARXPS" OR "cross-sectional TEM") AND (doping OR intercalation) AND (TMD OR graphene OR "layered chalcogenide")',
    # Exclude reviews but keep SI-rich experimentals
    '("Supporting Information" OR "Supplementary Information") AND (doping OR intercalation) AND ("2D" OR vdW) AND (site OR interlayer OR substitutional) AND NOT (review)'
]

def sanitize_filename(name):
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:100]

# -------------------
# Semantic Scholar
# -------------------
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,year,openAccessPdf"

def search_semanticscholar(query, limit=100, batch_size=50):
    results = []
    offset = 0
    while len(results) < limit:
        params = {
            "query": query, "offset": offset,
            "limit": min(batch_size, limit - len(results)),
            "fields": S2_FIELDS
        }
        
        # --- FIX: Exponential Backoff Logic ---
        max_retries = 5
        base_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                r = requests.get(S2_API, params=params, timeout=20)
                if r.status_code == 429:
                    # Explicitly handle rate-limiting error
                    print("Rate limit hit. Waiting to retry...")
                    r.raise_for_status() # This will trigger the except block
                r.raise_for_status() # Handle other errors like 500, 404 etc.
                
                # If successful, break the retry loop
                break 
            except requests.exceptions.RequestException as e:
                print(f"API request failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"Waiting for {delay:.2f} seconds before retrying.")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Skipping this batch.")
                    return [] # Return empty list for this failed batch
        # --- End of Fix ---

        data = r.json()
        papers = data.get("data", [])
        if not papers:
            break
        
        results.extend(papers)
        offset += len(papers)
        
        if 'next' not in data or data['next'] == 0:
            break

        time.sleep(1.5) # Increased base sleep time to be more respectful
        
    return results[:limit]

# -------------------
# arXiv (No changes needed, but good practice to add similar logic if it fails)
# -------------------
ARXIV_API = "http://export.arxiv.org/api/query"

def search_arxiv(query, max_results=100):
    params = {
        "search_query": query, "start": 0, "max_results": max_results
    }
    try:
        r = requests.get(ARXIV_API, params=params, timeout=15)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"arXiv API error: {e}")
        return []

    root = ET.fromstring(r.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        paper_id = entry.find("atom:id", ns).text
        title = entry.find("atom:title", ns).text.strip()
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]
                break
        if pdf_url:
            papers.append({"paperId": paper_id, "title": title, "pdf_url": pdf_url})
    return papers

# -------------------
# PDF Downloader
# -------------------
def download_pdf(url, out_path):
    try:
        if out_path.exists():
            print(f"~ Already exists: {out_path.name}")
            return False

        r = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code == 200 and r.headers.get("content-type","").lower().startswith("application/pdf"):
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"✓ Downloaded: {out_path.name}")
            return True
        else:
            print(f"✗ Not a valid PDF ({r.status_code}): {url}")
    except Exception as e:
        print(f"✗ Download failed for {url}: {e}")
    return False

# -------------------
# Main workflow
# -------------------
def main():
    downloaded_count = 0
    downloaded_paper_ids = set()

    # --- Semantic Scholar Loop ---
    print(f"\n--- Searching Semantic Scholar (Target: {TARGET_PAPER_COUNT} papers) ---")
    for i, query in enumerate(S2_QUERIES):
        if downloaded_count >= TARGET_PAPER_COUNT:
            break
        print(f"\nExecuting S2 Query {i+1}/{len(S2_QUERIES)}: '{query[:80]}...'")
        
        remaining_needed = TARGET_PAPER_COUNT - downloaded_count
        # Fetch more to account for duplicates and already existing files
        s2_papers = search_semanticscholar(query, limit=remaining_needed * 2) 

        for p in s2_papers:
            if downloaded_count >= TARGET_PAPER_COUNT:
                break
            
            paper_id = p.get("paperId")
            if not paper_id or paper_id in downloaded_paper_ids:
                continue

            pdf_info = p.get("openAccessPdf")
            if pdf_info and pdf_info.get("url"):
                fname = sanitize_filename(p["title"]) + ".pdf"
                out_path = SAVE_DIR / fname
                
                if download_pdf(pdf_info["url"], out_path):
                    downloaded_count += 1
                    downloaded_paper_ids.add(paper_id)
                    
    # --- arXiv Loop ---
    print(f"\n--- Searching arXiv (Target: {TARGET_PAPER_COUNT} papers) ---")
    for i, query in enumerate(ARXIV_QUERIES):
        if downloaded_count >= TARGET_PAPER_COUNT:
            break
        print(f"\nExecuting arXiv Query {i+1}/{len(ARXIV_QUERIES)}: '{query[:80]}...'")
        
        remaining_needed = TARGET_PAPER_COUNT - downloaded_count
        arxiv_papers = search_arxiv(query, max_results=remaining_needed * 2)

        for p in arxiv_papers:
            if downloaded_count >= TARGET_PAPER_COUNT:
                break

            paper_id = p.get("paperId")
            if not paper_id or paper_id in downloaded_paper_ids:
                continue

            fname = sanitize_filename(p["title"]) + ".pdf"
            out_path = SAVE_DIR / fname

            if download_pdf(p["pdf_url"], out_path):
                downloaded_count += 1
                downloaded_paper_ids.add(paper_id)

    print(f"\n--- Workflow Complete ---")
    print(f"Successfully downloaded {downloaded_count} new PDFs into '{SAVE_DIR}'")

if __name__ == "__main__":
    main()
