"""
Create sample test data for pipeline testing.
This bypasses the slow OpenAlex API fetch for initial testing.
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("papers/openalex_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABSTRACTS_DIR = OUTPUT_DIR / "abstracts"
ABSTRACTS_DIR.mkdir(exist_ok=True)

# Sample abstracts from real 2D materials doping research
SAMPLE_ABSTRACTS = [
    {
        "title": "Site-selective doping in MoS2 monolayers by chemical vapor deposition",
        "abstract": """We demonstrate site-selective doping of MoS2 monolayers using Nb substitution via chemical vapor deposition at 750°C.
Scanning transmission electron microscopy (STEM) and X-ray photoelectron spectroscopy (XPS) confirm that Nb atoms preferentially substitute Mo atoms at a concentration of 2 at%, forming substitutional defects.
The increased temperature from 650°C to 750°C enhances the substitutional fraction by promoting Mo vacancy formation.
Hall measurements reveal n-type conductivity with carrier concentration increasing from 1×10^18 to 5×10^18 cm^-3.
This controlled substitutional doping enables precise tuning of carrier mobility, which increased from 15 to 45 cm2/V·s.""",
        "keywords": ["MoS2", "CVD", "site-selective doping", "Nb substitution"]
    },
    {
        "title": "DFT study of phosphorus doping in graphene: substitutional vs pyridinic sites",
        "abstract": """First-principles density functional theory (DFT) calculations were performed to investigate phosphorus doping in graphene.
We find that P atoms preferentially occupy substitutional carbon sites at low doping concentrations (<1 at%).
The formation energy for substitutional doping is 1.2 eV lower than pyridinic configurations.
Band structure calculations show that substitutional P doping induces n-type behavior by shifting the Fermi level by 0.4 eV.
The calculated carrier concentration increases with P concentration, reaching 3×10^13 cm^-2 at 2 at% doping.""",
        "keywords": ["graphene", "DFT", "phosphorus doping", "substitutional"]
    },
    {
        "title": "Electrochemical intercalation of lithium in WS2: Site selectivity and structural changes",
        "abstract": """Galvanostatic electrochemical intercalation of Li+ ions into WS2 was performed at room temperature.
X-ray diffraction shows c-axis expansion from 12.3 Å to 15.8 Å, confirming Li intercalation in the van der Waals gap.
The intercalation occurs exclusively in the vdW gap, with no evidence of substitutional or surface adsorption.
Time-of-flight secondary ion mass spectrometry (ToF-SIMS) depth profiles confirm uniform Li distribution throughout the layers.
Electrical conductivity increased from 10^-3 to 10^2 S/cm due to increased carrier concentration.""",
        "keywords": ["WS2", "lithium intercalation", "electrochemical", "van der Waals gap"]
    },
    {
        "title": "Nitrogen doping in hBN by ion implantation and thermal annealing",
        "abstract": """Hexagonal boron nitride (hBN) monolayers were doped with nitrogen using ion implantation followed by thermal annealing at 900°C.
Atomic resolution STEM imaging reveals N atoms substitute at both B and N sites, with preference for B-site substitution (60%).
Annealing at higher temperatures (900°C vs 600°C) increases the fraction of substitutional defects by promoting defect healing and reducing interstitial incorporation.
Photoluminescence spectroscopy shows the optical band gap decreases from 6.0 eV to 5.2 eV with 5% N doping.""",
        "keywords": ["hBN", "nitrogen doping", "ion implantation", "band gap"]
    },
    {
        "title": "Molecular dynamics simulation of copper intercalation in Bi2Te3",
        "abstract": """Molecular dynamics (MD) simulations at 500K were employed to study Cu intercalation in Bi2Te3 quintuple layers.
The simulations predict that Cu atoms preferentially occupy the van der Waals gap between quintuple layers at concentrations below 5 at%.
The interlayer spacing increases linearly with Cu concentration, from 2.4 Å to 3.2 Å at 5 at% doping.
MD trajectories show surface adsorption is energetically unfavorable (ΔE = +0.8 eV) compared to vdW gap intercalation.
Electronic structure calculations indicate p-type conductivity with hole concentration of 2×10^19 cm^-3.""",
        "keywords": ["Bi2Te3", "copper intercalation", "MD simulation", "quintuple layer"]
    },
]

# Create metadata file
metadata = {
    "metadata": {
        "total_papers": len(SAMPLE_ABSTRACTS),
        "source": "Manual test data",
        "note": "Sample abstracts for pipeline testing"
    },
    "papers": []
}

# Save abstracts as text files and add to metadata
for i, paper in enumerate(SAMPLE_ABSTRACTS, 1):
    # Create filename
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in paper['title'])
    safe_title = safe_title[:80]
    filename = f"{i:03d}_{safe_title}.txt"

    # Create content
    content = f"TITLE: {paper['title']}\n\n"
    if paper.get('keywords'):
        content += f"KEYWORDS: {', '.join(paper['keywords'])}\n\n"
    content += f"ABSTRACT:\n{paper['abstract']}\n"

    # Save file
    (ABSTRACTS_DIR / filename).write_text(content, encoding='utf-8')

    # Add to metadata
    metadata["papers"].append({
        "openalex_id": f"test_{i}",
        "title": paper['title'],
        "abstract": paper['abstract'],
        "keywords": paper.get('keywords', []),
        "publication_year": 2024,
    })

# Save metadata
with open(OUTPUT_DIR / "openalex_test_papers.json", 'w', encoding='utf-8') as f:
    json.dump(metadata, indent=2, fp=f)

print(f"✓ Created {len(SAMPLE_ABSTRACTS)} test abstracts in {ABSTRACTS_DIR}")
print(f"✓ Saved metadata to {OUTPUT_DIR}/openalex_test_papers.json")
print("\nTest data created successfully!")
print("Next: python3 extract_robust.py")
