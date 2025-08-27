import fitz  # PyMuPDF
from pathlib import Path

def pdf_to_txt(pdf_path: Path, txt_path: Path):
    """Extract text from a PDF file and save as .txt."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
        txt_path.write_text(text, encoding="utf-8")
        print(f"✓ {pdf_path.name} → {txt_path.name}")
    except Exception as e:
        print(f"✗ Failed {pdf_path.name}: {e}")

def convert_papers(pdf_dir="papers/raw", txt_dir="papers/txt"):
    pdf_dir, txt_dir = Path(pdf_dir), Path(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in", pdf_dir)
        return

    for pdf in pdf_files:
        txt_file = txt_dir / (pdf.stem + ".txt")
        pdf_to_txt(pdf, txt_file)

    print(f"\nDone. Extracted {len(pdf_files)} PDFs into {txt_dir}")

if __name__ == "__main__":
    convert_papers()
