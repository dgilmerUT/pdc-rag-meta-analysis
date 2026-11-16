from docling.document_converter import DocumentConverter
import csv
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
PARSED_DIR = Path("data/parsed")
PARSED_DIR.mkdir(exist_ok=True)

converter = DocumentConverter()

# duplicate copies removed by scripts/dedup.py that macOS would not let it move out of data/raw
removed = set()
for log in Path("outputs").glob("dedup_*/removed_copies.csv"):
    with open(log, encoding="utf-8") as f:
        removed |= {r["file"] for r in csv.DictReader(f)}

pdfs = [p for p in RAW_DIR.rglob("*.pdf")
        if p.relative_to(RAW_DIR).parts[0] != "_skipped" and str(p) not in removed]
print(f"Found {len(pdfs)} PDFs to parse")

for i, pdf_path in enumerate(pdfs):
    rel = pdf_path.relative_to(RAW_DIR)
    if rel.parts[0] == "pdfs":
        out_path = PARSED_DIR / (pdf_path.stem + ".json")
    else:
        sub_dir = PARSED_DIR / rel.parts[0]
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_path = sub_dir / (pdf_path.stem + ".json")

    if out_path.exists():
        print(f"  Skipping (already done): {pdf_path.name}")
        continue

    print(f"  Parsing {i+1} of {len(pdfs)}: {pdf_path.name}")
    try:
        result = converter.convert(str(pdf_path))
        doc = result.document.export_to_dict()
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"    Done")
    except Exception as e:
        print(f"    Failed: {e}")

print(f"\nFinished. Parsed files are in: {PARSED_DIR}")
