import json
from pathlib import Path

PARSED_DIR = Path("data/parsed")
OUTPUT_FILE = Path("data/chunks.json")

SOURCE_TYPE_MAP = {
    "conventional_ceramics": "conventional_ceramic",
    "alternative_routes": "alternative_route",
    "applications": "application",
}


def classify_source_type(json_path):
    """Determine source_type from the parsed file's directory location."""
    rel = json_path.relative_to(PARSED_DIR)
    parts = rel.parts
    if len(parts) > 1:
        for dir_key, source_type in SOURCE_TYPE_MAP.items():
            if parts[0] == dir_key:
                return source_type
    return "pdc"


def chunk_text(text, source, section="unknown", source_type="pdc",
               chunk_size=250, overlap=50):
    chunks = []
    words = text.split()
    if not words:
        return chunks
    if len(words) <= chunk_size:
        chunks.append({
            "text": text,
            "source": source,
            "section": section,
            "type": "text",
            "source_type": source_type,
        })
    else:
        for i in range(0, len(words), chunk_size - overlap):
            segment = " ".join(words[i:i + chunk_size])
            if len(segment.strip()) > 80:
                chunks.append({
                    "text": segment,
                    "source": source,
                    "section": section,
                    "type": "text",
                    "source_type": source_type,
                })
    return chunks


def chunk_document(doc, source_name, source_type="pdc"):
    chunks = []
    texts = doc.get("texts", [])
    for item in texts:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "").strip()
        if not text or len(text) < 40:
            continue
        label = item.get("label", "unknown")
        section = str(label).lower() if label else "unknown"
        if section in ["page_header", "page_footer", "footnote"]:
            continue
        chunks.extend(chunk_text(text, source_name, section, source_type))
    tables = doc.get("tables", [])
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_text = ""
        data = table.get("data", {})
        if isinstance(data, dict):
            grid = data.get("grid", [])
            for row in grid:
                row_texts = []
                for cell in row:
                    cell_text = cell.get("text", "").strip()
                    if cell_text:
                        row_texts.append(cell_text)
                if row_texts:
                    table_text += " | ".join(row_texts) + "\n"
        if table_text.strip():
            chunks.append({
                "text": table_text.strip(),
                "source": source_name,
                "section": "table",
                "type": "table",
                "source_type": source_type,
            })
    return chunks


json_files = list(PARSED_DIR.rglob("*.json"))
print(f"Found {len(json_files)} parsed files to chunk")

if not json_files:
    print("No parsed files found. Make sure parse_pdfs.py ran successfully.")
    exit()

all_chunks = []
type_counts = {}

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except Exception as e:
            print(f"  Could not read {json_path.name}: {e}")
            continue
    source_type = classify_source_type(json_path)
    chunks = chunk_document(doc, json_path.stem, source_type)
    all_chunks.extend(chunks)
    type_counts[source_type] = type_counts.get(source_type, 0) + 1
    print(f"  [{source_type}] {json_path.stem}: {len(chunks)} chunks")

print(f"\nTotal chunks across all papers: {len(all_chunks)}")
print(f"\nCorpus breakdown by source type:")
for st, count in sorted(type_counts.items()):
    print(f"  {st}: {count} papers")

if all_chunks:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"\nSaved to: {OUTPUT_FILE}")
else:
    print("No chunks were created. Check that your parsed files contain text.")
