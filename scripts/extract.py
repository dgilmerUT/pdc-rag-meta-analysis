import json
import os
import sys
import chromadb
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# ── Setup ────────────────────────────────────────────────────────────────────

load_dotenv()

CHROMA_DIR = "vector_store/chroma_db"
OUTPUT_FILE = Path("outputs/descriptors.json")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("preceramic_polymers")

claude = Anthropic()

# ── Schema ───────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert in preceramic polymer chemistry. Extract the following
descriptors from the provided paper text as a JSON object with EXACTLY these fields.

For any field not clearly stated in the text, use null. Do not guess or infer values that
are not supported by the text. If a range is given, use the midpoint and note the range
in the "notes" field.

{
  "polymer_type": "string - e.g. polysilazane, polycarbosilane, polysiloxane, polyborosilazane",
  "polymer_name": "string - specific name/designation if given, e.g. HTT 1800, KiON Ceraset",

  "elemental_composition": {
    "Si_atomic_pct": "number or null",
    "C_atomic_pct": "number or null",
    "N_atomic_pct": "number or null",
    "O_atomic_pct": "number or null",
    "H_atomic_pct": "number or null",
    "other_elements": "string or null - any other elements with values"
  },

  "bonding_structure": {
    "Si_H_present": "true/false/null",
    "Si_CH3_present": "true/false/null",
    "N_H_present": "true/false/null",
    "Si_N_present": "true/false/null",
    "Si_O_present": "true/false/null",
    "Si_C_present": "true/false/null",
    "other_bonds": "string or null - any other notable bonds described"
  },

  "functional_groups": {
    "vinyl": "true/false/null",
    "methyl": "true/false/null",
    "amine": "true/false/null",
    "hydride": "true/false/null",
    "other": "string or null"
  },

  "thermal_pyrolysis": {
    "ceramic_yield_pct": "number or null - residual mass after pyrolysis",
    "pyrolysis_temp_C": "number or null - max temperature reached",
    "pyrolysis_atmosphere": "string or null - argon/nitrogen/vacuum/air",
    "heating_rate_C_per_min": "number or null",
    "hold_time_min": "number or null",
    "tga_notes": "string or null - describe TGA behavior if interesting"
  },

  "crosslink_density": {
    "reported": "true/false/null",
    "value": "string or null - include units if given",
    "crosslinking_method": "string or null - thermal/UV/chemical/catalytic"
  },

  "molecular_weight": {
    "Mn_g_per_mol": "number or null",
    "Mw_g_per_mol": "number or null",
    "PDI": "number or null",
    "measurement_method": "string or null - GPC/MALDI/etc"
  },

  "backbone_architecture": {
    "type": "string or null - linear/branched/cyclic/cage/hyperbranched/network",
    "description": "string or null - further details if given"
  },

  "microstructural": {
    "ceramic_phase": "string or null - e.g. SiC, Si3N4, SiCN, SiOC, amorphous",
    "phase_composition_notes": "string or null",
    "grain_size_nm": "number or null",
    "porosity_pct": "number or null"
  },

  "mechanical_properties": {
    "hardness_GPa": "number or null",
    "modulus_GPa": "number or null",
    "strength_MPa": "number or null",
    "other_mechanical": "string or null"
  },

  "conformation_crystallinity": {
    "crystallinity_state": "string or null - amorphous/semi-crystalline/crystalline",
    "xrd_phases_identified": "string or null",
    "crystallite_size_nm": "number or null"
  },

  "notes": "string or null - anything important not captured above"
}

Return ONLY the JSON object, no markdown fences, no explanation before or after."""


def get_paper_chunks(source_name, max_chunks=40):
    """Retrieve up to max_chunks chunks for a given paper, prioritizing informative sections."""
    results = collection.get(where={"source": source_name}, limit=5000)
    docs  = results.get("documents", [])
    metas = results.get("metadatas", [])

    # Score each chunk by section importance
    priority_sections = ["abstract", "text", "section_header", "paragraph", "table",
                         "methods", "results", "experimental", "discussion"]

    scored = []
    for doc, meta in zip(docs, metas):
        section = meta.get("section", "unknown").lower()
        score = 0
        for i, ps in enumerate(priority_sections):
            if ps in section:
                score = len(priority_sections) - i
                break
        # Tables and longer text chunks tend to be more informative
        if meta.get("type") == "table":
            score += 5
        scored.append((score, doc, meta))

    # Sort by score descending, take top chunks
    scored.sort(key=lambda x: -x[0])
    selected = scored[:max_chunks]

    source_type = "pdc"
    if selected:
        source_type = selected[0][2].get("source_type", "pdc")

    text = "\n\n".join([f"[Section: {m.get('section','unknown')}]\n{d}" for _, d, m in selected])
    return text, source_type


def extract_from_paper(source_name):
    """Send all relevant chunks from a paper to Claude for structured extraction."""
    text, source_type = get_paper_chunks(source_name)

    if not text.strip():
        return None, "pdc"

    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": f"{EXTRACTION_PROMPT}\n\n--- PAPER TEXT ---\n\n{text[:60000]}"
        }]
    )

    raw = message.content[0].text.strip()

    # Clean up potential markdown fences
    for prefix in ["```json", "```"]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    return json.loads(raw), source_type


# ── Main ─────────────────────────────────────────────────────────────────────

# Get unique paper sources (paginated for ChromaDB 1.x compatibility)
sources_set = set()
offset = 0
batch_size = 1000
total = collection.count()
while offset < total:
    result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
    for m in result.get("metadatas", []):
        if "source" in m:
            sources_set.add(m["source"])
    offset += batch_size
sources = sorted(sources_set)

print(f"Found {len(sources)} unique papers in the database")

# Load existing results so we can resume if interrupted
existing = []
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r") as f:
        existing = json.load(f)

done_sources = set(e.get("source") for e in existing)
remaining = [s for s in sources if s not in done_sources]

print(f"Already extracted: {len(done_sources)}")
print(f"To extract: {len(remaining)}")

if remaining and "--yes" not in sys.argv:
    confirm = input(
        f"\nThis will make {len(remaining)} Claude API calls (costs money). Continue? [y/N] "
    ).strip().lower()
    if confirm != "y":
        print("Aborted.")
        exit()

results = list(existing)

for i, source in enumerate(remaining):
    print(f"\n[{i+1}/{len(remaining)}] Extracting: {source[:70]}")
    try:
        desc, source_type = extract_from_paper(source)
        if desc:
            desc["source"] = source
            desc["source_type"] = source_type
            results.append(desc)

            # Save after every paper so we do not lose progress
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Success")
        else:
            print(f"  Skipped (no content)")
    except json.JSONDecodeError as e:
        print(f"  JSON parsing failed: {e}")
    except Exception as e:
        print(f"  Error: {e}")

print(f"\n" + "="*60)
print(f"Done. Extracted {len(results)} papers total.")
print(f"Saved to: {OUTPUT_FILE}")
print(f"="*60)

# Also save as CSV for easy viewing in Excel
import csv

def flatten(d, prefix=""):
    """Flatten nested dict into single-level with dotted keys."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten(v, key))
        else:
            flat[key] = v
    return flat

csv_path = OUTPUT_FILE.with_suffix(".csv")
if results:
    flat_rows = [flatten(r) for r in results]
    all_keys = set()
    for row in flat_rows:
        all_keys.update(row.keys())
    all_keys = sorted(all_keys)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"CSV version saved to: {csv_path}")
