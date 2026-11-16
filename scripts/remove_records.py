"""Remove whole records (papers) from the corpus, with a full backup so it can be undone.

The removal list is a JSON file of {"source", "kind", "kept_source", "reason"} entries, using the
exact `source` strings from outputs/descriptors.json. For each record this:
  - moves its parsed JSON and raw PDF to data/_dedup_removed_<date>/ (logged in removed_copies.csv,
    the same format scripts/dedup.py writes, so parse_pdfs.py keeps skipping them),
  - drops it from outputs/descriptors.json / .csv (pre-removal copies saved first),
  - drops its chunks from data/chunks.json (saved to removed_chunks.json),
  - deletes its chunks from the ChromaDB collection.
Undo with: python3 scripts/undo_dedup.py <date>

Usage (from the project root):
  python3 scripts/remove_records.py <date> <list.json>            # dry run
  python3 scripts/remove_records.py <date> <list.json> --apply
"""
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

DATE, LIST = sys.argv[1], Path(sys.argv[2])
APPLY = "--apply" in sys.argv
RAW, PARSED = Path("data/raw"), Path("data/parsed")
MOVE_ROOT = Path(f"data/_dedup_removed_{DATE}")
LOG_DIR = Path(f"outputs/dedup_{DATE}")
DESC, CHUNKS = Path("outputs/descriptors.json"), Path("data/chunks.json")

plan = json.load(open(LIST))
D = json.load(open(DESC))
index = {r["source"]: i for i, r in enumerate(D)}
tier = {r["source"]: r.get("source_type") or "pdc" for r in D}
missing = [p["source"] for p in plan if p["source"] not in index]
if missing:
    raise SystemExit(f"not in descriptors.json: {missing}")
drop = {p["source"] for p in plan}


def files_by_stem(root, pattern):
    out = defaultdict(list)
    for p in root.rglob(pattern):
        if p.relative_to(root).parts[0] != "_skipped":
            out[p.stem].append(p)
    return out


parsed, raw = files_by_stem(PARSED, "*.json"), files_by_stem(RAW, "*.pdf")
rows = []
for gi, p in enumerate(plan):
    s = p["source"]
    files = parsed.get(s, []) + raw.get(s, [])
    if not files:
        print(f"  warning: no files found for {s!r}")
    for f in files:
        rows.append(dict(kind=p["kind"], group=gi, source=s, kept_source=p.get("kept_source", ""),
                         tier=tier[s], descriptor_index=index[s], file=str(f)))

by_tier = defaultdict(int)
for s in drop:
    by_tier[tier[s]] += 1
print(f"records to remove: {len(drop)} {dict(by_tier)}; files to move: {len(rows)}")
print(f"descriptor records: {len(D)} -> {len(D) - len(drop)}")
if not APPLY:
    sys.exit()

LOG_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(DESC, LOG_DIR / "descriptors_pre_dedup.json")
shutil.copy2(DESC.with_suffix(".csv"), LOG_DIR / "descriptors_pre_dedup.csv")
shutil.copy2(LIST, LOG_DIR / "removal_list.json")

# files
locked = 0
for r in rows:
    src = Path(r["file"])
    dest = MOVE_ROOT / src
    r["moved_to"] = str(dest)
    if dest.exists() and not src.exists():
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.rename(dest)
    except PermissionError:
        r["moved_to"] = ""      # macOS-protected; parse_pdfs.py skips files listed in this log
        locked += 1
with open(LOG_DIR / "removed_copies.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["kind", "group", "source", "kept_source", "tier",
                                      "descriptor_index", "file", "moved_to"])
    w.writeheader()
    w.writerows(rows)

# descriptors
kept = [r for r in D if r["source"] not in drop]
json.dump(kept, open(DESC, "w"), indent=2)


def flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        flat.update(flatten(v, key) if isinstance(v, dict) else {key: v})
    return flat


flat = [flatten(r) for r in kept]
with open(DESC.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sorted(set().union(*flat)))
    w.writeheader()
    w.writerows(flat)
json.dump(sorted(drop), open(LOG_DIR / "removed_sources.json", "w"), indent=1)

# chunks
C = json.load(open(CHUNKS))
json.dump([c for c in C if c["source"] in drop], open(LOG_DIR / "removed_chunks.json", "w"))
kept_chunks = [c for c in C if c["source"] not in drop]
with open(CHUNKS, "w", encoding="utf-8") as f:
    json.dump(kept_chunks, f, indent=2)   # same format as chunk.py

# vector index
import chromadb
col = chromadb.PersistentClient(path="vector_store/chroma_db").get_collection("preceramic_polymers")
before = col.count()
for s in drop:
    col.delete(where={"source": s})
print(f"moved {len(rows) - locked} files ({locked} protected, left in place); descriptors {len(kept)}; "
      f"chunks {len(C)} -> {len(kept_chunks)}; vector index {before} -> {col.count()}; log in {LOG_DIR}")
