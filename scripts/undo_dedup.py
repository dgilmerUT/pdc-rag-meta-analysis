"""Undo a removal round (scripts/dedup.py or scripts/remove_records.py): move every removed file back
and restore the descriptor table as it was before that round.

Run from the project root. Undo the most recent round first.
  python3 scripts/undo_dedup.py              # the 2026-10-28 dedup
  python3 scripts/undo_dedup.py 2026-11-16   # the 2026-11-16 removal (DOI duplicates + proposals)

Rounds that saved removed_chunks.json get their chunks put back in data/chunks.json; then add them
back to the vector index with:  echo N | python3 scripts/ingest.py
For the 2026-10-28 round, rebuild the chunks and index instead:
  python3 scripts/chunk.py
  echo N | python3 scripts/ingest.py
"""
import csv
import json
import shutil
import sys
from pathlib import Path

DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-10-28"
LOG_DIR = Path(f"outputs/dedup_{DATE}")

with open(LOG_DIR / "removed_copies.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    src, dest = Path(r["moved_to"]), Path(r["file"])
    if r["moved_to"] and src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
shutil.copy2(LOG_DIR / "descriptors_pre_dedup.json", "outputs/descriptors.json")
shutil.copy2(LOG_DIR / "descriptors_pre_dedup.csv", "outputs/descriptors.csv")

removed_chunks = LOG_DIR / "removed_chunks.json"
if removed_chunks.exists():
    back = json.load(open(removed_chunks))
    C = json.load(open("data/chunks.json"))
    have = {c["source"] for c in C}
    C += [c for c in back if c["source"] not in have]
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(C, f, indent=2)
    print(f"put {len(back)} chunks back in data/chunks.json; now run: echo N | python3 scripts/ingest.py")
print(f"restored {len(rows)} files and the descriptor table from before the {DATE} round")
# the moved files are also excluded by parse_pdfs.py while this log exists; move the log aside after undoing
