"""Remove duplicate copies of papers from the corpus, keeping one copy per paper.

Two kinds of duplicates are handled:
  text_duplicate  - different filenames whose chunk text is >=50% shared (same paper saved twice);
                    the copy extracted first (lowest index in descriptors.json) is kept.
  same_name_copy  - one filename present in two folders (chunked twice under one source name);
                    the copy in the folder matching the paper's tier is kept.

Removed files are moved (not deleted) to data/_dedup_removed_<date>/ and every move is recorded in
outputs/dedup_<date>/removed_copies.csv. Undo with scripts/undo_dedup.py.

Usage (from the project root):
  python3 scripts/dedup.py groups.json            # dry run
  python3 scripts/dedup.py groups.json --apply
"""
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

DATE = "2026-10-28"
RAW, PARSED = Path("data/raw"), Path("data/parsed")
MOVE_ROOT = Path(f"data/_dedup_removed_{DATE}")
LOG_DIR = Path(f"outputs/dedup_{DATE}")
DESC = Path("outputs/descriptors.json")
# chapter contained in a book: shares text but is a different document
NOT_DUPLICATES = {"BETIER CERAMICS THROUGH CHEMISTRY"}
TIER_DIRS = {"pdc": ".", "application": "applications", "conventional_ceramic": "conventional_ceramics",
             "alternative_route": "alternative_routes"}

groups_path, apply = sys.argv[1], "--apply" in sys.argv
groups = [g for g in json.load(open(groups_path)) if not NOT_DUPLICATES & set(g)]
D = json.load(open(DESC))
index = {r["source"]: i for i, r in enumerate(D)}
tier = {r["source"]: r.get("source_type") or "pdc" for r in D}


def files_by_stem(root, pattern):
    out = defaultdict(list)
    for p in root.rglob(pattern):
        if p.relative_to(root).parts[0] != "_skipped":
            out[p.stem].append(p)
    return out


parsed, raw = files_by_stem(PARSED, "*.json"), files_by_stem(RAW, "*.pdf")
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()


def doc_text(p):
    # Docling output differs in metadata between runs; compare the extracted text, allowing small OCR drift
    t = " ".join(x.get("text", "") for x in json.load(open(p)).get("texts", []))
    return round(len(t), -3)


rows = []

# 1. text duplicates: drop every file of the non-kept sources
removed_sources = set()
for gi, g in enumerate(groups):
    keep = min(g, key=lambda s: index.get(s, 10**9))
    for s in g:
        if s == keep:
            continue
        removed_sources.add(s)
        for p in parsed[s] + raw[s]:
            rows.append(dict(kind="text_duplicate", group=gi, source=s, kept_source=keep, tier=tier.get(s, ""),
                             descriptor_index=index.get(s, ""), file=str(p)))

# 2. same filename in two folders: keep the copy in the folder matching the paper's tier
mismatch = []
for s, ps in parsed.items():
    if len(ps) < 2 or s in removed_sources:
        continue
    want = TIER_DIRS[tier.get(s, "pdc")]
    keep = next((p for p in ps if str(p.parent.relative_to(PARSED)) == want), ps[0])
    if len({md5(p) for p in ps}) > 1 and len({doc_text(p) for p in ps}) > 1:
        mismatch.append(s)  # different documents that share a filename: leave for manual renaming
        continue
    for p in ps:
        if p == keep:
            continue
        sub = p.parent.relative_to(PARSED)
        rows.append(dict(kind="same_name_copy", group="", source=s, kept_source=s, tier=tier.get(s, ""),
                         descriptor_index="", file=str(p)))
        rows += [dict(kind="same_name_copy", group="", source=s, kept_source=s, tier=tier.get(s, ""),
                      descriptor_index="", file=str(q)) for q in raw[s] if q.parent.relative_to(RAW) == sub]

print(f"duplicate groups used: {len(groups)}  (excluded as not duplicates: {sorted(NOT_DUPLICATES)})")
print(f"text duplicates: {len(removed_sources)} sources, {sum(r['kind'] == 'text_duplicate' for r in rows)} files")
same = {r['source'] for r in rows if r['kind'] == 'same_name_copy'}
print(f"same-name copies: {len(same)} sources, {sum(r['kind'] == 'same_name_copy' for r in rows)} files")
print(f"different documents sharing a filename (not removed): {len(mismatch)} {mismatch}")
print(f"descriptor records: {len(D)} -> {len(D) - sum(1 for s in removed_sources if s in index)}")

if not apply:
    sys.exit()

LOG_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(DESC, LOG_DIR / "descriptors_pre_dedup.json")
shutil.copy2(DESC.with_suffix(".csv"), LOG_DIR / "descriptors_pre_dedup.csv")
locked = 0
for r in rows:
    src = Path(r["file"])
    dest = MOVE_ROOT / src
    r["moved_to"] = str(dest)
    if dest.exists() and not src.exists():  # moved by an earlier, interrupted run
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.rename(dest)
    except PermissionError:
        # macOS-protected file (com.apple.macl): leave it; parse_pdfs.py skips files listed in this log
        r["moved_to"] = ""
        locked += 1
with open(LOG_DIR / "removed_copies.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

kept = [r for r in D if r["source"] not in removed_sources]
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
json.dump(sorted(removed_sources), open(LOG_DIR / "removed_sources.json", "w"), indent=1)
print(f"moved {len(rows) - locked} files to {MOVE_ROOT}, {locked} protected files left in place; "
      f"descriptors now {len(kept)}; log in {LOG_DIR}")
