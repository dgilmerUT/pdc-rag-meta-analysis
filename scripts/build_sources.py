"""Build the corpus source list (one row per descriptor record) with DOI and bibliographic data.

Step 1: look for a DOI in the paper's own first chunks (title page, header, footer).
Step 2: confirm it on Crossref and check that the Crossref title matches the paper.
Step 3: if there is no DOI or it does not match, search Crossref by the paper's title.
Rows that still have no confident match are left for manual entry (match = "none").

Resumable: Crossref answers are cached in outputs/sources_cache.json.
Usage: python3 scripts/build_sources.py [N]   (N = only the first N records, for testing)
"""
import csv
import json
import re
import time
import urllib.parse
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import requests

DESC = "outputs/descriptors.json"
CHUNKS = "data/chunks.json"
OUT = Path("outputs/sources.csv" if len(__import__("sys").argv) < 2 else "outputs/sources_test.csv")
CACHE = Path("outputs/sources_cache.json")
HEAD_CHUNKS = 15          # chunks treated as the title page
UA = {"User-Agent": "pdc-rag-meta-analysis/1.0 (corpus source list builder)"}
DOI_RX = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>()\[\],;]+)", re.I)

cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def crossref(url):
    if url in cache:
        return cache[url]
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 404:
                cache[url] = None
                break
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            cache[url] = r.json()["message"]
            break
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    else:
        return None
    time.sleep(0.25)
    if len(cache) % 25 == 0:
        CACHE.write_text(json.dumps(cache))
    return cache.get(url)


def norm(s):
    s = re.sub(r"<[^>]+>", "", (s or "").lower())          # Crossref markup such as <scp>
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", s)).strip()


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    short, long_ = sorted((a, b), key=len)
    if len(short) > 25 and short in long_ and len(short) >= 0.6 * len(long_):
        return 1.0
    cut = short.rsplit(" ", 1)[0]                # filenames are often cut off mid-title
    if len(cut) >= 35 and long_.startswith(cut):
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


def title_guess(source, head):
    """(title, year, first author) from the filename, else the first substantial header chunk.

    Filename patterns: Author_Year_Title, and Wiley's "Journal - Year - Author - Title".
    """
    m = re.match(r"^([A-Za-z\-']+)_(\d{4})_(.+)$", source)
    if m:
        return m.group(3).replace("_", " "), int(m.group(2)), m.group(1)
    m = re.match(r"^.+? - (\d{4}) - ([A-Za-z\-' ]+?) - (.+)$", source)
    if m:
        return m.group(3), int(m.group(1)), m.group(2).split()[-1]
    for c in head:
        if c["type"] == "section_header" and 20 < len(c["text"]) < 250:
            return c["text"], None, None
    return source.replace("_", " "), None, None


def search(guess, fyear, author=None):
    """Best Crossref title-search hit: (record, similarity)."""
    params = {"query.bibliographic": guess[:300], "rows": 5 if author else 3}
    if author:
        params["query.author"] = author
        params["filter"] = f"from-pub-date:{fyear - 1},until-pub-date:{fyear + 1}"
    msg = crossref("https://api.crossref.org/works?" + urllib.parse.urlencode(params))
    best, best_s = None, 0.0
    for it in (msg or {}).get("items", []):
        rec = record(it)
        s = sim(rec["title"], guess)
        if fyear and rec["year"] and abs(rec["year"] - fyear) > 1:
            s -= 0.15
        if s > best_s:
            best, best_s = rec, s
    return best, best_s


def record(msg):
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        parts = (msg.get(k) or {}).get("date-parts") or [[None]]
        if parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    authors = "; ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                        for a in msg.get("author", [])[:20])
    return {"doi": msg.get("DOI", "").lower(), "title": " ".join(msg.get("title") or [""]),
            "authors": authors, "year": year, "container": " ".join(msg.get("container-title") or [""]),
            "type": msg.get("type", "")}


def main(limit=None):
    D = json.load(open(DESC))[:limit]
    head = defaultdict(list)
    for c in json.load(open(CHUNKS)):
        if len(head[c["source"]]) < HEAD_CHUNKS:
            head[c["source"]].append(c)

    rows = []
    for i, r in enumerate(D):
        src = r["source"]
        h = head.get(src, [])
        guess, fyear, author = title_guess(src, h)
        row = {"source": src, "tier": r.get("source_type") or "pdc", "title_guess": guess,
               "doi": "", "title": "", "authors": "", "year": fyear or "", "container": "", "type": "",
               "match": "none", "title_similarity": ""}

        # 1. DOI printed on the paper itself
        found = []
        for c in h:
            for d in DOI_RX.findall(c["text"]):
                d = d.rstrip(".").lower()
                if d not in found:
                    found.append(d)
        for d in found[:3]:
            msg = crossref("https://api.crossref.org/works/" + urllib.parse.quote(d))
            if not msg:
                continue
            rec = record(msg)
            s = max(sim(rec["title"], guess), max((sim(rec["title"], c["text"]) for c in h[:6]), default=0))
            if s >= 0.75:
                row.update(rec, match="doi_in_text", title_similarity=round(s, 2))
                break

        # 2. Crossref title search
        if row["match"] == "none" and len(norm(guess)) > 15:
            best, best_s = search(guess, fyear)
            if best_s < 0.85 and author and fyear:   # retry narrowed to the filename's author and year
                b2, s2 = search(guess, fyear, author)
                if s2 > best_s:
                    best, best_s = b2, s2
            if best and best["title"].lower().startswith(("cheminform", "erratum", "corrigendum")):
                best_s = min(best_s, 0.8)            # abstracting-service or correction record: check by hand
            if best and best_s >= 0.85:
                row.update(best, match="title_search", title_similarity=round(best_s, 2))
            elif best and best_s >= 0.65:
                row.update({"match": "check", "title_similarity": round(best_s, 2),
                            "doi": best["doi"], "title": best["title"], "authors": best["authors"],
                            "year": best["year"] or row["year"], "container": best["container"],
                            "type": best["type"]})
        rows.append(row)
        if (i + 1) % 50 == 0:
            done = sum(1 for x in rows if x["match"] in ("doi_in_text", "title_search"))
            print(f"{i + 1}/{len(D)}  matched {done}", flush=True)

    # a DOI can belong to only one record: keep the best title match, flag the others for checking
    by_doi = defaultdict(list)
    for x in rows:
        if x["doi"] and x["match"] in ("doi_in_text", "title_search"):
            by_doi[x["doi"]].append(x)
    for xs in by_doi.values():
        for x in sorted(xs, key=lambda x: -float(x["title_similarity"] or 0))[1:]:
            x["match"] = "check"

    CACHE.write_text(json.dumps(cache))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("Done:", dict(Counter(x["match"] for x in rows)), "->", OUT)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
