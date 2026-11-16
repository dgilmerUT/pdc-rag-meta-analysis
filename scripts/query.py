"""Search the preceramic_polymers ChromaDB collection.

Usage:
    python3 scripts/query.py "SiOC creep above 1200 C"
    python3 scripts/query.py "polysilazane crosslinking" -k 10 --tier pdc
    python3 scripts/query.py "boron doping" --per-paper 1   # best chunk per paper only
    python3 scripts/query.py "crystallization of SiOC" --contains "1400"  # semantic + exact text

Chunks are paragraph-sized (median ~180 characters) and the embedding is plain
SciBERT, so similarity alone is rough; --contains sharpens it a lot.
"""
import argparse
import logging
import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
import chromadb
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("-k", type=int, default=8, help="results to show")
ap.add_argument("--tier", help="source_type filter: pdc, conventional_ceramic, application, alternative")
ap.add_argument("--per-paper", type=int, default=0, help="max chunks per paper (0 = no limit)")
ap.add_argument("--min-len", type=int, default=80, help="skip chunks shorter than this (figure labels, headers)")
ap.add_argument("--contains", help="only chunks containing this exact text (case-sensitive)")
ap.add_argument("--chars", type=int, default=600, help="characters of text to print per hit")
a = ap.parse_args()

model = SentenceTransformer("allenai/scibert_scivocab_uncased")
col = chromadb.PersistentClient(path="vector_store/chroma_db").get_collection("preceramic_polymers")

where = {"source_type": a.tier} if a.tier else None
fetch = max(400, a.k * 50)
res = col.query(query_embeddings=model.encode([a.query]).tolist(), n_results=fetch,
                where=where, where_document={"$contains": a.contains} if a.contains else None,
                include=["documents", "metadatas", "distances"])

seen, shown = {}, 0
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    src = meta["source"]
    if len(doc) < a.min_len:
        continue
    if a.per_paper and seen.get(src, 0) >= a.per_paper:
        continue
    seen[src] = seen.get(src, 0) + 1
    shown += 1
    print(f"\n[{shown}] {src}  |  {meta.get('section', '')}  |  {meta.get('source_type', '')}  |  sim {1 - dist:.3f}")
    print("   " + doc[:a.chars].replace("\n", " ") + ("..." if len(doc) > a.chars else ""))
    if shown >= a.k:
        break
