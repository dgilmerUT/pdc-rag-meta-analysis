import json
import hashlib
import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNKS_FILE = Path("data/chunks.json")
CHROMA_DIR = "vector_store/chroma_db"

# ── Load chunks ──────────────────────────────────────────────────────────────

if not CHUNKS_FILE.exists():
    print("chunks.json not found. Run chunk.py first.")
    exit()

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print(f"Loaded {len(all_chunks)} chunks")

# ── Load embedding model ─────────────────────────────────────────────────────

print("\nLoading embedding model (this may take a minute the first time)...")
model = SentenceTransformer("allenai/scibert_scivocab_uncased")
print("Model ready")

# ── Connect to ChromaDB ──────────────────────────────────────────────────────

os.makedirs(CHROMA_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name="preceramic_polymers",
    metadata={"hnsw:space": "cosine"}
)

# Check existing count; if rebuilding (chunk size/ID scheme changed), start fresh
existing_count = collection.count()
print(f"Already indexed: {existing_count} chunks")

if existing_count > 0:
    rebuild = input(
        f"Collection has {existing_count} chunks from a previous run.\n"
        "Rebuild from scratch (recommended after chunk size changes)? [y/N] "
    ).strip().lower()
    if rebuild == "y":
        client.delete_collection("preceramic_polymers")
        collection = client.get_or_create_collection(
            name="preceramic_polymers",
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared.")
        existing_count = 0

existing_ids = set()
if existing_count > 0:
    offset = 0
    batch = 1000
    while offset < existing_count:
        result = collection.get(limit=batch, offset=offset, include=[])
        existing_ids.update(result["ids"])
        offset += batch
    print(f"Loaded {len(existing_ids)} existing IDs")

# ── Embed and store in batches ───────────────────────────────────────────────

BATCH_SIZE = 64
new_chunks = []
new_ids = []

seen_ids = set()
for chunk in all_chunks:
    content_hash = hashlib.sha256(
        f"{chunk['source']}:{chunk['section']}:{chunk['text']}".encode()
    ).hexdigest()[:16]
    chunk_id = f"{chunk['source']}_{content_hash}"
    if chunk_id not in existing_ids and chunk_id not in seen_ids:
        seen_ids.add(chunk_id)
        new_chunks.append((chunk_id, chunk))

print(f"New chunks to index: {len(new_chunks)}")

if len(new_chunks) == 0:
    print("Nothing new to index. Add more papers and re-run chunk.py first.")
    exit()

total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in range(0, len(new_chunks), BATCH_SIZE):
    batch = new_chunks[batch_num:batch_num + BATCH_SIZE]
    ids     = [item[0] for item in batch]
    chunks  = [item[1] for item in batch]
    texts   = [c["text"] for c in chunks]
    metas   = [{"source": c["source"], "section": c["section"], "type": c["type"],
                 "source_type": c.get("source_type", "pdc")} for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metas,
        ids=ids
    )

    done = min(batch_num + BATCH_SIZE, len(new_chunks))
    pct = int(done / len(new_chunks) * 100)
    print(f"  Progress: {done}/{len(new_chunks)} chunks ({pct}%)")

print(f"\nDone. Database now contains {collection.count()} total chunks.")
print(f"Stored in: {CHROMA_DIR}")