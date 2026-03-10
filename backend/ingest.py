"""
ingest.py — Day 2: Embed chunks and store them in ChromaDB

What this file does:
1. Loads chunks from chunks.json (produced by scraper.py)
2. Embeds each chunk using sentence-transformers (local model, no API cost)
3. Stores embeddings + text + metadata in ChromaDB (persisted to ./chroma_db/)
4. Provides a query() function so you can test retrieval immediately

What are embeddings?
  - A chunk of text → a list of ~384 numbers (a vector)
  - Similar meaning → similar vectors (close in "embedding space")
  - We use this to find doc chunks relevant to a user's question:
      embed(question) → search for nearest chunk vectors → return top-K

Why sentence-transformers / all-MiniLM-L6-v2?
  - Runs entirely on CPU (no GPU needed, no API call)
  - all-MiniLM-L6-v2 is 80MB, fast, and good enough for docs retrieval
  - Produces 384-dimensional vectors

Why ChromaDB?
  - Stores vectors on disk (./chroma_db/) — survives restarts
  - Built-in ANN search (approximate nearest neighbor)
  - Simple Python API, no server process needed
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNKS_PATH = "chunks.json"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "langchain_docs"

# all-MiniLM-L6-v2: 384 dims, ~80MB download on first run, very fast on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many docs to return per query (used in chain.py too)
TOP_K = 5


# ---------------------------------------------------------------------------
# 1. Load chunks
# ---------------------------------------------------------------------------

def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    """Load chunks produced by scraper.py."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"'{path}' not found. Run `python scraper.py` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Set up ChromaDB with sentence-transformer embeddings
# ---------------------------------------------------------------------------

def get_chroma_collection(persist_dir: str = CHROMA_DIR):
    """
    Create (or load existing) ChromaDB collection.

    ChromaDB handles:
      - Storing the raw text (documents)
      - Storing the vectors (embeddings)
      - Storing metadata (source, url, chunk_index)
      - ANN search: given a query vector, return top-K similar documents
    """
    client = chromadb.PersistentClient(path=persist_dir)

    # SentenceTransformerEmbeddingFunction tells ChromaDB how to embed text
    # ChromaDB will call this automatically when you add() or query()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # get_or_create: safe to call multiple times (idempotent)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity (standard for text)
    )

    return collection


# ---------------------------------------------------------------------------
# 3. Ingest chunks into ChromaDB
# ---------------------------------------------------------------------------

def ingest(chunks: list[dict], collection) -> None:
    """
    Add all chunks to ChromaDB in batches.

    ChromaDB .add() takes parallel lists:
      - ids:        unique string ID per chunk
      - documents:  the raw text (ChromaDB embeds this automatically)
      - metadatas:  dicts with source/url/etc.
    """
    # Check what's already ingested (so we can re-run safely)
    existing_count = collection.count()
    if existing_count > 0:
        print(f"[INFO] Collection already has {existing_count} chunks.")
        print("[INFO] Clearing and re-ingesting (clean slate)...")
        collection.delete(where={"chunk_index": {"$gte": 0}})

    print(f"\nIngesting {len(chunks)} chunks into ChromaDB...")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Persist directory: {CHROMA_DIR}")

    # Process in batches — ChromaDB handles large collections better this way
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        ids = [f"chunk_{i + j}" for j, _ in enumerate(batch)]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        end = min(i + batch_size, len(chunks))
        print(f"  Embedded chunks {i}–{end - 1}  ({end}/{len(chunks)})")

    print(f"\n[OK] Ingestion complete. Total in collection: {collection.count()}")


# ---------------------------------------------------------------------------
# 4. Query function — test retrieval before building the chain
# ---------------------------------------------------------------------------

def query(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed a question and retrieve the top-K most similar chunks.

    This is the core of RAG retrieval. The chain.py will call this same
    logic, but through a LangChain Retriever wrapper.

    Returns a list of dicts with: text, source, url, score
    """
    collection = get_chroma_collection()

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's response format (nested lists because batch query)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "url": meta.get("url", ""),
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score: 1 - distance/2 → [0, 1]
            "score": round(1 - dist / 2, 4),
        })

    return output


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LearnChat Ingest — Day 2")
    print("=" * 60)

    # Step 1: Load chunks
    chunks = load_chunks()
    print(f"\nLoaded {len(chunks)} chunks from {CHUNKS_PATH}")

    # Step 2: Ingest into ChromaDB (this embeds everything — takes ~1-2 min first time)
    collection = get_chroma_collection()
    ingest(chunks, collection)

    # Step 3: Test query to verify retrieval works
    print("\n" + "=" * 60)
    print("Test Query: 'What is LCEL?'")
    print("=" * 60)

    results = query("What is LCEL?")

    for i, result in enumerate(results, 1):
        print(f"\n[Result {i}] score={result['score']}  source={result['source']}")
        print(f"  URL: {result['url']}")
        print(f"  Text: {result['text'][:200]}...")

    print("\n" + "=" * 60)
    print("Next step: run `uvicorn main:app --reload` and curl the /chat endpoint")
