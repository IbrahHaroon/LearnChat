"""
ingest.py — Embed chunks and store them in Pinecone

What changed from ChromaDB:
  - ChromaDB (local file) → Pinecone (hosted cloud vector DB)
  - sentence-transformers (local 500MB model) → HuggingFace Inference API (HTTP call)

Why these changes?
  - Vercel serverless functions have a 250MB size limit — sentence-transformers
    + PyTorch alone exceed this. Using the HF Inference API means no model download.
  - ChromaDB writes to disk, which doesn't work on Vercel's ephemeral filesystem.
    Pinecone is a hosted vector DB — data lives in the cloud, survives deploys.

Run this ONCE locally to populate your Pinecone index.
You do NOT need to re-run unless you want to re-index the docs.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from embeddings import HFEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNKS_PATH = "chunks.json"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-docs")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384   # all-MiniLM-L6-v2 produces 384-dimensional vectors
TOP_K = 5


# ---------------------------------------------------------------------------
# 1. Embeddings — HuggingFace Inference API (no local model needed)
# ---------------------------------------------------------------------------

def get_embeddings():
    return HFEmbeddings(api_key=os.getenv("HUGGINGFACE_API_KEY"), model=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# 2. Create Pinecone index if it doesn't exist
# ---------------------------------------------------------------------------

def get_or_create_index(pc: Pinecone) -> None:
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


# ---------------------------------------------------------------------------
# 3. Load chunks from scraper output
# ---------------------------------------------------------------------------

def load_chunks(path: str = CHUNKS_PATH) -> list:
    if not Path(path).exists():
        raise FileNotFoundError(f"'{path}' not found. Run `python scraper.py` first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 4. Ingest into Pinecone
# ---------------------------------------------------------------------------

def ingest() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    get_or_create_index(pc)

    embeddings = get_embeddings()

    # Convert raw chunks to LangChain Document objects
    docs = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in chunks
    ]

    print(f"\nEmbedding and uploading {len(docs)} chunks to Pinecone...")
    print("This calls the HuggingFace Inference API — may take a few minutes.\n")

    # Upload in batches of 50
    batch_size = 50
    index = pc.Index(INDEX_NAME)

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=INDEX_NAME,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )
        end = min(i + batch_size, len(docs))
        print(f"  Uploaded chunks {i}–{end - 1}  ({end}/{len(docs)})")

    print(f"\n[OK] Ingestion complete. {len(docs)} chunks in Pinecone index '{INDEX_NAME}'.")


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LearnChat Ingest — Pinecone")
    print("=" * 60)
    ingest()
