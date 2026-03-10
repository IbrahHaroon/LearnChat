"""
scraper.py — Day 1: Fetch and chunk LangChain/LangGraph/LangSmith docs

What this file does:
1. Downloads doc pages from the official sites
2. Strips HTML → plain text
3. Splits text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter
4. Saves chunks to chunks.json so you can inspect them before embedding

Why chunking matters:
  - LLMs have a context window limit (can't feed 500 pages at once)
  - We only want the *relevant* pieces — chunking + vector search finds them
  - Overlap (e.g. 200 chars) prevents a sentence being split mid-thought across chunks

Why RecursiveCharacterTextSplitter:
  - Tries to split on "\n\n" first (paragraphs), then "\n", then " ", then chars
  - This keeps semantic units together as much as possible
"""

import json
import time
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# 1. Define which doc pages to scrape
# ---------------------------------------------------------------------------
# We keep this small for Day 1. You can expand URLS later.
# Each entry is (url, source_label) — the label appears in the UI as a citation.

URLS = [
    # LangChain core concepts
    ("https://python.langchain.com/docs/introduction/", "langchain:introduction"),
    ("https://python.langchain.com/docs/concepts/lcel/", "langchain:lcel"),
    ("https://python.langchain.com/docs/concepts/rag/", "langchain:rag"),
    ("https://python.langchain.com/docs/concepts/vectorstores/", "langchain:vectorstores"),
    ("https://python.langchain.com/docs/concepts/retrievers/", "langchain:retrievers"),
    ("https://python.langchain.com/docs/concepts/chat_models/", "langchain:chat_models"),
    ("https://python.langchain.com/docs/concepts/prompt_templates/", "langchain:prompt_templates"),
    # LangGraph
    ("https://langchain-ai.github.io/langgraph/concepts/high_level/", "langgraph:overview"),
    ("https://langchain-ai.github.io/langgraph/concepts/low_level/", "langgraph:low_level"),
    # LangSmith
    ("https://docs.smith.langchain.com/overview", "langsmith:overview"),
    ("https://docs.smith.langchain.com/concepts/tracing", "langsmith:tracing"),
]

# ---------------------------------------------------------------------------
# 2. Fetch + extract text from a single page
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """Download a page and return its main text content (no nav, no sidebar)."""
    headers = {"User-Agent": "LearnChat-scraper/1.0 (educational project)"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove navigation, header, footer, sidebar — we only want article text
    for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style"]):
        tag.decompose()

    # Try to find the main content area (common patterns across doc sites)
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_="md-content")     # MkDocs
        or soup.find(class_="theme-doc-markdown")  # Docusaurus
        or soup.find(id="content")
        or soup.body
    )

    if main is None:
        return ""

    # get_text with separator=" " turns <br> and block elements into spaces
    text = main.get_text(separator=" ", strip=True)

    # Collapse multiple whitespace into single spaces (cleaner chunks)
    import re
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# 3. Chunk the text
# ---------------------------------------------------------------------------

def make_splitter() -> RecursiveCharacterTextSplitter:
    """
    chunk_size=1000  — each chunk is ~1000 characters (~200 tokens)
    chunk_overlap=200 — 200-char overlap so context isn't lost at boundaries

    These numbers are a starting point. You'd tune them based on:
      - Your LLM's context window
      - How detailed your answers need to be
      - Retrieval precision vs. recall trade-off
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        # Split order: paragraphs → sentences → words → characters
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(text: str, source: str, url: str) -> list[dict]:
    """Split text into chunks, attaching metadata to each chunk."""
    splitter = make_splitter()
    pieces = splitter.split_text(text)

    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append({
            "text": piece,
            "metadata": {
                "source": source,
                "url": url,
                "chunk_index": i,
                "char_count": len(piece),
            }
        })
    return chunks


# ---------------------------------------------------------------------------
# 4. Main scraping loop
# ---------------------------------------------------------------------------

def scrape_all() -> list[dict]:
    """Scrape all URLs and return a flat list of chunks."""
    all_chunks = []

    for url, source in URLS:
        print(f"\n→ Scraping: {source}")
        print(f"  URL: {url}")

        text = fetch_page(url)

        if not text:
            print("  [SKIP] Empty response")
            continue

        chunks = chunk_text(text, source=source, url=url)
        all_chunks.extend(chunks)

        print(f"  [OK] {len(text):,} chars → {len(chunks)} chunks")

        # Be polite to the servers — don't hammer them
        time.sleep(0.5)

    return all_chunks


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LearnChat Scraper — Day 1")
    print("=" * 60)

    chunks = scrape_all()

    print(f"\n{'=' * 60}")
    print(f"Total chunks collected: {len(chunks)}")
    print(f"{'=' * 60}")

    # Save to JSON so you can inspect the raw chunks before embedding
    output_path = "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")
    print("\nSample chunk (first one):")
    print("-" * 40)
    if chunks:
        sample = chunks[0]
        print(f"Source:  {sample['metadata']['source']}")
        print(f"URL:     {sample['metadata']['url']}")
        print(f"Chars:   {sample['metadata']['char_count']}")
        print(f"Text:    {sample['text'][:300]}...")
    print("-" * 40)
    print("\nNext step: run `python ingest.py` to embed these chunks into ChromaDB")
