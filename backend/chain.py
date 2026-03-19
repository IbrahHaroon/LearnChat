"""
chain.py — Day 3: The RAG chain that powers every answer

What this file does:
1. Wraps ChromaDB in a LangChain Retriever (standard interface)
2. Builds a prompt template that grounds Gemini in the retrieved docs
3. Composes retriever → prompt → LLM into an LCEL chain
4. Exposes two functions: ask() (full response) and ask_stream() (token stream)

What is LCEL?
  LCEL = LangChain Expression Language. It's a pipe-based syntax for
  composing chains:

      chain = retriever | format_docs | prompt | llm | output_parser

  Each "|" passes the output of the left side as input to the right side.
  This makes chains composable, lazy (only runs when invoked), and easy to
  trace in LangSmith.

Why a custom retriever wrapper?
  ChromaDB's native query() works fine, but LangChain's Retriever interface
  gives us:
    - Automatic LangSmith tracing of what was retrieved
    - Easy swapping (want to try Pinecone? Just swap the retriever)
    - Integration with LCEL's pipe operator

Why system prompt engineering matters:
  Without grounding, Gemini might hallucinate. The prompt explicitly tells
  it to ONLY use the provided context, which is the whole point of RAG.
"""

import os
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_groq import ChatGroq

import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "langchain_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# Groq: free tier, fast inference, great for RAG
LLM_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# 1. Custom LangChain Retriever wrapping ChromaDB
# ---------------------------------------------------------------------------

class ChromaRetriever(BaseRetriever):
    """
    A LangChain-compatible retriever backed by our ChromaDB collection.

    Why subclass BaseRetriever instead of using ChromaDB's LangChain integration?
    - Transparency: you see exactly what happens during retrieval
    - The LangChain ChromaDB wrapper is fine too, but this shows the internals
    - We can attach source URLs as document metadata (important for the frontend)
    """

    chroma_dir: str = CHROMA_DIR
    collection_name: str = COLLECTION_NAME
    embedding_model: str = EMBEDDING_MODEL
    top_k: int = TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Called by LangChain whenever the retriever is invoked.
        run_manager is injected by LangChain — it handles LangSmith tracing.
        """
        client = chromadb.PersistentClient(path=self.chroma_dir)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
        )

        results = collection.query(
            query_texts=[query],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # Convert to LangChain Document objects (standard format)
        return [
            Document(
                page_content=doc,
                metadata={
                    **meta,
                    # Attach similarity score for debugging / display
                    "score": round(1 - dist / 2, 4),
                },
            )
            for doc, meta, dist in zip(docs, metas, distances)
        ]


# ---------------------------------------------------------------------------
# 2. Prompt template
# ---------------------------------------------------------------------------

# This is the heart of RAG. The template:
#   - Gives the LLM its role (helpful assistant for LangChain docs)
#   - Injects the retrieved context chunks
#   - Asks it to cite sources and say "I don't know" if unsure
#
# {context} and {question} are filled in by the chain at runtime.

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions about \
LangChain, LangGraph, and LangSmith based on their official documentation.

Use ONLY the context provided below to answer the question. \
If the context doesn't contain enough information to answer confidently, \
say "I don't have enough information in the docs to answer that."

When you answer:
- Be clear and concise
- Reference specific concepts by name
- Mention which doc section the info comes from (e.g., "According to the LCEL docs...")
- Use the conversation history below to understand follow-up questions and pronouns like "it", "that", "they"

Context from documentation:
{context}

Conversation history:
{history}
"""),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# 3. Helper: format retrieved docs into the {context} string
# ---------------------------------------------------------------------------

def format_docs(docs: list[Document]) -> str:
    """
    Turn a list of Document objects into a single context string for the prompt.

    We include the source label so the LLM can reference it in its answer,
    and the URL so we can surface it in the frontend.
    """
    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        url = doc.metadata.get("url", "")
        sections.append(
            f"[Source {i}: {source}]\n"
            f"URL: {url}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# 4. Build the RAG chain
# ---------------------------------------------------------------------------

def build_chain():
    """
    Compose the full RAG pipeline using LCEL.

    Data flow:
      question (str)
        → retriever       : finds top-K relevant doc chunks
        → format_docs     : formats them into a context string
        → prompt          : injects context + question into the template
        → llm             : Gemini generates the answer
        → StrOutputParser : extracts the plain text from the LLM response

    The | operator is LCEL's pipe. Each step receives the previous step's output.

    Note: We use a dict to split the input so both {context} and {question}
    can reach the prompt template. This is a common LCEL pattern.
    """
    retriever = ChromaRetriever()

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.2,       # Low temp = more factual, less creative
        streaming=True,        # Enable token-by-token streaming
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # The chain:
    # 1. Take {"question": "..."} as input
    # 2. Retrieve docs AND pass question through to prompt
    # 3. Format docs into context string
    # 4. Fill prompt template
    # 5. Call LLM
    # 6. Parse output to string
    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ---------------------------------------------------------------------------
# 5. Public API used by main.py
# ---------------------------------------------------------------------------

def ask(question: str) -> dict:
    """
    Single-shot Q&A: returns the full answer + source URLs.
    Used by main.py for non-streaming responses (Day 3).
    """
    chain, retriever = build_chain()

    # Get sources separately so we can return them alongside the answer
    sources = retriever.invoke(question)
    answer = chain.invoke({"question": question})

    return {
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "url": doc.metadata.get("url"),
                "score": doc.metadata.get("score"),
            }
            for doc in sources
        ],
    }


def ask_stream(question: str, history: list[dict] = []) -> Iterator[str]:
    """
    Streaming Q&A: yields tokens as they arrive from Gemini.
    Used by main.py for SSE streaming (Day 4).

    Yields special markers:
      - Regular tokens: just the text
      - At the end: a JSON-encoded "sources" event so the frontend
        can display citations after the answer streams in
    """
    import json

    chain, retriever = build_chain()

    # Retrieve sources first (fast, local ChromaDB query)
    sources = retriever.invoke(question)

    # Format history as a readable string for the prompt
    history_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    ) if history else "No prior conversation."

    # Stream the LLM response token by token
    for token in chain.stream({"question": question, "history": history_str}):
        yield token

    # After all tokens, send a special delimiter + sources JSON
    # The frontend will parse this out and display the source links
    source_data = [
        {
            "source": doc.metadata.get("source"),
            "url": doc.metadata.get("url"),
            "score": doc.metadata.get("score"),
        }
        for doc in sources
    ]
    yield f"\n\n__SOURCES__{json.dumps(source_data)}"


# ---------------------------------------------------------------------------
# 6. Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LearnChat Chain — Day 3 Test")
    print("=" * 60)
    print("\nMake sure you've run scraper.py and ingest.py first!\n")

    question = "What is LCEL and why would I use it?"
    print(f"Question: {question}\n")
    print("-" * 60)

    result = ask(question)
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (score={s['score']})")
        print(f"    {s['url']}")
