# LearnChat

A RAG-powered chatbot that answers questions about LangChain, LangGraph, and LangSmith documentation. Ask natural language questions and get grounded answers with source citations, streamed in real time.

## Features

- **RAG pipeline** — retrieves relevant documentation chunks from Pinecone before generating answers
- **Streaming responses** — tokens appear in real time via Server-Sent Events
- **Source citations** — every answer links back to the original documentation pages
- **Conversation history** — follow-up questions retain prior context
- **Rate limiting** — 10 requests/minute per IP
- **Serverless-ready** — deployed on Vercel (frontend + backend)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python, Uvicorn |
| LLM | Llama 3.3 70B via Groq |
| Vector DB | Pinecone |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| Orchestration | LangChain / LCEL |
| Observability | LangSmith (optional) |

## Project Structure

```
LearnChat/
├── backend/
│   ├── main.py          # FastAPI app — /chat, /chat/stream, /health endpoints
│   ├── chain.py         # LCEL RAG chain (retriever → prompt → LLM → parser)
│   ├── embeddings.py    # HuggingFace Inference API embeddings wrapper
│   ├── scraper.py       # Scrapes & chunks LangChain/LangGraph/LangSmith docs
│   ├── ingest.py        # Embeds chunks and uploads to Pinecone
│   ├── requirements.txt
│   ├── .env.example
│   └── api/
│       └── index.py     # Vercel serverless entry point
└── frontend/
    ├── app/
    │   ├── page.tsx     # Chat page — state, SSE handling, streaming logic
    │   ├── layout.tsx
    │   └── globals.css
    ├── components/
    │   ├── ChatWindow.tsx    # Scrollable message list
    │   ├── MessageBubble.tsx # Individual message renderer
    │   ├── SidePanel.tsx     # Source document sidebar
    │   ├── SourceLinks.tsx   # Source link display
    │   └── types.ts          # Shared TypeScript interfaces
    └── package.json
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- API keys for: [Groq](https://console.groq.com), [Pinecone](https://app.pinecone.io), [HuggingFace](https://huggingface.co/settings/tokens)
- (Optional) [LangSmith](https://smith.langchain.com) for tracing

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your API keys
```

**One-time data ingestion** (only needed once, or when docs change):

```bash
python scraper.py   # fetches docs → chunks.json
python ingest.py    # embeds chunks → uploads to Pinecone
```

**Start the API server:**

```bash
uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:3000` in your browser.

## Environment Variables

### Backend (`.env`)

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key for LLM inference |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone index name (e.g. `langchain-docs`) |
| `HUGGINGFACE_API_KEY` | HuggingFace token (read access) |
| `LANGCHAIN_API_KEY` | LangSmith API key (optional) |
| `LANGCHAIN_TRACING_V2` | `true` to enable LangSmith tracing (optional) |
| `LANGCHAIN_PROJECT` | LangSmith project name (optional) |

### Frontend (`.env.local`)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend URL (`http://localhost:8000` for local dev) |

## How It Works

```
User question
     │
     ▼
Frontend (Next.js)
     │  SSE connection to /chat/stream
     ▼
Backend (FastAPI)
     │
     ├─► Pinecone: retrieve top-5 similar chunks
     │
     ├─► Prompt: inject retrieved context + question + history
     │
     └─► Groq (Llama 3.3 70B): stream tokens back via SSE
           │
           ▼
     Frontend: accumulates tokens in real time, displays sources on completion
```

**Data ingestion** (one-time setup):
1. `scraper.py` fetches 11 documentation pages, splits them into ~1,000-character chunks with 200-character overlap, and saves to `chunks.json`
2. `ingest.py` embeds each chunk using HuggingFace (`all-MiniLM-L6-v2`, 384 dimensions) and uploads batches to Pinecone

## Deployment

The project is deployed on Vercel:

- **Frontend**: standard Next.js deployment
- **Backend**: Python serverless functions via `backend/vercel.json`

Pinecone is used instead of a local vector store because Vercel's ephemeral filesystem does not persist between function invocations. HuggingFace Inference API is used for embeddings to stay within Vercel's 250 MB bundle limit (avoids downloading `sentence-transformers` locally).

Live demo: [learn-chat-zeta.vercel.app](https://learn-chat-zeta.vercel.app)
