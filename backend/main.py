"""
main.py — Day 3 (endpoint) + Day 4 (streaming + LangSmith)

What this file does:
1. Creates a FastAPI app
2. Exposes POST /chat — returns a full JSON answer (Day 3)
3. Exposes GET /chat/stream — SSE stream of tokens (Day 4)
4. Adds CORS so the Next.js frontend (localhost:3000) can call it

What is FastAPI?
  - An async Python web framework built on Starlette + Pydantic
  - Async matters here: while waiting for Gemini to generate tokens,
    the server can handle other requests (vs blocking a thread)
  - Automatic /docs (Swagger UI) and /redoc endpoints — visit them!

What is CORS?
  - Browsers block requests from one origin (localhost:3000) to another
    (localhost:8000) unless the server explicitly allows it
  - CORSMiddleware adds the right headers to tell browsers it's OK

What is SSE?
  - Server-Sent Events: a simple HTTP streaming protocol
  - The server keeps the connection open and sends "data: ..." lines
  - The browser's EventSource API reads them natively
  - Simpler than WebSockets for one-directional streaming (server → client)

Run this with:
  uvicorn main:app --reload --port 8000

Test it:
  # Full response
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "What is LCEL?"}'

  # Streaming
  curl -N http://localhost:8000/chat/stream?question=What+is+LCEL
"""

import json
import os
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from chain import ask, ask_stream

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="LearnChat API",
    description="RAG chatbot for LangChain/LangGraph/LangSmith docs",
    version="0.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow requests from the Next.js dev server (and any localhost port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response schemas (Pydantic validates these automatically)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What is LCEL?"}]
        }
    }


class SourceDoc(BaseModel):
    source: Optional[str]
    url: Optional[str]
    score: Optional[float]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Quick liveness check — useful for deployment health checks."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Day 3 endpoint: full synchronous response.

    Returns the complete answer + source citations in one JSON payload.
    Good for testing the RAG chain before adding streaming.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = ask(body.question)
        return ChatResponse(
            answer=result["answer"],
            sources=[SourceDoc(**s) for s in result["sources"]],
        )

    except Exception as e:
        # In production you'd log this properly; for now print to console
        print(f"[ERROR] /chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(request: Request, question: str, history: str = "[]"):
    """
    Day 4 endpoint: SSE streaming response.

    How it works:
      1. Client opens a GET request (EventSource in the browser)
      2. Server keeps the connection open
      3. Server yields "data: <token>\\n\\n" as each token arrives from Gemini
      4. At the end, server sends a special "data: __SOURCES__[...]\\n\\n" event
      5. Client parses the sources and displays them as citations

    SSE format (required by the spec):
      data: <content>\\n\\n   ← two newlines end each event
      event: <type>\\n\\n     ← optional custom event type
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def event_generator() -> AsyncIterator[str]:
        """
        Wraps the synchronous ask_stream() generator in an async generator.

        Why the try/except: if the client disconnects mid-stream, we stop
        generating tokens rather than wasting compute on Gemini calls.
        """
        try:
            parsed_history = json.loads(history)
            for token in ask_stream(question, parsed_history):
                if "__SOURCES__" in token:
                    # Send sources as a named SSE event so frontend can distinguish
                    sources_json = token[token.index("__SOURCES__") + len("__SOURCES__"):]
                    yield f"event: sources\ndata: {sources_json}\n\n"
                else:
                    # Escape newlines in token for SSE format compliance
                    # (SSE uses \n\n as event separator)
                    safe_token = token.replace("\n", "\\n")
                    yield f"data: {safe_token}\n\n"

            # Signal stream completion
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            print(f"[ERROR] Stream failed: {e}")
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Prevent nginx/proxies from buffering the stream
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
