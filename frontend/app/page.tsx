"use client";

/**
 * page.tsx — Day 5/6: Main chat page
 *
 * This is where everything comes together:
 *   - Manages the list of messages (state)
 *   - Calls the FastAPI SSE stream endpoint
 *   - Passes data down to ChatWindow
 *
 * Key React patterns used here:
 *   - useState: hold the messages array and input value
 *   - useRef: scroll the chat window to the bottom automatically
 *   - EventSource (browser API): consumes the SSE stream from FastAPI
 */

import { useState, useRef, useCallback } from "react";
import ChatWindow from "@/components/ChatWindow";
import type { Message, SourceDoc } from "@/components/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const sendMessage = useCallback(async () => {
    const question = input.trim();
    if (!question || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    // Add user message immediately
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: question,
    };

    // Placeholder for the bot response (will be filled token by token)
    const botMsgId = (Date.now() + 1).toString();
    const botMsg: Message = {
      id: botMsgId,
      role: "assistant",
      content: "",
      sources: [],
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, botMsg]);

    // Open SSE connection to FastAPI
    // EventSource only supports GET, so we pass the question as a query param
    const url = `${API_BASE}/chat/stream?question=${encodeURIComponent(question)}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      // Default "message" event = a token from Gemini
      const token = event.data.replace(/\\n/g, "\n");
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsgId ? { ...m, content: m.content + token } : m
        )
      );
    };

    es.addEventListener("sources", (event) => {
      // Custom "sources" event — sent after all tokens are done
      try {
        const sources: SourceDoc[] = JSON.parse(event.data);
        setMessages((prev) =>
          prev.map((m) => (m.id === botMsgId ? { ...m, sources } : m))
        );
      } catch {
        // Malformed JSON from server — ignore
      }
    });

    es.addEventListener("done", () => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsgId ? { ...m, isStreaming: false } : m
        )
      );
      setIsStreaming(false);
      es.close();
    });

    es.addEventListener("error", (event) => {
      const errorEvent = event as MessageEvent;
      const errorText = errorEvent.data ?? "An error occurred.";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsgId
            ? { ...m, content: `Error: ${errorText}`, isStreaming: false, isError: true }
            : m
        )
      );
      setIsStreaming(false);
      es.close();
    });

    es.onerror = () => {
      // Connection error (network / server down)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsgId
            ? {
                ...m,
                content: m.content || "Connection error. Is the backend running?",
                isStreaming: false,
                isError: true,
              }
            : m
        )
      );
      setIsStreaming(false);
      es.close();
    };
  }, [input, isStreaming]);

  const clearChat = useCallback(() => {
    if (isStreaming) {
      eventSourceRef.current?.close();
      setIsStreaming(false);
    }
    setMessages([]);
  }, [isStreaming]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen" style={{ background: "var(--bg)" }}>
      {/* Header */}
      <header
        className="flex items-center justify-between px-6 py-3 border-b"
        style={{ borderColor: "var(--border)", background: "var(--surface)" }}
      >
        <div className="flex items-center gap-3">
          {/* Traffic light dots — terminal aesthetic */}
          <div className="flex gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-500 opacity-80" />
            <span className="w-3 h-3 rounded-full bg-yellow-400 opacity-80" />
            <span className="w-3 h-3 rounded-full bg-green-500 opacity-80" />
          </div>
          <span className="text-sm" style={{ color: "var(--text-muted)" }}>
            learnchat — LangChain / LangGraph / LangSmith
          </span>
        </div>

        <button
          onClick={clearChat}
          className="text-xs px-3 py-1 rounded border transition-colors hover:opacity-80"
          style={{
            borderColor: "var(--border)",
            color: "var(--text-muted)",
            background: "transparent",
          }}
          title="Clear chat (⌘K)"
        >
          clear
        </button>
      </header>

      {/* Chat messages */}
      <ChatWindow messages={messages} isStreaming={isStreaming} />

      {/* Input area */}
      <div
        className="border-t px-4 py-3"
        style={{ borderColor: "var(--border)", background: "var(--surface)" }}
      >
        <div
          className="flex items-end gap-3 max-w-3xl mx-auto rounded-lg border px-4 py-3"
          style={{ borderColor: "var(--border)", background: "var(--bg)" }}
        >
          <span className="text-sm pb-0.5 select-none" style={{ color: "var(--accent)" }}>
            &gt;
          </span>
          <textarea
            className="flex-1 bg-transparent resize-none outline-none text-sm leading-relaxed"
            style={{ color: "var(--text)", caretColor: "var(--accent)" }}
            placeholder="Ask about LangChain, LangGraph, or LangSmith..."
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              // Auto-resize up to ~5 lines
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
            }}
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isStreaming}
            className="text-xs px-3 py-1.5 rounded transition-all disabled:opacity-30"
            style={{
              background: "var(--accent)",
              color: "#0d1117",
              fontWeight: 600,
            }}
          >
            {isStreaming ? "..." : "send"}
          </button>
        </div>
        <p className="text-center text-xs mt-2" style={{ color: "var(--text-muted)" }}>
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
