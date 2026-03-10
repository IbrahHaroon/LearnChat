"use client";

/**
 * ChatWindow.tsx — Day 5/6: The scrollable message list
 *
 * Responsibilities:
 *   - Renders the list of messages
 *   - Auto-scrolls to the bottom when new content arrives
 *   - Shows an empty state when there are no messages
 *
 * Why useEffect + scrollIntoView?
 *   React updates the DOM asynchronously. We use useEffect to run
 *   the scroll *after* the DOM has been updated with the new message.
 */

import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import type { Message } from "./types";

interface ChatWindowProps {
  messages: Message[];
  isStreaming: boolean;
}

export default function ChatWindow({ messages, isStreaming }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom whenever messages update (new token arrives, new message added)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-6 px-6">
        <div className="text-center max-w-lg">
          <p className="text-2xl mb-2" style={{ color: "var(--accent)" }}>
            LearnChat
          </p>
          <p className="text-sm mb-6" style={{ color: "var(--text-muted)" }}>
            Ask anything about LangChain, LangGraph, or LangSmith.
            <br />
            Answers are grounded in the official docs.
          </p>
          {/* Example prompts */}
          <div className="flex flex-col gap-2 text-left">
            {[
              "What is LCEL and why would I use it?",
              "How does LangGraph differ from LangChain?",
              "How do I trace my chain with LangSmith?",
              "What are the main components of a RAG pipeline?",
            ].map((prompt) => (
              <div
                key={prompt}
                className="px-4 py-2 rounded border text-xs cursor-default"
                style={{
                  borderColor: "var(--border)",
                  color: "var(--text-muted)",
                  background: "var(--surface)",
                }}
              >
                {prompt}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="max-w-3xl mx-auto flex flex-col gap-4">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {/* Invisible anchor we scroll to */}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
