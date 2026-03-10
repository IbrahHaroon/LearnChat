"use client";

/**
 * MessageBubble.tsx — Day 5/6: Individual message display
 *
 * Renders a single message (user or assistant) with:
 *   - Different styles for user vs assistant
 *   - Blinking cursor while the assistant is streaming
 *   - Source links after the assistant finishes
 *
 * Design choices:
 *   - Terminal aesthetic: monospace font, muted greens/blues, minimal
 *   - User messages: right-aligned with blue background
 *   - Assistant messages: left-aligned with dark surface background
 */

import type { Message } from "./types";
import SourceLinks from "./SourceLinks";

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div
          className="max-w-[80%] px-4 py-2.5 rounded-lg text-sm leading-relaxed"
          style={{ background: "var(--user-bubble)", color: "#e6edf3" }}
        >
          {message.content}
        </div>
      </div>
    );
  }

  // Assistant message
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="max-w-[90%] flex flex-col gap-2">
        {/* Role label */}
        <span className="text-xs px-1" style={{ color: "var(--text-muted)" }}>
          assistant
        </span>

        {/* Message content */}
        <div
          className="px-4 py-3 rounded-lg text-sm leading-relaxed border"
          style={{
            background: "var(--bot-bubble)",
            borderColor: "var(--border)",
            color: message.isError ? "#f85149" : "var(--text)",
            whiteSpace: "pre-wrap",    // Respect newlines in the response
            wordBreak: "break-word",
          }}
        >
          {message.content || (message.isStreaming ? "" : "…")}
          {/* Show blinking cursor while streaming */}
          {message.isStreaming && (
            <span className="cursor-blink" aria-hidden="true" />
          )}
        </div>

        {/* Source links — shown after streaming completes */}
        {!message.isStreaming && message.sources && message.sources.length > 0 && (
          <SourceLinks sources={message.sources} />
        )}
      </div>
    </div>
  );
}
