"use client";

import type { Message, SourceDoc } from "./types";

interface SidePanelProps {
  sources: SourceDoc[];
  messages: Message[];
}

export default function SidePanel({ sources, messages }: SidePanelProps) {
  const conversationTurns = messages.filter((m) => m.role === "user");

  return (
    <div
      className="flex flex-col gap-4 overflow-y-auto p-4 w-80 shrink-0 border-l"
      style={{ borderColor: "var(--border)", background: "var(--surface)" }}
    >
      {/* Sources */}
      <section>
        <h2 className="text-xs font-semibold mb-3 uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
          Sources
        </h2>
        {sources.length === 0 ? (
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            No sources yet.
          </p>
        ) : (
          <div className="flex flex-col gap-3">
            {sources.map((s, i) => {
              const [namespace, page] = (s.source ?? "").split(":");
              return (
                <div
                  key={i}
                  className="rounded border p-3 flex flex-col gap-2"
                  style={{ borderColor: "var(--border)", background: "var(--bg)" }}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="px-1.5 py-0.5 rounded text-xs font-semibold"
                      style={{ background: namespaceColor(namespace), color: "#0d1117" }}
                    >
                      {namespace}
                    </span>
                    <span className="text-xs truncate" style={{ color: "var(--text)" }}>
                      {page ?? s.source}
                    </span>
                    {s.url && (
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-auto text-xs hover:opacity-70"
                        style={{ color: "var(--accent)" }}
                      >
                        ↗
                      </a>
                    )}
                  </div>

                  {/* Relevance score bar */}
                  {s.score !== null && (
                    <div className="flex items-center gap-2">
                      <div
                        className="flex-1 rounded-full overflow-hidden"
                        style={{ height: 4, background: "var(--border)" }}
                      >
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.round((s.score ?? 0) * 100)}%`,
                            background: scoreColor(s.score ?? 0),
                          }}
                        />
                      </div>
                      <span className="text-xs tabular-nums" style={{ color: "var(--text-muted)" }}>
                        {((s.score ?? 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      <div style={{ borderTop: "1px solid var(--border)" }} />

      {/* Context memory */}
      <section>
        <h2 className="text-xs font-semibold mb-3 uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
          Context Memory
        </h2>
        {conversationTurns.length === 0 ? (
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            No questions yet.
          </p>
        ) : (
          <div className="flex flex-col gap-2">
            {conversationTurns.map((m, i) => (
              <div
                key={m.id}
                className="rounded border px-3 py-2 text-xs"
                style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--text-muted)" }}
              >
                <span style={{ color: "var(--accent)" }}>Q{i + 1}</span>{" "}
                {m.content}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function namespaceColor(ns: string): string {
  switch (ns) {
    case "langchain":  return "#3fb950";
    case "langgraph":  return "#58a6ff";
    case "langsmith":  return "#d2a8ff";
    default:           return "#8b949e";
  }
}

function scoreColor(score: number): string {
  if (score >= 0.75) return "#3fb950";  // green — high relevance
  if (score >= 0.55) return "#58a6ff";  // blue  — medium
  return "#f85149";                      // red   — low
}
