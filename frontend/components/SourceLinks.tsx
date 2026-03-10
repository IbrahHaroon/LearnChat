"use client";

/**
 * SourceLinks.tsx — Day 6: Cited source documents
 *
 * Displays the docs chunks that were retrieved to generate the answer.
 * This is the "Retrieval" part of RAG made visible to the user.
 *
 * Why show sources?
 *   - Transparency: users can verify the answer against the actual docs
 *   - Trust: grounded answers with citations feel more reliable
 *   - Learning: clicking through to source docs is educational
 *
 * The score is the cosine similarity between the question embedding
 * and the chunk embedding (0–1, higher = more relevant).
 */

import type { SourceDoc } from "./types";

interface SourceLinksProps {
  sources: SourceDoc[];
}

export default function SourceLinks({ sources }: SourceLinksProps) {
  // Filter out any sources without a URL
  const valid = sources.filter((s) => s.url);

  if (valid.length === 0) return null;

  return (
    <div className="flex flex-col gap-1 mt-1">
      <span className="text-xs px-1" style={{ color: "var(--text-muted)" }}>
        sources
      </span>
      <div className="flex flex-wrap gap-2">
        {valid.map((source, i) => (
          <SourceChip key={i} source={source} index={i + 1} />
        ))}
      </div>
    </div>
  );
}

function SourceChip({
  source,
  index,
}: {
  source: SourceDoc;
  index: number;
}) {
  const label = source.source ?? `source ${index}`;
  const [namespace, page] = label.split(":");

  return (
    <a
      href={source.url ?? "#"}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-1.5 px-2.5 py-1 rounded border text-xs transition-colors hover:opacity-80"
      style={{
        borderColor: "var(--border)",
        background: "var(--surface)",
        color: "var(--accent)",
        textDecoration: "none",
      }}
      title={`Similarity: ${source.score?.toFixed(3) ?? "n/a"}`}
    >
      {/* Namespace badge (langchain / langgraph / langsmith) */}
      <span
        className="px-1 rounded text-xs"
        style={{
          background: namespaceColor(namespace),
          color: "#0d1117",
          fontWeight: 600,
        }}
      >
        {namespace}
      </span>
      {/* Page name */}
      <span style={{ color: "var(--text-muted)" }}>{page ?? label}</span>
      {/* External link icon */}
      <span style={{ color: "var(--text-muted)", fontSize: "0.6rem" }}>↗</span>
    </a>
  );
}

function namespaceColor(ns: string): string {
  switch (ns) {
    case "langchain":  return "#3fb950";   // green
    case "langgraph":  return "#58a6ff";   // blue
    case "langsmith":  return "#d2a8ff";   // purple
    default:           return "#8b949e";   // grey
  }
}
