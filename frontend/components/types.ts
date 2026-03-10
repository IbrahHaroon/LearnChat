/**
 * Shared TypeScript types used across components.
 * Keeping them in one file avoids circular imports.
 */

export interface SourceDoc {
  source: string | null;
  url: string | null;
  score: number | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceDoc[];
  isStreaming?: boolean;
  isError?: boolean;
}
