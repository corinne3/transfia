// Appels à l'API FastAPI. Le streaming utilise fetch + ReadableStream : on lit la réponse
// morceau par morceau et on découpe les événements SSE ("data: {...}\n\n").

export type Citation = {
  n: number;
  article_id: string;
  title: string;
  authors: string[];
  year: number | null;
  doi: string | null;
  doi_url?: string | null;
  page: number | null;
  excerpt: string;
};

export type Check = { sentence: string; cited: number[]; supported: boolean; score: number; method: string };

export type StreamEvent =
  | { type: "step"; text?: string; step?: AgentStep }
  | { type: "token"; text: string }
  | { type: "answer"; text: string; citations_added_post_hoc?: number }
  | { type: "sources"; queries: string[]; hits: { article_id: string; title: string; score: number }[] }
  | { type: "citations"; citations: Citation[] }
  | { type: "verification"; checks: Check[] }
  | { type: "refusal"; text: string }
  | { type: "final"; text: string; reason: string }
  | { type: "done"; trace_id?: string }
  | { type: "error"; text: string };

export type AgentStep = {
  kind: "thought" | "tool" | "final" | "error";
  content: string;
  tool?: string | null;
  arguments?: Record<string, unknown> | null;
  observation?: string | null;
};

export async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export async function streamSSE(url: string, body: unknown, onEvent: (e: StreamEvent) => void): Promise<void> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok || !r.body) throw new Error(`${r.status} ${await r.text()}`);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)) as StreamEvent);
    }
  }
}
