import { useState } from "react";
import { postJSON } from "../api";

type Hit = {
  rank: number;
  score: number;
  article_id: string;
  title: string;
  label: string;
  doi: string | null;
  section: string;
  page: number | null;
  text: string;
  species: string[];
  themes: string[];
};

const MODES = ["hybrid", "dense", "bm25"] as const;

// Compare les 3 modes côte à côte : c'est la meilleure façon de "voir" lexical vs sémantique.
export default function Search() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Record<string, Hit[]>>({});
  const [busy, setBusy] = useState(false);

  async function run() {
    if (!query.trim()) return;
    setBusy(true);
    try {
      const entries = await Promise.all(
        MODES.map(async (mode) => [mode, await postJSON<Hit[]>("/api/search", { query, mode, k: 5 })] as const)
      );
      setResults(Object.fromEntries(entries));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <div className="composer">
        <input
          value={query}
          placeholder="ex. émissions de méthane, sevrage des porcelets…"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <button className="primary" onClick={run} disabled={busy}>
          Comparer les 3 modes
        </button>
      </div>
      <div className="columns">
        {MODES.map((mode) => (
          <div key={mode} className="column">
            <h3>{mode === "hybrid" ? "Hybride + rerank" : mode === "dense" ? "Dense (embeddings)" : "BM25 (lexical)"}</h3>
            {(results[mode] ?? []).map((h) => (
              <div key={h.rank} className="card small">
                <div className="muted">
                  #{h.rank} · score {h.score} · {h.label}
                </div>
                <strong>{h.title}</strong>
                <p>{h.text.slice(0, 260)}…</p>
                <div className="tags">
                  {[...h.species, ...h.themes].map((t) => (
                    <span key={t} className="tag">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
