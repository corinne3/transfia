import { useState } from "react";
import { Check, Citation, postJSON, streamSSE } from "../api";

type Turn = {
  question: string;
  answer: string;
  steps: string[];
  citations: Citation[];
  checks: Check[];
  refused: boolean;
  traceId?: string;
};

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [espece, setEspece] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);

  const update = (fn: (t: Turn) => Turn) =>
    setTurns((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])]);

  async function ask() {
    if (!question.trim() || busy) return;
    const history = turns.slice(-3).flatMap((t) => [`user: ${t.question}`, `assistant: ${t.answer}`]);
    setTurns((prev) => [...prev, { question, answer: "", steps: [], citations: [], checks: [], refused: false }]);
    setQuestion("");
    setBusy(true);
    try {
      await streamSSE("/api/ask/stream", { question, espece, history }, (e) => {
        if (e.type === "step" && e.text) update((t) => ({ ...t, steps: [...t.steps, e.text!] }));
        if (e.type === "token") update((t) => ({ ...t, answer: t.answer + e.text }));
        if (e.type === "answer" && e.text) update((t) => ({ ...t, answer: e.text }));   // texte avec [n] ajoutés
        if (e.type === "citations") update((t) => ({ ...t, citations: e.citations }));
        if (e.type === "verification") update((t) => ({ ...t, checks: e.checks }));
        if (e.type === "refusal") update((t) => ({ ...t, answer: e.text, refused: true }));
        if (e.type === "done") update((t) => ({ ...t, traceId: e.trace_id }));
        if (e.type === "error") update((t) => ({ ...t, answer: `Erreur : ${e.text}`, refused: true }));
      });
    } finally {
      setBusy(false);
    }
  }

  async function feedback(t: Turn, rating: 1 | -1) {
    await postJSON("/api/feedback", { question: t.question, answer: t.answer, rating, trace_id: t.traceId });
  }

  return (
    <section>
      <div className="turns">
        {turns.length === 0 && (
          <p className="muted">
            Exemples : « Quels sont les facteurs de risque des mammites ? », « Pourquoi le colostrum est-il important
            pour le veau ? »
          </p>
        )}
        {turns.map((t, i) => {
          const supported = t.checks.filter((c) => c.supported).length;
          return (
            <article key={i} className="card">
              <p className="q">{t.question}</p>
              {t.steps.map((s, j) => (
                <p key={j} className="step">
                  … {s}
                </p>
              ))}
              <p className={t.refused ? "answer refused" : "answer"}>{t.answer || (busy ? "▍" : "")}</p>
              {t.checks.length > 0 && (
                <p className={supported === t.checks.length ? "badge ok" : "badge warn"}>
                  Vérification : {supported}/{t.checks.length} phrases soutenues par leurs sources
                </p>
              )}
              {t.checks
                .filter((c) => !c.supported)
                .map((c, j) => (
                  <p key={j} className="unsupported">
                    ⚠️ Non vérifiée : {c.sentence}
                  </p>
                ))}
              {t.citations.length > 0 && (
                <ol className="citations">
                  {t.citations.map((c) => (
                    <li key={c.n} value={c.n}>
                      <strong>{c.authors.slice(0, 2).join(", ")}</strong> ({c.year ?? "s.d."}) — {c.title}
                      {c.page ? `, p. ${c.page}` : ""}{" "}
                      {c.doi_url && (
                        <a href={c.doi_url} target="_blank" rel="noreferrer">
                          DOI
                        </a>
                      )}
                      <div className="excerpt">« {c.excerpt} »</div>
                    </li>
                  ))}
                </ol>
              )}
              {t.answer && !busy && (
                <div className="feedback">
                  Utile ? <button onClick={() => feedback(t, 1)}>👍</button>
                  <button onClick={() => feedback(t, -1)}>👎</button>
                </div>
              )}
            </article>
          );
        })}
      </div>
      <div className="composer">
        <select value={espece} onChange={(e) => setEspece(e.target.value)}>
          <option value="">Toutes espèces</option>
          {["bovin", "porcin", "ovin", "caprin", "volaille", "poisson", "lapin"].map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <input
          value={question}
          placeholder="Pose une question sur le corpus…"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button className="primary" onClick={ask} disabled={busy}>
          {busy ? "…" : "Demander"}
        </button>
      </div>
    </section>
  );
}
