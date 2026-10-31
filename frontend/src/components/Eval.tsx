import { useEffect, useState } from "react";
import { getJSON } from "../api";

type Metric = { mean: number | null; ci95?: [number, number]; n: number };
type Report = {
  file: string;
  label: string;
  date: string;
  llm: string;
  kind: "retrieval" | "rag";
  dataset?: string;
  note?: string;
  n_items?: number;
  config?: string;
  fingerprint?: string;
  summary?: Record<string, Metric | number | boolean | Record<string, unknown>>;
  configs?: Record<string, Record<string, Metric | number>>;
};

const num = (m: unknown) =>
  m && typeof m === "object" && "mean" in (m as Metric) ? (m as Metric).mean ?? "–" : String(m ?? "–");
const ci = (m: unknown) =>
  m && typeof m === "object" && (m as Metric).ci95 ? `[${(m as Metric).ci95!.join(" ; ")}]` : "";

// Date lisible, à l'heure du PC qui a lancé l'évaluation (celle de ta machine).
const when = (iso: string) => {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
};

const RAG_METRICS: [string, string][] = [
  ["faithfulness", "Fidélité (phrases soutenues)"],
  ["refusal_correct", "Refus corrects"],
  ["source_hit", "Bonne source citée"],
  ["f1", "F1 vs réponse de référence"],
  ["relevance", "Pertinence (embeddings)"],
  ["judge_mean", "Jury de LLM"],
];

// Les rapports viennent de `transfia eval retrieval` et `transfia eval rag` (fichiers data/eval/*.json),
// du plus récent au plus ancien. Astuce : `--note "ce que je teste"` pour retrouver chaque manip.
export default function Eval() {
  const [reports, setReports] = useState<Report[]>([]);
  useEffect(() => {
    getJSON<Report[]>("/api/eval/reports").then(setReports);
  }, []);

  if (!reports.length)
    return <p className="muted">Aucun rapport : lance `transfia eval retrieval` puis `transfia eval rag`.</p>;

  return (
    <section>
      <p className="muted">
        Du plus récent au plus ancien. Chaque carte = une commande <code>transfia eval</code> (date et heure de ton PC).
        Ajoute <code>--note "…"</code> à la commande pour noter ce que tu testais.
      </p>
      {reports.map((r) => (
        <div key={r.file} className="card">
          <h3>
            {r.kind === "retrieval" ? "🔎 Recherche" : "💬 Réponses (RAG)"} — {r.label}{" "}
            <small className="muted">{when(r.date)}</small>
          </h3>
          <p className="muted">
            Jeu : {r.dataset ?? "?"} ({r.n_items ?? "?"} questions) · LLM : {r.llm} · config : {r.config ?? "?"}
            {r.fingerprint ? ` (${r.fingerprint})` : ""} · fichier : {r.file}
            {r.note ? <><br />📝 {r.note}</> : null}
          </p>
          {r.configs && (
            <table>
              <thead>
                <tr><th>Mode de recherche</th><th>Hit@5</th><th>MRR</th><th>nDCG@5</th><th>p50 (ms)</th></tr>
              </thead>
              <tbody>
                {Object.entries(r.configs).map(([name, c]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{num(c["hit@k"] ?? c["recall@k"])} <small className="muted">{ci(c["hit@k"] ?? c["recall@k"])}</small></td>
                    <td>{num(c["mrr"])} <small className="muted">{ci(c["mrr"])}</small></td>
                    <td>{num(c["ndcg@k"])}</td><td>{num(c["latency_p50_ms"])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {r.summary && !r.configs && (
            <table>
              <thead>
                <tr><th>Métrique</th><th>Moyenne</th><th>IC 95 %</th></tr>
              </thead>
              <tbody>
                {RAG_METRICS.filter(([k]) => r.summary![k] !== undefined && num(r.summary![k]) !== "–").map(([k, label]) => (
                  <tr key={k}><td>{label}</td><td>{num(r.summary![k])}</td><td>{ci(r.summary![k])}</td></tr>
                ))}
                <tr><td>Latence p50 / p95 (s)</td><td colSpan={2}>{num(r.summary["latency_p50_s"])} / {num(r.summary["latency_p95_s"])}</td></tr>
                <tr><td>Énergie (Wh / question)</td><td colSpan={2}>{num(r.summary["wh_per_question"])}</td></tr>
                {r.summary["llm_cache"] === true && (
                  <tr><td colSpan={3} className="warn">⚠️ cache LLM actif : latence et énergie non représentatives</td></tr>
                )}
              </tbody>
            </table>
          )}
        </div>
      ))}
    </section>
  );
}
