import { useState } from "react";
import { postJSON } from "../api";

type Result = {
  kind: string;
  content: Record<string, unknown>;
  markdown: string;
  attempts: number;
  readability: { score: number; cible_min: number; ok: boolean; interpretation: string };
  warnings: string[];
};

export default function Generator() {
  const [kind, setKind] = useState("quiz");
  const [niveau, setNiveau] = useState("bts");
  const [topic, setTopic] = useState("");
  const [res, setRes] = useState<Result | null>(null);
  const [view, setView] = useState<"md" | "json">("md");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    if (!topic.trim()) return;
    setBusy(true);
    setError("");
    try {
      setRes(await postJSON<Result>("/api/generate", { kind, topic, niveau }));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <div className="composer">
        <select value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="quiz">QCM</option>
          <option value="fiche">Fiche de synthèse</option>
          <option value="cas_pratique">Cas pratique</option>
          <option value="glossaire">Glossaire</option>
        </select>
        <select value={niveau} onChange={(e) => setNiveau(e.target.value)}>
          <option value="grand_public">Grand public</option>
          <option value="bac_pro">Bac pro</option>
          <option value="bts">BTS</option>
          <option value="master">Master</option>
        </select>
        <input value={topic} placeholder="Thème, ex. le colostrum chez le veau" onChange={(e) => setTopic(e.target.value)} />
        <button className="primary" onClick={run} disabled={busy}>
          {busy ? "Génération… (lent sur CPU)" : "Générer"}
        </button>
      </div>
      {error && <p className="unsupported">{error}</p>}
      {res && (
        <div className="card">
          <p className="muted">
            {res.attempts} tentative(s) · lisibilité {res.readability.score} ({res.readability.interpretation}, cible ≥{" "}
            {res.readability.cible_min})
          </p>
          {res.warnings.map((w) => (
            <p key={w} className="unsupported">
              ⚠️ {w}
            </p>
          ))}
          <div className="toggle">
            <button className={view === "md" ? "tab active" : "tab"} onClick={() => setView("md")}>
              Lecture
            </button>
            <button className={view === "json" ? "tab active" : "tab"} onClick={() => setView("json")}>
              JSON validé (schéma Pydantic)
            </button>
          </div>
          <pre className="output">{view === "md" ? res.markdown : JSON.stringify(res.content, null, 2)}</pre>
          <button onClick={() => navigator.clipboard.writeText(res.markdown)}>Copier le Markdown</button>
        </div>
      )}
    </section>
  );
}
