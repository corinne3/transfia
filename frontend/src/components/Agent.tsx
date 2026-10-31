import { useState } from "react";
import { AgentStep, streamSSE } from "../api";

// Affiche la boucle ReAct en direct : pensée -> appel d'outil -> observation -> ... -> réponse finale.
export default function Agent() {
  const [task, setTask] = useState("");
  const [mode, setMode] = useState("native");
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [busy, setBusy] = useState(false);

  async function run() {
    if (!task.trim()) return;
    setSteps([]);
    setBusy(true);
    try {
      await streamSSE("/api/agent/stream", { task, mode }, (e) => {
        if (e.type === "step" && e.step) setSteps((s) => [...s, e.step!]);
        if (e.type === "error") setSteps((s) => [...s, { kind: "error", content: e.text }]);
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <div className="composer">
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="native">Tool calling natif</option>
          <option value="text">ReAct textuel</option>
        </select>
        <input
          value={task}
          placeholder="ex. Prépare 3 questions de QCM niveau bac pro sur le sevrage des porcelets"
          onChange={(e) => setTask(e.target.value)}
        />
        <button className="primary" onClick={run} disabled={busy}>
          {busy ? "L'agent travaille…" : "Lancer"}
        </button>
      </div>
      <div className="timeline">
        {steps.map((s, i) => (
          <div key={i} className={`step-card ${s.kind}`}>
            {s.kind === "thought" && <p>💭 {s.content}</p>}
            {s.kind === "tool" && (
              <>
                <p>
                  🔧 <code>{s.tool}</code>({JSON.stringify(s.arguments)})
                </p>
                <details>
                  <summary>Observation</summary>
                  <pre>{s.observation}</pre>
                </details>
              </>
            )}
            {s.kind === "final" && <pre className="output">✅ {s.content}</pre>}
            {s.kind === "error" && <p className="unsupported">{s.content}</p>}
          </div>
        ))}
      </div>
    </section>
  );
}
