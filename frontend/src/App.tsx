import { useEffect, useState } from "react";
import { getJSON } from "./api";
import Chat from "./components/Chat";
import Search from "./components/Search";
import Generator from "./components/Generator";
import Agent from "./components/Agent";
import GraphView from "./components/GraphView";
import Themes from "./components/Themes";
import Eval from "./components/Eval";

const TABS = [
  { id: "chat", label: "Questions sourcées", hint: "RAG (F5)" },
  { id: "search", label: "Recherche", hint: "Hybride + rerank (F4)" },
  { id: "generate", label: "Générateur", hint: "Sorties structurées (F7)" },
  { id: "agent", label: "Assistant", hint: "Agent + tools + skills (F8)" },
  { id: "graph", label: "Graphe", hint: "GraphRAG (F6)" },
  { id: "themes", label: "Carte des thèmes", hint: "Clustering (F3)" },
  { id: "eval", label: "Évaluation", hint: "Banc d'éval (F11, F14)" },
] as const;

type TabId = (typeof TABS)[number]["id"];
type Health = { status: string; build?: string; config: string; llm: string; chunks: number; skills: string[] };

export default function App() {
  const [tab, setTab] = useState<TabId>("chat");
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getJSON<Health>("/api/health").then(setHealth).catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="layout">
      <header>
        <div>
          <h1>TransfIA</h1>
          <p className="muted">De la recherche en élevage aux contenus pédagogiques sourcés</p>
        </div>
        <div className="status">
          {health ? (
            <span>
              <span className="dot ok" /> {health.llm} · {health.chunks} passages · config {health.config} · {health.build ?? "version ?"}
            </span>
          ) : (
            <span>
              <span className="dot ko" /> {error ? "API injoignable : lance `transfia serve`" : "connexion…"}
            </span>
          )}
        </div>
      </header>
      <nav>
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
            {t.label}
            <small>{t.hint}</small>
          </button>
        ))}
      </nav>
      <main>
        {tab === "chat" && <Chat />}
        {tab === "search" && <Search />}
        {tab === "generate" && <Generator />}
        {tab === "agent" && <Agent />}
        {tab === "graph" && <GraphView />}
        {tab === "themes" && <Themes />}
        {tab === "eval" && <Eval />}
      </main>
      <footer className="muted">
        Contenus générés par IA à partir des sources citées : à vérifier avant diffusion. Corpus : INRAE Productions
        Animales (CC BY 4.0).
      </footer>
    </div>
  );
}
