import { useEffect, useMemo, useState } from "react";
import { getJSON } from "../api";

type Node = { id: string; label: string; group: string; community: number; size: number };
type Link = { source: string; target: string; weight: number; relation: string };
type Graph = { nodes: Node[]; links: Link[] };
type P = { x: number; y: number; vx: number; vy: number };

const W = 900;
const H = 560;
// marges : les étiquettes sont écrites AU-DESSUS des nœuds ; sans marge, celles du haut sont coupées (vu en vrai)
const PAD_X = 70;
const PAD_TOP = 45;
const PAD_BOTTOM = 25;
const COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2", "#4d7c0f", "#be185d"];

// Mise en page "force-directed" codée à la main (sans bibliothèque) :
// - répulsion entre tous les nœuds (comme des charges électriques),
// - attraction le long des liens (comme des ressorts),
// - légère gravité vers le centre. On itère jusqu'à stabilisation.
function layout(g: Graph, iterations = 300): Record<string, P> {
  const pos: Record<string, P> = {};
  g.nodes.forEach((n, i) => {
    const a = (2 * Math.PI * i) / g.nodes.length;
    pos[n.id] = { x: W / 2 + 200 * Math.cos(a), y: H / 2 + 200 * Math.sin(a), vx: 0, vy: 0 };
  });
  for (let it = 0; it < iterations; it++) {
    const ids = Object.keys(pos);
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const a = pos[ids[i]], b = pos[ids[j]];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = Math.max(dx * dx + dy * dy, 25);
        const f = 900 / d2;            // répulsion plus douce : moins de nœuds plaqués contre les bords
        a.vx += f * dx; a.vy += f * dy; b.vx -= f * dx; b.vy -= f * dy;
      }
    }
    for (const l of g.links) {
      const a = pos[l.source], b = pos[l.target];
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const f = 0.004 * Math.min(l.weight, 5);
      a.vx += f * dx; a.vy += f * dy; b.vx -= f * dx; b.vy -= f * dy;
    }
    for (const p of Object.values(pos)) {
      p.vx += (W / 2 - p.x) * 0.006; p.vy += (H / 2 - p.y) * 0.006;   // gravité vers le centre plus forte
      p.x = Math.min(W - PAD_X, Math.max(PAD_X, p.x + p.vx * 0.5));
      p.y = Math.min(H - PAD_BOTTOM, Math.max(PAD_TOP, p.y + p.vy * 0.5));
      p.vx *= 0.6; p.vy *= 0.6;
    }
  }
  return pos;
}

export default function GraphView() {
  const [graph, setGraph] = useState<Graph | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    getJSON<Graph>("/api/graph").then(setGraph);
  }, []);

  const pos = useMemo(() => (graph ? layout(graph) : {}), [graph]);
  if (!graph) return <p className="muted">Chargement du graphe…</p>;

  const neighbors = new Set(
    graph.links.filter((l) => l.source === selected || l.target === selected).flatMap((l) => [l.source, l.target])
  );

  return (
    <section>
      <p className="muted">
        {graph.nodes.length} concepts, {graph.links.length} relations. Couleur = communauté (algorithme de Louvain).
        Clique sur un concept pour voir ses voisins.
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="graph" style={{ overflow: "visible" }}>
        {graph.links.map((l, i) => {
          const a = pos[l.source], b = pos[l.target];
          if (!a || !b) return null;
          const on = selected && (l.source === selected || l.target === selected);
          return (
            <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} strokeWidth={Math.min(1 + l.weight, 5)}
              className={on ? "edge on" : "edge"}>
              <title>{l.relation} (poids {l.weight})</title>
            </line>
          );
        })}
        {graph.nodes.map((n) => {
          const p = pos[n.id];
          const dim = selected && n.id !== selected && !neighbors.has(n.id);
          return (
            <g key={n.id} transform={`translate(${p.x},${p.y})`} opacity={dim ? 0.2 : 1}
              onClick={() => setSelected(selected === n.id ? null : n.id)} style={{ cursor: "pointer" }}>
              <circle r={5 + Math.min(n.size, 10)} fill={COLORS[(n.community + COLORS.length) % COLORS.length]} />
              <text y={-10 - Math.min(n.size, 10)} textAnchor="middle" className="node-label">
                {n.label}
              </text>
            </g>
          );
        })}
      </svg>
    </section>
  );
}
