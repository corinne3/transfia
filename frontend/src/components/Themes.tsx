import { useEffect, useState } from "react";
import { getJSON } from "../api";

type ThemeMap = {
  k: number | null;
  silhouette_by_k: Record<string, number>;
  clusters: Record<string, string[]>;
  points: { x: number; y: number; cluster: number; title: string }[];
};

const COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626", "#0891b2", "#4d7c0f", "#be185d"];

// Projection 2D des embeddings (PCA ou UMAP) + clusters (KMeans / HDBSCAN) nommés par c-TF-IDF.
export default function Themes() {
  const [data, setData] = useState<ThemeMap | null>(null);
  useEffect(() => {
    getJSON<ThemeMap>("/api/themes").then(setData);
  }, []);
  if (!data) return <p className="muted">Calcul de la carte…</p>;

  const xs = data.points.map((p) => p.x), ys = data.points.map((p) => p.y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (x: number) => 30 + ((x - minX) / (maxX - minX || 1)) * 740;
  const sy = (y: number) => 30 + ((y - minY) / (maxY - minY || 1)) * 440;

  return (
    <section className="columns">
      <svg viewBox="0 0 800 500" className="graph" style={{ flex: 2 }}>
        {data.points.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={6}
            fill={p.cluster < 0 ? "#9aa5b1" : COLORS[p.cluster % COLORS.length]} opacity={0.8}>
            <title>{p.title}</title>
          </circle>
        ))}
      </svg>
      <div className="column">
        <h3>Clusters (k = {data.k ?? "auto"})</h3>
        {Object.entries(data.clusters).map(([c, words]) => (
          <p key={c}>
            <span className="tag" style={{ background: COLORS[Number(c) % COLORS.length], color: "white" }}>{c}</span>{" "}
            {words.join(", ")}
          </p>
        ))}
        <h3>Score silhouette par k</h3>
        <p className="muted">{Object.entries(data.silhouette_by_k).map(([k, s]) => `k=${k} : ${s}`).join(" · ")}</p>
      </div>
    </section>
  );
}
