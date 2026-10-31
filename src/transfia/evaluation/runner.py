"""Banc d'évaluation (F11 + F14) : comparer des configurations avec des CHIFFRES.

- ``eval_retrieval`` : dense vs BM25 vs hybride (± reranking) -> Recall@k, MRR, nDCG + latence.
- ``eval_rag``       : réponses complètes -> fidélité, F1, taux de refus correct, tokens, latence,
                       énergie, et notes du jury de LLM (optionnel).
- Chaque rapport enregistre l'EMPREINTE de la configuration et les VERSIONS des prompts.
- ``compare_runs``   : test de permutation apparié entre deux runs sur les mêmes questions.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from transfia.core.prompts import all_versions
from transfia.evaluation.dataset import EvalItem
from transfia.evaluation.metrics import (dedupe, hit_rate_at_k, ndcg_at_k, paired_permutation_test,
                                         reciprocal_rank, recall_at_k, summarize, token_f1)
from transfia.observability.metrics import EnergyMeter, percentile


def _relevant(item: EvalItem, level: str) -> set[str]:
    return set(item.source_chunk_ids if level == "chunk" and item.source_chunk_ids else item.source_article_ids)


def eval_retrieval(engine, items: list[EvalItem], configs: dict[str, dict] | None = None, k: int = 5) -> dict:
    """configs : {"nom": {"mode": "hybrid", "rerank": True}, ...}"""
    configs = configs or {
        "bm25": {"mode": "bm25", "rerank": False},
        "dense": {"mode": "dense", "rerank": False},
        "hybrid": {"mode": "hybrid", "rerank": False},
        "hybrid+rerank": {"mode": "hybrid", "rerank": True},
    }
    items = [it for it in items if it.source_article_ids or it.source_chunk_ids]
    report = {}
    for name, cfg in configs.items():
        rows, lat = [], []
        for it in items:
            t0 = time.perf_counter()
            hits = engine.search(it.question, k=k, **cfg)
            lat.append(time.perf_counter() - t0)
            level = "chunk" if it.source_chunk_ids else "article"
            ids = [h.chunk.id for h in hits] if level == "chunk" else dedupe([h.chunk.article_id for h in hits])
            rel = _relevant(it, level)
            rows.append({"id": it.id, "recall": recall_at_k(ids, rel, k), "hit": hit_rate_at_k(ids, rel, k),
                         "mrr": reciprocal_rank(ids, rel), "ndcg": ndcg_at_k(ids, rel, k)})
        report[name] = {
            "recall@k": summarize([r["recall"] for r in rows]), "hit@k": summarize([r["hit"] for r in rows]),
            "mrr": summarize([r["mrr"] for r in rows]), "ndcg@k": summarize([r["ndcg"] for r in rows]),
            "latency_p50_ms": round(percentile(lat, 50) * 1000, 1), "latency_p95_ms": round(percentile(lat, 95) * 1000, 1),
            "rows": rows,
        }
    return {"k": k, "n_items": len(items), "configs": report}


def eval_rag(rag, items: list[EvalItem], settings, panel=None, embedder=None, label: str = "run") -> dict:
    obs = settings.observability
    meter = EnergyMeter(obs.energy, obs.cpu_watts, obs.grid_gco2_per_kwh)
    rows = []
    with meter.measure("eval_rag") as m:
        for it in items:
            t0 = time.perf_counter()
            ans = rag.answer(it.question)
            latency = time.perf_counter() - t0
            expected_refusal = not it.source_article_ids
            row = {
                "id": it.id, "question": it.question, "answer": ans.text, "refused": ans.refused,
                "refusal_correct": float(ans.refused == expected_refusal),
                "faithfulness": ans.faithfulness if ans.faithfulness is not None else float("nan"),
                "f1": token_f1(ans.text, it.reference_answer) if it.reference_answer else float("nan"),
                "source_hit": float(bool({c.article_id for c in ans.citations} & set(it.source_article_ids)))
                if it.source_article_ids else float("nan"),
                "latency_s": latency, "tokens_in": ans.usage.get("prompt_tokens", 0),
                "tokens_out": ans.usage.get("completion_tokens", 0), "model": ans.model,
            }
            if embedder is not None and not ans.refused:
                row["relevance"] = float(embedder.embed_query(it.question) @ embedder.embed_query(ans.text))
            if panel is not None:
                ctx = "\n".join(f"[{i + 1}] {h.chunk.text[:600]}" for i, h in enumerate(ans.hits))
                row["judge"] = panel.score(question=it.question, answer=ans.text, context=ctx,
                                           reference=it.reference_answer)
                row["judge_mean"] = row["judge"]["mean"] if row["judge"]["mean"] is not None else float("nan")
            rows.append(row)
    lat = [r["latency_s"] for r in rows]
    summary = {k: summarize([r[k] for r in rows if k in r]) for k in
               ("faithfulness", "f1", "source_hit", "refusal_correct", "relevance", "judge_mean")}
    summary.update({
        "latency_p50_s": round(percentile(lat, 50), 3), "latency_p95_s": round(percentile(lat, 95), 3),
        "tokens_in_mean": round(sum(r["tokens_in"] for r in rows) / max(len(rows), 1), 1),
        "tokens_out_mean": round(sum(r["tokens_out"] for r in rows) / max(len(rows), 1), 1),
        "energy": m.as_dict(),
        "wh_per_question": round(m.kwh * 1000 / max(len(rows), 1), 5),
    })
    summary["llm_cache"] = settings.llm.cache          # True = coûts non représentatifs
    return {"label": label, "date": datetime.now().isoformat(timespec="seconds"), "config": settings.name,
            "config_fingerprint": settings.fingerprint, "llm": settings.llm.model,
            "prompts": all_versions(), "n_items": len(rows), "summary": summary, "rows": rows}


def compare_runs(run_a: dict, run_b: dict, metric: str = "faithfulness") -> dict:
    a = {r["id"]: r.get(metric) for r in run_a["rows"]}
    b = {r["id"]: r.get(metric) for r in run_b["rows"]}
    common = [i for i in a if i in b and a[i] == a[i] and b[i] == b[i] and a[i] is not None and b[i] is not None]
    if len(common) < 2:
        return {"metric": metric, "n": len(common), "p_value": None}
    xa, xb = [a[i] for i in common], [b[i] for i in common]
    p = paired_permutation_test(xa, xb)
    return {"metric": metric, "n": len(common), "mean_a": round(sum(xa) / len(xa), 4),
            "mean_b": round(sum(xb) / len(xb), 4), "p_value": round(p, 4),
            "verdict": "différence significative (p < 0,05)" if p < 0.05 else "pas de différence démontrée"}


def save_report(report: dict, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}-{name}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = directory / f"{stamp}-{name}.md"
    md.write_text(to_markdown(report), encoding="utf-8")
    return path


def refusal_breakdown(rows: list[dict]) -> dict:
    """Matrice de confusion du REFUS. Les 2 erreurs n'ont pas le même coût :
    - refus à tort (question couverte, mais refusée) : utilisateur frustré ;
    - acceptation à tort (question hors corpus, mais réponse donnée) : risque d'HALLUCINATION, plus grave."""
    inside = [r for r in rows if "source_hit" in r and r["source_hit"] == r["source_hit"]]      # pas NaN
    ids_in = {r["id"] for r in inside}
    outside = [r for r in rows if r["id"] not in ids_in]
    return {
        "couvertes": len(inside), "refus_a_tort": [r["id"] for r in inside if r["refused"]],
        "hors_corpus": len(outside), "acceptations_a_tort": [r["id"] for r in outside if not r["refused"]],
    }


def split_summary(rows: list[dict]) -> dict:
    """Coût mesuré sur les questions RÉPONDUES seulement (un refus avant le LLM coûte ~0 : vu en vrai,
    les mélanger fait baisser artificiellement la latence et les tokens moyens)."""
    answered = [r for r in rows if not r["refused"]]
    lat = [r["latency_s"] for r in answered]
    if not answered:
        return {}
    return {"n": len(answered), "latency_p50_s": round(percentile(lat, 50), 3),
            "latency_p95_s": round(percentile(lat, 95), 3),
            "tokens_in_mean": round(sum(r["tokens_in"] for r in answered) / len(answered), 1),
            "tokens_out_mean": round(sum(r["tokens_out"] for r in answered) / len(answered), 1)}


def to_markdown(report: dict) -> str:
    lines = [f"# Rapport d'évaluation — {report.get('label', '')}", ""]
    if "configs" in report:
        meta = " — ".join(x for x in (report.get("date", ""), report.get("dataset", ""), report.get("note", "")) if x)
        lines += [f"k = {report['k']}, {report['n_items']} questions" + (f" — {meta}" if meta else ""), "",
                  "| Config | Recall@k | Hit@k | MRR | nDCG@k | p50 (ms) |", "|---|---|---|---|---|---|"]
        for name, r in report["configs"].items():
            lines.append(f"| {name} | {r['recall@k']['mean']} | {r['hit@k']['mean']} | {r['mrr']['mean']} | "
                         f"{r['ndcg@k']['mean']} | {r['latency_p50_ms']} |")
    else:
        s = report["summary"]
        lines += [f"Config : `{report['config']}` ({report['config_fingerprint']}) — LLM : `{report['llm']}` — "
                  f"{report['n_items']} questions"
                  + "".join(f" — {report[k]}" for k in ("date", "dataset", "note") if report.get(k)),
                  "", "| Métrique | Moyenne | IC 95 % |", "|---|---|---|"]
        for k in ("faithfulness", "f1", "source_hit", "refusal_correct", "relevance", "judge_mean"):
            if s.get(k, {}).get("mean") is not None:
                lines.append(f"| {k} | {s[k]['mean']} | {s[k].get('ci95')} |")
        lines += ["", f"- Latence p50 / p95 : {s['latency_p50_s']} s / {s['latency_p95_s']} s",
                  f"- Tokens moyens entrée / sortie : {s['tokens_in_mean']} / {s['tokens_out_mean']}",
                  f"- Énergie estimée : {s['wh_per_question']} Wh par question ({s['energy']['method']})"]
        if s.get("llm_cache", True) and "llm_cache" in s:
            lines.append("- ⚠️ cache LLM ACTIVÉ pendant ce run : latence, tokens et énergie peuvent venir du cache "
                         "(non représentatifs)")
        rows = report.get("rows", [])
        if rows:
            b = refusal_breakdown(rows)
            lines += ["", "**Refus (matrice de confusion)**",
                      f"- Questions couvertes : {b['couvertes']} — refusées à tort : {len(b['refus_a_tort'])} "
                      f"{b['refus_a_tort'] or ''}",
                      f"- Questions hors corpus : {b['hors_corpus']} — **répondues à tort** (risque d'hallucination) : "
                      f"{len(b['acceptations_a_tort'])} {b['acceptations_a_tort'] or ''}"]
            a = split_summary(rows)
            if a and len(a) and a["n"] != len(rows):
                lines += [f"- Coût sur les {a['n']} questions RÉPONDUES seulement : latence p50 / p95 "
                          f"{a['latency_p50_s']} s / {a['latency_p95_s']} s ; tokens {a['tokens_in_mean']} / "
                          f"{a['tokens_out_mean']} (les refus, quasi gratuits, faussent la moyenne générale)"]
            wrong = set(b["refus_a_tort"]) | set(b["acceptations_a_tort"])
            for r in rows:
                if r["id"] in wrong:
                    lines.append(f"  - {r['id']} : « {r['question'][:80]} » → "
                                 f"{'refusée' if r['refused'] else r['answer'][:100].replace(chr(10), ' ')}")
    return "\n".join(lines)
