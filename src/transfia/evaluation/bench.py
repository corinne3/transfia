"""Banc d'essai (phase 1d / phase 2) : rejoue les tests clés de TransfIA et enregistre des chiffres COMPARABLES.

Pourquoi : pour comparer petit et gros modèle, il faut les MÊMES questions, les MÊMES tâches, les MÊMES mesures.
Une commande, une configuration :
    transfia bench --label petit                              (config/local.yaml, qwen2.5:1.5b)
    transfia --config config/local-3b.yaml bench --label 3b
    transfia bench compare data/bench/<petit>.json data/bench/<3b>.json

Scénarios (chacun chronométré, avec son énergie estimée) :
1. rag        : éval RAG sur le jeu synthétique (cache LLM désactivé) : fidélité, refus, hallucinations hors corpus
2. quiz       : 3 QCM sur 3 thèmes du corpus : questions gardées, lettres corrigées, ambiguïtés, réparations
3. glossaire  : termes absents de leur source (définis de mémoire)
4. agent      : 3 tâches : l'agent utilise-t-il ses outils, ou faut-il le plan B ?
5. editorial  : multi-agents : tours, problèmes, élagage, alertes finales
6. plan       : (option --full, long) plan-and-execute : livrable cité ou non
Chaque scénario capture ses erreurs : une panne n'arrête pas le banc, elle est enregistrée.
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from transfia.observability.metrics import EnergyMeter

QUIZ_TOPICS = ["réduction du méthane entérique", "qualité du colostrum chez le veau",
               "distribution des fourrages aux chèvres"]
AGENT_TASKS = ["D'où vient le méthane des ruminants ?", "Que dit le corpus sur le colostrum ?",
               "Prépare un QCM niveau bts sur la sélection génomique des bovins"]
WARNING_KINDS = {"corrigée": "lettres corrigées", "ambiguë": "ambiguës", "reprend les mêmes": "doublons",
                 "réparé": "réparations réussies", "réparation essayée": "réparations ratées",
                 "retirées": "questions retirées", "À REVOIR": "quiz à revoir", "ABSENT": "termes absents"}


def _count(warnings: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for w in warnings:
        for key, name in WARNING_KINDS.items():
            if key in w:
                out[name] = out.get(name, 0) + 1
    return out


class Bench:
    def __init__(self, ctx, dataset: Path | None = None, full: bool = False, log=print,
                 only: list[str] | None = None, max_questions: int = 0):
        self.ctx = ctx
        self.only = only or []              # ex. ["rag"] : utile avec un quota de requêtes (API Gemini gratuite)
        self.max_questions = max_questions  # 0 = toutes les questions du jeu RAG
        self.dataset = dataset
        self.full = full
        self.log = log
        obs = ctx.settings.observability
        self.meter = EnergyMeter(obs.energy, obs.cpu_watts, obs.grid_gco2_per_kwh)

    def _run(self, name: str, fn) -> dict:
        self.log(f"▶ {name}…")
        with self.meter.measure(name) as m:
            try:
                res = fn()
                res["ok"] = True
            except Exception as exc:                      # un scénario en panne n'arrête pas le banc
                res = {"ok": False, "erreur": f"{type(exc).__name__}: {exc}",
                       "trace": traceback.format_exc()[-800:]}
        res["secondes"] = round(m.seconds, 1)
        res["wh"] = round(m.kwh * 1000, 4)
        self.log(f"  ↳ {name} : {res['secondes']} s" + ("" if res["ok"] else f" — ERREUR {res['erreur'][:120]}"))
        return res

    # ------------------------------------------------------------------ scénarios
    def rag(self) -> dict:
        from transfia.evaluation.dataset import is_self_contained, load_dataset
        from transfia.evaluation.runner import eval_rag, refusal_breakdown, split_summary

        if not self.dataset or not self.dataset.exists():
            return {"ignoré": f"jeu introuvable : {self.dataset} (lance `transfia eval synth --n 10 --llm --hors-corpus 8`)"}
        items = [it for it in load_dataset(self.dataset) if is_self_contained(it.question)]
        if self.max_questions:              # échantillon équilibré : questions du corpus ET hors corpus
            inside = [it for it in items if it.source != "hors_corpus"][: (self.max_questions + 1) // 2]
            outside = [it for it in items if it.source == "hors_corpus"][: self.max_questions // 2]
            items = inside + outside
        rep = eval_rag(self.ctx.rag, items, self.ctx.settings, embedder=None, label="bench")
        s, rows = rep["summary"], rep["rows"]
        b = refusal_breakdown(rows)
        mean = lambda k: s.get(k, {}).get("mean") if isinstance(s.get(k), dict) else None
        return {"questions": len(rows), "fidelite": mean("faithfulness"), "refus_corrects": mean("refusal_correct"),
                "bonne_source": mean("source_hit"), "f1": mean("f1"),
                "refus_a_tort": len(b["refus_a_tort"]), "hallucinations_hors_corpus": len(b["acceptations_a_tort"]),
                "cout_reponses": split_summary(rows)}

    def quiz(self) -> dict:
        out = []
        for topic in QUIZ_TOPICS:
            t0 = time.perf_counter()
            try:
                res = self.ctx.generator.generate("quiz", topic, "bts", n_questions=3)
                out.append({"theme": topic, "questions_gardees": len(res.content.questions), "tentatives": res.attempts,
                            "defauts": _count(res.warnings), "secondes": round(time.perf_counter() - t0, 1)})
            except Exception as exc:
                out.append({"theme": topic, "erreur": f"{type(exc).__name__}: {exc}"})
        kept = [q["questions_gardees"] for q in out if "questions_gardees" in q]
        return {"themes": out, "questions_gardees_total": sum(kept), "questions_demandees": 3 * len(QUIZ_TOPICS)}

    def glossaire(self) -> dict:
        res = self.ctx.generator.generate("glossaire", "le colostrum chez le veau", "bts")
        return {"termes": len(res.content.termes), "defauts": _count(res.warnings)}

    def agent(self) -> dict:
        out = []
        for task in AGENT_TASKS:
            t0 = time.perf_counter()
            try:
                r = self.ctx.agent("native").run(task)
                out.append({"tache": task, "fin": r.stopped_reason, "outils": r.tools_used,
                            "secondes": round(time.perf_counter() - t0, 1)})
            except Exception as exc:
                out.append({"tache": task, "erreur": f"{type(exc).__name__}: {exc}"})
        reasons = [o.get("fin") for o in out]
        return {"taches": out, "autonome": sum(r in ("final", "return_direct") for r in reasons),
                "plan_b": sum(r == "forced_search" for r in reasons), "total": len(out)}

    def editorial(self) -> dict:
        passages = self.ctx.engine.search("le colostrum", k=5)
        st = self.ctx.editorial().run("le colostrum", "grand_public", passages)
        return {"tours": st.rounds, "problemes_par_tour": [len([p for p in r.problemes if p.statut != "soutenue"])
                                                           for r in st.reviews],
                "elagage": any(h.startswith("élagage (") and not h.startswith("élagage (0") for h in st.history),
                "alertes_finales": len(st.final_warnings), "parcours": st.history}

    def plan(self) -> dict:
        from transfia.agents.planner import PlanAndExecute

        run = PlanAndExecute(self.ctx.llm, self.ctx.tools(), self.ctx.skills, approver=lambda p: p,
                             rag=self.ctx.rag).run("Construis une séquence de 2 séances sur la réduction du méthane entérique")
        return {"couvert": run.covered, "etapes": len(run.plan.etapes), "alertes": run.warnings,
                "cite": "## Sources" in run.final}

    # ------------------------------------------------------------------ préparation de la machine
    def prepare(self) -> dict:
        """Conditions de mesure, fixées et notées AVANT de chronométrer (leçon B.2 et banc 4B du 29/10).

        - RAM libre au départ (une RAM saturée double les latences) ;
        - Ollama : version, autres modèles déchargés (sinon l'ancien modèle reste 10 min en mémoire),
          puis un appel de chauffe qui charge le modèle : son temps de chargement ne pollue pas le 1er scénario ;
        - réflexion (qwen3) : vérifie que `think: false` est bien respecté par la version d'Ollama installée."""
        info: dict = {}
        try:
            import psutil

            info["ram_libre_go"] = round(psutil.virtual_memory().available / 1e9, 1)
        except Exception:
            pass
        llm = self.ctx.settings.llm
        if llm.provider != "ollama":
            return info
        import requests

        base = llm.base_url.rstrip("/")
        try:
            info["ollama_version"] = requests.get(f"{base}/api/version", timeout=10).json().get("version")
            loaded = [m.get("name") for m in requests.get(f"{base}/api/ps", timeout=10).json().get("models", [])]
            for name in loaded:
                if name != llm.model:
                    requests.post(f"{base}/api/generate", json={"model": name, "keep_alive": 0}, timeout=60)
            info["modeles_decharges"] = [n for n in loaded if n != llm.model]
            from transfia.llm.base import Message

            t0 = time.perf_counter()
            r = self.ctx.llm.chat([Message("user", "Réponds juste : OK")], max_tokens=20)
            info["chauffe_s"] = round(time.perf_counter() - t0, 1)
            raw = (r.raw or {}).get("message", {}) if isinstance(r.raw, dict) else {}
            info["reflexion_detectee"] = bool(raw.get("thinking")) or "<think>" in (raw.get("content") or "")
        except Exception as exc:
            info["erreur"] = f"{type(exc).__name__}: {exc}"
        return info

    # ------------------------------------------------------------------ exécution
    def run(self, label: str, on_progress=None) -> dict:
        from transfia import BUILD

        st = self.ctx.settings
        st.llm.cache = False                     # un banc d'essai mesure de VRAIS coûts
        report = {"label": label, "date": datetime.now().isoformat(timespec="seconds"), "build": BUILD,
                  "config": st.name, "config_fingerprint": st.fingerprint, "llm": st.llm.model,
                  "generation_model": st.llm.generation_model or st.llm.model,
                  "timeout_s": getattr(st.llm, "timeout_s", None), "scenarios": {}}
        report["machine"] = self.prepare()
        self.log(f"Conditions : {report['machine']}")
        steps = [("rag", self.rag), ("quiz", self.quiz), ("glossaire", self.glossaire), ("agent", self.agent),
                 ("editorial", self.editorial)] + ([("plan", self.plan)] if self.full else [])
        if self.only:
            steps = [(n, f) for n, f in steps if n in self.only]
        for name, fn in steps:
            report["scenarios"][name] = self._run(name, fn)
            if on_progress:                      # sauvegarde après CHAQUE scénario : un arrêt (Ctrl+C) ne perd rien
                _totals(report)
                on_progress(report)
        _totals(report)
        return report


def _totals(report: dict) -> None:
    report["total_secondes"] = round(sum(s["secondes"] for s in report["scenarios"].values()), 1)
    report["total_wh"] = round(sum(s["wh"] for s in report["scenarios"].values()), 3)


def new_path(directory: Path, label: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.json"


def save(report: dict, directory: Path, path: Path | None = None) -> Path:
    path = path or new_path(directory, report["label"])
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    path.with_suffix(".md").write_text(to_markdown([report]), encoding="utf-8")
    return path


def _row(reports: list[dict], title: str, getter) -> str:
    vals = []
    for r in reports:
        try:
            v = getter(r)
        except Exception:
            v = "—"
        vals.append("—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
    return f"| {title} | " + " | ".join(vals) + " |"


def _errors(items: list[dict]) -> str:
    n = sum("erreur" in it for it in items)
    return f" ⚠️ {n} en erreur" if n else ""


def merge(reports: list[dict]) -> list[dict]:
    """Rapports de même étiquette fusionnés (ex. `bench --only editorial` rejoué après une panne) :
    le scénario le plus récent remplace l'ancien ; l'ordre des étiquettes est conservé."""
    out: dict[str, dict] = {}
    for r in sorted(reports, key=lambda r: r["date"]):
        if r["label"] in out:
            base = out[r["label"]]
            base["scenarios"].update(r["scenarios"])
            base["date"] = f"{base['date'].split(' +')[0]} + {r['date']}"
        else:
            out[r["label"]] = json.loads(json.dumps(r))
    for r in out.values():
        r["total_secondes"] = round(sum(s["secondes"] for s in r["scenarios"].values()), 1)
        r["total_wh"] = round(sum(s["wh"] for s in r["scenarios"].values()), 3)
    order = list(dict.fromkeys(r["label"] for r in reports))
    return [out[label] for label in order]


def to_markdown(reports: list[dict]) -> str:
    """Tableau côte à côte (1 colonne par rapport) : lisible par un humain, prêt pour le rapport final."""
    sc = lambda r, n: r["scenarios"][n]
    head = "| Mesure | " + " | ".join(f"{r['label']} ({r['llm']})" for r in reports) + " |"
    lines = [f"# Banc d'essai — {', '.join(r['label'] for r in reports)}", "", head,
             "|---|" + "---|" * len(reports)]
    rows = [
        ("RAG questions", lambda r: sc(r, "rag")["questions"]),
        ("**RAG** fidélité", lambda r: sc(r, "rag")["fidelite"]),
        ("RAG refus corrects", lambda r: sc(r, "rag")["refus_corrects"]),
        ("RAG hallucinations hors corpus", lambda r: sc(r, "rag")["hallucinations_hors_corpus"]),
        ("RAG refus à tort", lambda r: sc(r, "rag")["refus_a_tort"]),
        ("RAG latence p50 / p95 (s)", lambda r: f"{sc(r, 'rag')['cout_reponses']['latency_p50_s']} / "
                                               f"{sc(r, 'rag')['cout_reponses']['latency_p95_s']}"),
        ("**Quiz** questions gardées / demandées", lambda r: f"{sc(r, 'quiz')['questions_gardees_total']} / "
                                                           f"{sc(r, 'quiz')['questions_demandees']}"
                                                           + _errors(sc(r, "quiz")["themes"])),
        ("Quiz défauts", lambda r: json.dumps({k: v for t in sc(r, "quiz")["themes"]
                                               for k, v in t.get("defauts", {}).items()}, ensure_ascii=False)),
        ("**Glossaire** termes absents de leur source", lambda r: sc(r, "glossaire")["defauts"].get("termes absents", 0)),
        ("**Agent** tâches menées seul / plan B / total", lambda r: f"{sc(r, 'agent')['autonome']} / "
                                                                  f"{sc(r, 'agent')['plan_b']} / {sc(r, 'agent')['total']}"
                                                                  + _errors(sc(r, "agent")["taches"])),
        ("Agent : fins des tâches", lambda r: ", ".join(str(t.get("fin") or "erreur") for t in sc(r, "agent")["taches"])),
        ("**Multi-agents** tours / élagage / alertes", lambda r: f"{sc(r, 'editorial')['tours']} / "
                                                                f"{sc(r, 'editorial')['elagage']} / "
                                                                f"{sc(r, 'editorial')['alertes_finales']}"),
        ("**Plan** cité (option --full)", lambda r: sc(r, "plan")["cite"] if "plan" in r["scenarios"] else None),
        ("RAM libre au départ (Go)", lambda r: r.get("machine", {}).get("ram_libre_go")),
        ("Chauffe du modèle (s)", lambda r: r.get("machine", {}).get("chauffe_s")),
        ("Réflexion qwen3 détectée", lambda r: r.get("machine", {}).get("reflexion_detectee")),
        ("Délai max d'un appel LLM (s)", lambda r: r.get("timeout_s")),
        ("Durée totale (min)", lambda r: round(r["total_secondes"] / 60, 1)),
        ("Énergie totale estimée (Wh)", lambda r: r["total_wh"]),
    ]
    lines += [_row(reports, t, g) for t, g in rows]
    errs = [f"- {r['label']} / {n} : {s['erreur']}" for r in reports for n, s in r["scenarios"].items() if not s.get("ok")]
    if errs:
        lines += ["", "**Scénarios en erreur**"] + errs
    lines += ["", "_Build, configuration et empreinte : " + " ; ".join(
        f"{r['label']} = {r['build']} / {r['config']} ({r['config_fingerprint']}) / {r['date']}" for r in reports) + "_"]
    return "\n".join(lines)
