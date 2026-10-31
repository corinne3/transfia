"""Ligne de commande TransfIA : une commande par brique, pour tester brique par brique.

    transfia doctor                      # vérifie l'environnement (Python, RAM, Ollama, modèles)
    transfia ingest --samples            # indexe le corpus d'exemple (hors ligne)
    transfia collect --max 30            # télécharge le vrai corpus (OpenAlex, CC BY)
    transfia ingest                      # indexe le vrai corpus
    transfia search "mammites"           # F4
    transfia ask "..." --stream          # F5
    transfia generate quiz "colostrum"   # F7
    transfia agent "..."                 # F8 (ajoute --mcp pour passer par le serveur MCP)
    transfia plan "..."                  # F8 plan-and-execute avec validation humaine
    transfia editorial "..."             # F9
    transfia graph build|query|global    # F6
    transfia classify | themes           # F3 (ML classique, clustering)
    transfia eval retrieval|rag|synth|compare|export-human|agreement   # F11
    transfia size | pareto               # F14
    transfia traces                      # F12 : arbre de la dernière trace
    transfia serve | mcp                 # API web (F13) / serveur MCP
    transfia foundations                 # démos "from scratch" (toute l'IA)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
import sys
from pathlib import Path


def _ctx(args):
    from transfia.config import load_settings
    from transfia.container import AppContext

    return AppContext(load_settings(args.config))


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str) if not isinstance(obj, str) else obj)


# ------------------------------------------------------------------ doctor
def cmd_doctor(args) -> None:
    import importlib.util
    import platform

    import transfia
    from transfia import BUILD

    print(f"TransfIA {BUILD} — code chargé depuis : {Path(transfia.__file__).parent}")

    from transfia.config import load_settings
    from transfia.observability.metrics import rss_mb
    from transfia.sobriety.sizing import sizing_report

    s = load_settings(args.config)
    print(f"Python {platform.python_version()} — {platform.system()} {platform.release()}")
    try:
        import psutil

        ram = psutil.virtual_memory()
        print(f"RAM : {ram.total / 1e9:.1f} Go au total, {ram.available / 1e9:.1f} Go disponibles")
    except ImportError:
        print("RAM : installe psutil pour la mesurer (pip install psutil)")
    rss = rss_mb()
    print(f"Configuration : {s.name} (empreinte {s.fingerprint}) — mémoire du processus "
          f"{round(rss) if rss else '?'} Mo")
    for pkg, why in [("sentence_transformers", "embeddings e5 + reranker"), ("qdrant_client", "base vectorielle"),
                     ("fitz", "parsing PDF PyMuPDF"), ("fastapi", "API web"), ("mcp", "serveur MCP"),
                     ("pytest", "tests"), ("codecarbon", "mesure d'énergie"), ("psutil", "mesure mémoire"),
                     ("docling", "parsing avancé (optionnel)"), ("langgraph", "agents LangGraph (optionnel)"),
                     ("litellm", "cloud (optionnel)"), ("matplotlib", "graphiques (optionnel)")]:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  {'✓' if ok else '·'} {pkg:22s} {why}")
    if s.llm.provider == "ollama":
        from transfia.llm.ollama import OllamaClient

        models = OllamaClient(base_url=s.llm.base_url).list_models()
        if not models:
            print(f"✗ Ollama injoignable sur {s.llm.base_url} : installe-le (https://ollama.com) et lance-le.")
        else:
            print(f"✓ Ollama : {len(models)} modèle(s) installé(s)")
            for m in filter(None, [s.llm.model, s.llm.big_model, *s.eval.judges]):
                present = any(x == m or x.startswith(m + ":") or x.split(":")[0] == m for x in models)
                print(f"  {'✓' if present else '✗'} {m}" + ("" if present else f"   -> ollama pull {m}"))
    print("Dimensionnement du LLM principal (estimation) :")
    size = s.llm.model.split(":")[-1].replace("b", "")
    try:
        _print(sizing_report(float(size), 4, s.llm.num_ctx))
    except ValueError:
        pass


# --------------------------------------------------------------- ingestion
def cmd_collect(args) -> None:
    from transfia.config import load_settings
    from transfia.ingestion.collect import collect

    s = load_settings(args.config)
    print(f"Collecte OpenAlex : {args.max} articles max, année >= {args.year_min}, licence CC BY uniquement…")
    _print(collect(s, max_items=args.max, year_min=args.year_min, download=not args.no_download))


def cmd_ingest(args) -> None:
    ctx = _ctx(args)
    if args.samples:
        _print(ctx.index_samples())
        return
    from transfia.ingestion.collect import Catalog

    rows = Catalog(ctx.settings.path("catalog.sqlite")).all()
    if not rows:
        print("Catalogue vide : lance d'abord `transfia collect`, ou `transfia ingest --samples`.")
        return
    items = [(a, pdf if pdf and Path(pdf).exists() else None) for a, pdf, _ in rows[: args.limit or None]]
    stats = ctx.indexer().run(items, force=args.force)
    ctx.reload_bm25()
    _print(stats)


# ---------------------------------------------------------------- recherche
def cmd_search(args) -> None:
    ctx = _ctx(args)
    ctx.ensure_index()
    filters = {"species": args.espece} if args.espece else None
    hits = ctx.engine.search(args.query, k=args.k, filters=filters, mode=args.mode, rerank=not args.no_rerank)
    for h in hits:
        print(f"[{h.rank + 1}] {h.score:.3f}  {h.chunk.meta.get('label')} — {h.chunk.meta.get('title', '')[:70]}")
        print(f"     {h.chunk.text[:220]}…\n")


def cmd_ask(args) -> None:
    ctx = _ctx(args)
    if args.routing:
        ctx.settings.llm.routing = True            # gros modèle autorisé pour CETTE question
    ctx.ensure_index()
    filters = {"species": args.espece} if args.espece else None
    if args.stream:
        for ev in ctx.rag.stream(args.question, filters):
            if ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "step":
                print(f"… {ev['text']}", file=sys.stderr)
            elif ev["type"] == "citations":
                print("\n\nSources :")
                for c in ev["citations"]:
                    print(f"  [{c['n']}] {', '.join(c['authors'][:2])} ({c['year']}) — {c['title']}"
                          + (f" — {c['doi_url']}" if c.get("doi_url") else ""))
            elif ev["type"] == "verification":
                ok = sum(c["supported"] for c in ev["checks"])
                print(f"\nVérification : {ok}/{len(ev['checks'])} phrases soutenues par leurs sources")
            elif ev["type"] == "refusal":
                print(ev["text"])
        return
    ans = ctx.rag.answer(args.question, filters)
    print(ans.text, "\n")
    if ans.refused:
        if ans.usage.get("blocked_answer"):
            print("🛡️  Réponse du modèle BLOQUÉE : aucune de ses phrases n'est soutenue par les sources.")
            print(f"    (réponse bloquée : « {ans.usage['blocked_answer'][:200]} »)")
        else:
            if ans.refusal_reason == "unknown_subject":
                print(f"🛡️  Refus : le corpus ne contient AUCUN passage avec « {', '.join(ans.usage.get('unknown_terms', []))} »"
                      " — le LLM n'a pas été appelé.")
            else:
                print(f"🛡️  Refus ({ans.refusal_reason}) — le LLM n'a pas été appelé.")
        if args.trace:
            from transfia.observability.tracing import get_tracer

            print("\nTrace :\n" + get_tracer().tree(ans.trace_id))
        return
    for c in ans.citations:
        print(f"  [{c.n}] {', '.join(c.authors[:2])} ({c.year}) — {c.title}" + (f" — {c.doi_url}" if c.doi_url else ""))
    if ans.checks:
        print(f"\nFidélité : {ans.faithfulness:.0%} des phrases soutenues ({ans.checks[0].method})")
        for ch in ans.checks:
            if not ch.supported:
                print(f"  ⚠️ non soutenue : {ch.sentence[:120]}")
            for n in ch.bad_citations:
                print(f"  ⚠️ citation [{n}] injustifiée : ce passage ne soutient pas « {ch.sentence[:70]}… »")
    if ans.usage.get("citations_added_post_hoc"):
        print(f"\nℹ️  {ans.usage['citations_added_post_hoc']} phrase(s) sans citation : source retrouvée "
              "a posteriori (attribution).")
    if ans.usage.get("escalated"):
        print(f"\n↗️  Cascade : {ans.usage['escalated']}")
    print(f"\nModèle : {ans.model} — tokens {ans.usage.get('prompt_tokens')} → {ans.usage.get('completion_tokens')}"
          f" — {ans.usage.get('latency_s')} s — requêtes : {ans.queries}")
    if args.trace:
        from transfia.observability.tracing import get_tracer

        print("\nTrace :\n" + get_tracer().tree(ans.trace_id))


# --------------------------------------------------------------- génération
def cmd_generate(args) -> None:
    from transfia.generation.generator import to_markdown

    ctx = _ctx(args)
    if args.model:
        ctx.settings.llm.generation_model = args.model
    if args.no_cache:                     # sinon, même prompt = même 1re réponse (lue dans le cache SQLite)
        ctx.settings.llm.cache = False
    ctx.ensure_index()
    res = ctx.generator.generate(args.kind, args.topic, args.niveau, n_questions=args.n)
    d = res.to_dict()
    md = to_markdown(args.kind, res.content, d["sources"])
    print(md)
    print(f"\nTentatives : {res.attempts} — lisibilité : {res.readability} — avertissements : {res.warnings}")
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"→ enregistré dans {args.out}")


# ------------------------------------------------------------------- agents
def _print_step(step) -> None:
    if step.kind == "tool":
        print(f"🔧 {step.tool}({json.dumps(step.arguments, ensure_ascii=False)[:120]})")
        print(f"   ↳ {step.observation[:200].replace(chr(10), ' ')}…")
    elif step.kind == "thought":
        print(f"💭 {step.content[:300]}")
    elif step.kind == "error":
        print(f"🛡️ {step.content}")
    elif step.kind == "final":
        print(f"\n✅ {step.content}")


def _print_final(ev) -> None:
    if ev["reason"] == "ungrounded":
        print("⚠️ Réponse donnée SANS consulter le corpus (de mémoire) : non sourcée, à ne pas utiliser telle quelle.")
    elif ev["reason"] == "forced_search":
        print("ℹ️ L'agent n'a pas su utiliser ses outils : c'est le code qui a lancé la recherche (plan B).")
    elif ev["reason"] == "max_steps":
        print("⏱️ Arrêt au nombre maximal d'étapes.")


def cmd_agent(args) -> None:
    ctx = _ctx(args)
    facts = ctx.memory.recall(args.user) if args.user else None
    if args.mcp:
        # ⚠️ on n'ouvre PAS l'index ici : c'est le serveur MCP (autre processus) qui le lit.
        # (Qdrant local n'accepte qu'un seul processus à la fois sur un même dossier.)
        import os

        from transfia.agents.mcp_client import MCPToolBridge
        from transfia.agents.react import ReActAgent

        env = dict(os.environ)
        if args.config:
            env["TRANSFIA_CONFIG"] = args.config
        with MCPToolBridge(env=env) as bridge:
            print(f"Outils découverts via MCP : {[t.name for t in bridge.tools]}")
            agent = ReActAgent(ctx.llm, bridge.registry(), ctx.skills, mode=args.mode, memory_facts=facts)
            for ev in agent.stream(args.task):
                if ev["type"] == "step":
                    _print_step(ev["step"])
                elif ev["type"] == "final":
                    _print_final(ev)
        return
    ctx.ensure_index()
    agent = ctx.agent(args.mode, memory_facts=facts)
    for ev in agent.stream(args.task):
        if ev["type"] == "step":
            _print_step(ev["step"])
        elif ev["type"] == "final":
            _print_final(ev)


def cmd_plan(args) -> None:
    from transfia.agents.planner import PlanAndExecute

    ctx = _ctx(args)
    ctx.ensure_index()

    def approver(plan):
        print(f"\nPLAN PROPOSÉ — {plan.objectif}")
        for i, e in enumerate(plan.etapes, 1):
            print(f"  {i}. {e.titre} : {e.description}")
        if args.yes:
            return plan
        answer = input("Valider ce plan ? [o/n] ").strip().lower()
        return plan if answer in ("o", "oui", "y", "yes", "") else None

    def on_step(i, n, titre, result):
        if result is None:
            print(f"\n▶️  Étape {i}/{n} : {titre}…")
        else:
            print(f"   ↳ {result[:200].replace(chr(10), ' ')}…")

    run = PlanAndExecute(ctx.llm, ctx.tools(), ctx.skills, approver=approver, rag=ctx.rag,
                         on_step=on_step).run(args.task)
    if not run.covered:
        print("🛡️ " + run.final)
        return
    print("\n" + run.final)
    for w in run.warnings:
        print(f"⚠️ {w}")


def cmd_editorial(args) -> None:
    ctx = _ctx(args)
    ctx.ensure_index()
    passages = ctx.engine.search(args.topic, k=5)
    st = ctx.editorial().run(args.topic, args.niveau, passages)
    print("Parcours :", " → ".join(st.history))
    for i, r in enumerate(st.reviews, 1):
        bad = [p for p in r.problemes if p.statut != "soutenue"]
        print(f"Relecture {i} : {'valide' if r.valide else f'{len(bad)} problème(s)'}")
        for p in bad:
            print(f"   - {p.statut} : {p.affirmation[:100]}")
    print("\n--- Texte final ---\n" + st.final)
    for w in st.final_warnings:
        print(f"⚠️ {w}")


# -------------------------------------------------------------------- graphe
def cmd_graph(args) -> None:
    from transfia.graph.kg import KnowledgeGraph

    ctx = _ctx(args)
    ctx.ensure_index()
    path = ctx.settings.path("index", "graph.json")
    if args.action == "build":
        kg = KnowledgeGraph().build_cooccurrence(ctx.store.all_chunks())
        if args.llm:
            kg.build_from_llm(ctx.llm, ctx.store.all_chunks(), max_chunks=args.max_chunks)
        kg.summarize_communities(ctx.llm if args.llm else None)
        kg.save(path)
        _print(kg.stats())
        for cid, summ in kg.community_summaries.items():
            print(f"  communauté {cid} : {summ[:150]}")
    elif args.action == "query":
        facts = ctx.kg.local_facts(args.text, hops=args.hops)
        if not facts:
            print(f"Entité inconnue du graphe : {args.text}")
        for f in facts[: args.limit]:
            print(" -", f)
        if len(facts) > args.limit:
            print(f"   … {len(facts) - args.limit} autres relations (--limit pour en voir plus)")
    elif args.action == "global":
        trace: list = []
        answer = ctx.kg.global_answer(ctx.llm, args.text, chunks=ctx.store.all_chunks(), trace=trace)
        for t in trace:                                    # étape MAP visible : on voit où ça casse
            print(f"  [map] communauté {t['communaute']} : {t['extraits']} extrait(s) -> "
                  f"{'gardé' if t['garde'] else 'écarté'} : {t['reponse'][:110].replace(chr(10), ' ')}…")
        print("\n[reduce]\n" + answer)


# ------------------------------------------------------------------------ ML
def cmd_classify(args) -> None:
    from transfia.ml.classic import compare_classifiers, top_terms_per_class

    ctx = _ctx(args)
    ctx.ensure_index()
    chunks = ctx.store.all_chunks()
    res = compare_classifiers(chunks, embedder=ctx.embedder, llm=ctx.llm if args.llm else None)
    if "error" in res:
        print(res["error"])
        return
    print("Classes (passages) :", res["classes"])
    print("Articles par classe :", res["articles_par_classe"])
    g = res["groupe"]
    print("\n--- Résumé : F1 macro ---")
    print("  (A) naïve, toutes classes | (B) naïve, mêmes classes que (C) | (C) groupée par article")
    print("  Fuite = B - C (même périmètre). A vs B = effet du changement de classes, PAS la fuite.")
    for base in ("tfidf_logreg", "embeddings_logreg"):
        a = res.get(base + "_naif", {}).get("f1_macro")
        b = res.get(base + "_naif_meme_perimetre", {}).get("f1_macro")
        c = res.get(base, {}).get("f1_macro")
        if a is None:
            continue
        leak = f"  -> fuite : {b - c:+.3f}" if b is not None and c is not None else ""
        print(f"  {base:18s} (A) {a}  (B) {b if b is not None else '-'}  (C) {c if c is not None else 'impossible'}{leak}")
    if g.get("classes_retirees"):
        print(f"  ⚠️ classes retirées de la CV groupée (un seul article) : {', '.join(g['classes_retirees'])}")
    if g.get("erreur"):
        print(f"  🛡️ {g['erreur']}")
    for name in ("tfidf_logreg", "embeddings_logreg", "llm_zero_shot"):
        if name in res:
            print(f"\n=== {name} ({'zero-shot' if 'llm' in name else 'CV groupée par article'}) : "
                  f"F1 macro {res[name]['f1_macro']} — accuracy {res[name]['accuracy']}")
            print(res[name]["report"])
    print("\nMots les plus discriminants (TF-IDF + régression logistique) :")
    for cls, words in top_terms_per_class(chunks).items():
        print(f"  {cls:16s} {', '.join(words)}")


def cmd_themes(args) -> None:
    from transfia.ml.clustering import theme_map

    ctx = _ctx(args)
    ctx.ensure_index()
    res = theme_map(ctx.store.all_chunks(), ctx.embedder, method=args.method, projection=args.projection)
    out = ctx.settings.path("index", "themes.json")
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    sil = res["silhouette_by_k"]
    print(f"k = {res['k']} — silhouette par k : {sil}")
    best = sil.get(res["k"], 0) if isinstance(sil, dict) else 0
    if isinstance(sil, dict) and sil and res["k"] == max(sil):
        print("  ⚠️ le meilleur k est au BORD de la plage testée : pas de nombre de groupes « naturel ».")
    if best < 0.25:
        print(f"  ⚠️ silhouette {best} < 0,25 : groupes peu séparés (structure faible) ; à lire comme une "
              "tendance, pas comme des thèmes nets.")
    for c, words in res["clusters"].items():
        print(f"  cluster {c} : {', '.join(words)}")
    print(f"→ carte enregistrée : {out}")


# ---------------------------------------------------------------------- éval
def cmd_bench(args) -> None:
    """Banc d'essai : rejoue les tests clés et enregistre des chiffres comparables entre configurations."""
    from transfia.evaluation import bench

    if args.action == "compare":
        reports = bench.merge([json.loads(Path(p).read_text(encoding="utf-8")) for p in args.files])
        md = bench.to_markdown(reports)
        print(md)
        if args.out:                     # écrit en UTF-8 (un `| Tee-Object` PowerShell abîme les accents)
            Path(args.out).write_text(md, encoding="utf-8")
            print(f"\n→ {args.out}")
        return
    ctx = _ctx(args)
    ctx.ensure_index()
    dataset = Path(args.dataset) if args.dataset else ctx.settings.resolve("data/eval/synthetic.jsonl")
    label = args.label or ctx.settings.name
    print(f"Banc d'essai « {label} » — modèle {ctx.settings.llm.model} — {'complet' if args.full else 'standard'}"
          " (cache LLM désactivé : durées réelles)")
    only = [x.strip() for x in args.only.split(",") if x.strip()]
    directory = ctx.settings.path("bench", ".keep").parent
    path = bench.new_path(directory, label)
    print(f"Résultats enregistrés au fil de l'eau dans {path}")
    rep = bench.Bench(ctx, dataset, full=args.full, only=only, max_questions=args.max_questions).run(
        label, on_progress=lambda r: bench.save(r, directory, path))
    bench.save(rep, directory, path)
    print("\n" + bench.to_markdown([rep]) + f"\n\n→ {path}")


def cmd_eval(args) -> None:
    from transfia.evaluation import runner
    from transfia.evaluation.dataset import (dataset_coverage, load_dataset, out_of_corpus_items, pseudo_queries,
                                             save_dataset, synthetic_questions)

    ctx = _ctx(args)
    if args.action == "rag" and not args.cache:
        # Une ÉVALUATION ne lit pas le cache : sinon latence et énergie sont fausses (vu en vrai : 0,8 s au lieu de 22 s,
        # les réponses étant relues dans le cache SQLite). --cache pour l'autoriser (rejouer vite la qualité seule).
        ctx.settings.llm.cache = False
    out_dir = ctx.settings.path("eval", ".keep").parent
    if args.action in ("retrieval", "rag", "synth"):
        ctx.ensure_index()
    dataset_path = Path(args.dataset) if args.dataset else ctx.settings.resolve("data/samples/golden_sample.jsonl")
    if args.action == "synth":
        chunks = ctx.store.all_chunks()
        items = synthetic_questions(ctx.llm, chunks, n=args.n) if args.llm else pseudo_queries(chunks, n=args.n)
        n_in = len(items)
        if args.hors_corpus:
            items += out_of_corpus_items(args.hors_corpus)        # doivent être REFUSÉES
        p = out_dir / ("synthetic.jsonl" if args.llm else "pseudo.jsonl")
        save_dataset(items, p)
        print(f"{len(items)} questions → {p}  ({n_in} avec réponse dans le corpus, {len(items) - n_in} hors corpus)")
    if args.action in ("retrieval", "rag"):
        indexed = {c.article_id for c in ctx.store.all_chunks()}
        cov = dataset_coverage(load_dataset(dataset_path), indexed)
        if cov < 0.5:
            print(f"🛡️ Jeu d'évaluation INCOMPATIBLE avec l'index : seules {cov:.0%} des questions ont leur source "
                  f"attendue dans les {len(indexed)} articles indexés ({dataset_path.name}).\n"
                  "   Les scores seraient à 0 : ils mesureraient le décalage, pas la qualité.\n"
                  "   → Crée un jeu sur TON corpus : transfia eval synth --n 30\n"
                  "     puis : transfia eval retrieval --dataset data\\eval\\pseudo.jsonl")
            return
    if args.action == "retrieval":
        items = [it for it in load_dataset(dataset_path) if it.source_article_ids]   # hors corpus : pas de cible
        rep = runner.eval_retrieval(ctx.engine, items, k=args.k)
        rep["label"] = args.label or f"retrieval-{ctx.settings.name}"
        rep.update({"kind": "retrieval", "date": datetime.now().isoformat(timespec="seconds"),
                    "dataset": dataset_path.name, "note": args.note, "config": ctx.settings.name,
                    "config_fingerprint": ctx.settings.fingerprint, "llm": "— (sans LLM)"})
        path = runner.save_report(rep, out_dir, rep["label"])
        print(runner.to_markdown(rep))
        print(f"\n→ {path}")
    elif args.action == "rag":
        from transfia.evaluation.judge import JudgePanel, LLMJudge

        from transfia.evaluation.dataset import is_self_contained

        items = load_dataset(dataset_path)
        bad = [it for it in items if not is_self_contained(it.question)]
        if bad:                                      # on écarte les questions invalides d'un ancien jeu
            print(f"⚠️ {len(bad)} question(s) non autonome(s) écartée(s) du jeu : "
                  + " ; ".join(f"« {it.question[:70]} »" for it in bad))
            items = [it for it in items if is_self_contained(it.question)]
        items = items[: args.limit or None]
        panel = JudgePanel([LLMJudge(llm) for llm in ctx.judge_llms()]) if args.judges else None
        rep = runner.eval_rag(ctx.rag, items, ctx.settings, panel=panel, embedder=ctx.embedder,
                              label=args.label or f"rag-{ctx.settings.name}")
        rep.update({"kind": "rag", "dataset": dataset_path.name, "note": args.note})
        path = runner.save_report(rep, out_dir, rep["label"])
        print(runner.to_markdown(rep))
        print(f"\n→ {path}")
    elif args.action == "show":                       # relire un rapport SANS relancer l'évaluation
        print(runner.to_markdown(json.loads(Path(args.files[0]).read_text(encoding="utf-8"))))
    elif args.action == "compare":
        a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.files)
        _print(runner.compare_runs(a, b, args.metric))
    elif args.action == "export-human":
        from transfia.evaluation.human import export_for_annotation

        run = json.loads(Path(args.files[0]).read_text(encoding="utf-8"))
        print("→", export_for_annotation(run, out_dir / "annotation.csv"))
    elif args.action == "agreement":
        from transfia.evaluation.human import agreement, load_annotations

        run = json.loads(Path(args.files[0]).read_text(encoding="utf-8"))
        _print(agreement(run, load_annotations(Path(args.files[1]))))


# ------------------------------------------------------------------ sobriété
def cmd_size(args) -> None:
    from transfia.sobriety import sizing

    _print(sizing.sizing_report(args.params, args.bits, args.ctx, args.ram, args.bandwidth))
    tokens = sizing.rag_prompt_tokens(args.chunks, 300)
    print(f"\nPrompt RAG avec {args.chunks} passages de ~300 tokens : ~{tokens} tokens d'entrée")
    print(f"Index de 100 000 chunks en 384 dimensions : ~{sizing.index_size_mb(100_000, 384)} Mo")


def cmd_pareto(args) -> None:
    from transfia.sobriety.pareto import ConfigPoint, pareto_front, plot

    ctx = _ctx(args)
    points = []
    for p in sorted(ctx.settings.path("eval", ".keep").parent.glob("*.json")):
        rep = json.loads(p.read_text(encoding="utf-8"))
        s = rep.get("summary")
        if not s:
            continue
        q = (s.get(args.quality) or {}).get("mean")
        cost = s.get(args.cost)
        if q is not None and cost is not None:
            points.append(ConfigPoint(f"{rep['label']} ({rep['llm']})", q, cost))
    if not points:
        print("Aucun rapport `eval rag` trouvé : lance d'abord `transfia eval rag` avec plusieurs configurations.")
        return
    for p in pareto_front(points):
        print(f"  ★ {p.name} : {args.quality} {p.quality:.3f} — {args.cost} {p.cost}")
    out = plot(points, str(ctx.settings.path("eval", "pareto.png")), args.cost, args.quality)
    if out:
        print("→ graphique :", out)


# -------------------------------------------------------------------- divers
def cmd_traces(args) -> None:
    from transfia.config import load_settings

    s = load_settings(args.config)
    path = s.path("traces", "traces.jsonl")
    if not path.exists():
        print("Aucune trace pour l'instant.")
        return
    spans = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[-3000:] if line.strip()]
    roots = [sp for sp in spans if sp["parent_id"] is None]
    for root in roots[-args.last:]:
        tid = root["trace_id"]
        tree = [sp for sp in spans if sp["trace_id"] == tid]
        print(f"\n=== trace {tid} — {root['name']} — {root['duration_ms']:.0f} ms")
        children: dict = {}
        for sp in tree:
            children.setdefault(sp["parent_id"], []).append(sp)

        def walk(pid, depth):
            for sp in sorted(children.get(pid, []), key=lambda x: x["start"]):
                a = {k: v for k, v in sp["attributes"].items()
                     if k in ("tokens_in", "tokens_out", "n_hits", "model", "tool", "tok_per_s", "faithfulness")}
                print(f"{'  ' * depth}- {sp['name']} [{sp['duration_ms']:.0f} ms] {a if a else ''}"
                      + (f" ❌ {sp['error']}" if sp.get("error") else ""))
                walk(sp["span_id"], depth + 1)

        walk(None, 0)


def cmd_prompts(args) -> None:
    from transfia.core.prompts import all_versions

    for name, v in all_versions().items():
        print(f"  {name:16s} {v}")


def cmd_memory(args) -> None:
    ctx = _ctx(args)
    if args.action == "remember":
        ctx.memory.remember(args.user, args.fact)
    elif args.action == "forget":
        ctx.memory.forget(args.user, args.fact)
    _print(ctx.memory.recall(args.user))


def cmd_models(args) -> None:
    """Copie les modèles Hugging Face DANS le projet (models/), pour fonctionner 100 % hors ligne.

    1. Si le modèle est déjà dans le cache de Hugging Face (déjà utilisé une fois) : simple COPIE, sans réseau.
    2. Sinon : téléchargement direct dans models/.
    Ensuite, le code charge les modèles depuis models/ et n'appelle plus jamais Hugging Face.
    """
    import os
    import shutil

    from transfia.config import load_settings

    s = load_settings(args.config)
    names = [s.embedder.model, s.retrieval.reranker_model]
    target_root = s.resolve(s.models_dir)
    if args.action == "list":
        for n in names:
            local = s.model_path(n) != n
            print(f"  {'✓ dans models/' if local else '· pas encore copié'}  {n}")
        return
    offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"            # on autorise le réseau le temps de cette commande seulement
    try:
        from huggingface_hub import snapshot_download

        for n in names:
            dest = target_root / n.replace("/", "__")
            if (dest / "config.json").exists():
                print(f"  ✓ déjà présent : {dest}")
                continue
            try:
                cached = snapshot_download(n, local_files_only=True)        # déjà dans le cache ?
                shutil.copytree(cached, dest, dirs_exist_ok=True)
                print(f"  ✓ copié depuis le cache : {n} → {dest}")
            except Exception:
                snapshot_download(n, local_dir=str(dest))
                print(f"  ✓ téléchargé : {n} → {dest}")
    finally:
        if offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = offline
    print("Les modèles sont chargés depuis models/ : plus aucun accès à Hugging Face.")


def cmd_serve(args) -> None:
    import uvicorn

    from transfia.api.app import create_app
    from transfia.config import load_settings

    uvicorn.run(create_app(load_settings(args.config)), host=args.host, port=args.port)


def cmd_mcp(args) -> None:
    from transfia.mcp_server.server import main as mcp_main

    mcp_main(["--config", args.config] if args.config else [])


def cmd_foundations(args) -> None:
    import importlib

    for name in ("search_symbolic", "ml_scratch", "neural_net", "transformer", "rl", "lora_quant", "word2vec"):
        if args.only and name != args.only:
            continue
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        importlib.import_module(f"transfia.foundations.{name}").demo()


# ------------------------------------------------------------------- parseur
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="transfia", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="fichier YAML (défaut : config/local.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("collect")
    p.add_argument("--max", type=int, default=30)
    p.add_argument("--year-min", type=int, default=2020)
    p.add_argument("--no-download", action="store_true", help="métadonnées et résumés seulement")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("ingest")
    p.add_argument("--samples", action="store_true", help="corpus d'exemple (hors ligne)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.add_argument("--espece", default="")
    p.add_argument("--no-rerank", action="store_true")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--espece", default="")
    p.add_argument("--trace", action="store_true", help="affiche l'arbre des étapes")
    p.add_argument("--routing", action="store_true", help="autorise le gros modèle (routage + cascade)")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("generate")
    p.add_argument("kind", choices=["quiz", "fiche", "cas_pratique", "glossaire"])
    p.add_argument("topic")
    p.add_argument("--niveau", default="bts", choices=["grand_public", "bac_pro", "bts", "master"])
    p.add_argument("--n", type=int, default=4, help="nombre de questions (quiz)")
    p.add_argument("--out", default="")
    p.add_argument("--model", default="", help="modèle de génération, ex. qwen2.5:3b (sinon config)")
    p.add_argument("--no-cache", action="store_true", help="redemande au LLM au lieu de relire le cache")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("agent")
    p.add_argument("task")
    p.add_argument("--mode", choices=["native", "text"], default="native")
    p.add_argument("--mcp", action="store_true", help="outils fournis par le serveur MCP")
    p.add_argument("--user", default="", help="identifiant pour la mémoire long terme")
    p.set_defaults(fn=cmd_agent)

    p = sub.add_parser("plan")
    p.add_argument("task")
    p.add_argument("--yes", action="store_true", help="valider le plan sans demander")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("editorial")
    p.add_argument("topic")
    p.add_argument("--niveau", default="grand_public")
    p.set_defaults(fn=cmd_editorial)

    p = sub.add_parser("bench", help="banc d'essai : rejoue les tests clés (comparaison de modèles)")
    p.add_argument("action", nargs="?", default="run", choices=["run", "compare"])
    p.add_argument("files", nargs="*", help="compare : rapports JSON de data/bench/")
    p.add_argument("--label", default="")
    p.add_argument("--dataset", default="", help="jeu RAG (défaut : data/eval/synthetic.jsonl)")
    p.add_argument("--full", action="store_true", help="ajoute le plan-and-execute (long)")
    p.add_argument("--only", default="", help="scénarios à jouer, ex. rag,quiz (quota d'API limité)")
    p.add_argument("--out", default="", help="compare : fichier Markdown à écrire (UTF-8)")
    p.add_argument("--max-questions", type=int, default=0, help="RAG : échantillon (moitié corpus, moitié hors corpus)")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("graph")
    p.add_argument("action", choices=["build", "query", "global"])
    p.add_argument("text", nargs="?", default="")
    p.add_argument("--llm", action="store_true", help="extraction de triplets par LLM (lent)")
    p.add_argument("--max-chunks", type=int, default=20)
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--limit", type=int, default=15, help="nombre de relations affichées (query)")
    p.set_defaults(fn=cmd_graph)

    p = sub.add_parser("classify")
    p.add_argument("--llm", action="store_true", help="ajoute le LLM zero-shot (lent)")
    p.set_defaults(fn=cmd_classify)

    p = sub.add_parser("themes")
    p.add_argument("--method", choices=["kmeans", "hdbscan"], default="kmeans")
    p.add_argument("--projection", choices=["pca", "umap"], default="pca")
    p.set_defaults(fn=cmd_themes)

    p = sub.add_parser("eval")
    p.add_argument("action", choices=["retrieval", "rag", "synth", "show", "compare", "export-human", "agreement"])
    p.add_argument("files", nargs="*")
    p.add_argument("--dataset", default="")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--llm", action="store_true", help="synth : questions générées par LLM (sinon pseudo-requêtes)")
    p.add_argument("--cache", action="store_true", help="rag : autorise le cache LLM (coûts alors FAUX)")
    p.add_argument("--hors-corpus", type=int, default=0, dest="hors_corpus",
                   help="synth : ajoute N questions hors corpus (bonne réponse = refus)")
    p.add_argument("--judges", action="store_true", help="rag : notation par le jury de LLM")
    p.add_argument("--label", default="")
    p.add_argument("--note", default="", help="ce qu'on teste (affiché dans l'onglet Évaluation), ex. \"après maj32\"")
    p.add_argument("--metric", default="faithfulness")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("size")
    p.add_argument("--params", type=float, default=3)
    p.add_argument("--bits", type=int, default=4, choices=[16, 8, 5, 4, 3, 2])
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ram", type=float, default=8)
    p.add_argument("--bandwidth", type=float, default=40, help="bande passante mémoire (Go/s)")
    p.add_argument("--chunks", type=int, default=5)
    p.set_defaults(fn=cmd_size)

    p = sub.add_parser("pareto")
    p.add_argument("--quality", default="faithfulness")
    p.add_argument("--cost", default="latency_p50_s")
    p.set_defaults(fn=cmd_pareto)

    p = sub.add_parser("traces")
    p.add_argument("--last", type=int, default=1)
    p.set_defaults(fn=cmd_traces)

    sub.add_parser("prompts").set_defaults(fn=cmd_prompts)

    p = sub.add_parser("memory")
    p.add_argument("action", choices=["remember", "recall", "forget"])
    p.add_argument("user")
    p.add_argument("fact", nargs="?", default=None)
    p.set_defaults(fn=cmd_memory)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    sub.add_parser("mcp").set_defaults(fn=cmd_mcp)

    p = sub.add_parser("models", help="copier les modèles dans models/ (100 % hors ligne)")
    p.add_argument("action", choices=["download", "list"])
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("foundations")
    p.add_argument("--only", default="")
    p.set_defaults(fn=cmd_foundations)
    return ap


def _load_cloud_keys(path: Path | None = None) -> None:
    """Lit les clés cloud du fichier .env (GEMINI_API_KEY, AZURE_API_…), sans dépendance externe.

    Seules les clés cloud sont lues, et une variable déjà définie n'est jamais écrasée :
    la version locale garde exactement le même comportement."""
    import os

    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        key, sep, value = line.strip().partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if sep and value and not key.startswith("#") and (key.startswith(("GEMINI_", "GOOGLE_", "AZURE_"))):
            os.environ.setdefault(key, value)


def main(argv: list[str] | None = None) -> None:
    import os

    # Windows sans "mode développeur" : Hugging Face avertit à chaque lancement qu'il ne peut pas
    # créer de liens symboliques. Ça marche quand même : on coupe ce bruit.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if hasattr(sys.stdout, "reconfigure"):
        try:                                   # Windows : accents et émojis dans la console
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    _load_cloud_keys()
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except ImportError as exc:                 # dépendance optionnelle absente : message clair, pas de traceback
        sys.exit(f"Dépendance manquante : {exc}")
    except Exception as exc:
        if type(exc).__name__ == "OllamaError":
            sys.exit(f"❌ {exc}")
        raise


if __name__ == "__main__":
    main()
