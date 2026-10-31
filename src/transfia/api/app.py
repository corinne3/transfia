"""API web (F13) : FastAPI + Server-Sent Events (SSE) pour le streaming.

SSE : la réponse HTTP reste ouverte et le serveur envoie des lignes ``data: {...}\\n\\n`` au fil
de l'eau. Le front affiche les tokens et les étapes de l'agent dès qu'ils arrivent :
c'est la LATENCE PERÇUE (temps jusqu'au premier token) qui s'améliore, pas le temps total.

Documentation interactive générée automatiquement (OpenAPI) : http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterator, Literal

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from transfia.config import Settings, load_settings
from transfia.container import AppContext


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    espece: str = ""


class AskRequest(BaseModel):
    question: str
    espece: str = ""
    history: list[str] = []


class GenerateRequest(BaseModel):
    kind: Literal["quiz", "fiche", "cas_pratique", "glossaire"]
    topic: str
    niveau: Literal["grand_public", "bac_pro", "bts", "master"] = "bts"
    n_questions: int = 4


class AgentRequest(BaseModel):
    task: str
    mode: Literal["native", "text"] = "native"


class EditorialRequest(BaseModel):
    topic: str
    niveau: Literal["grand_public", "bac_pro", "bts", "master"] = "grand_public"


class Feedback(BaseModel):
    question: str
    answer: str
    rating: Literal[-1, 1]
    comment: str = ""
    trace_id: str | None = None


def sse(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        try:
            for ev in events:
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:                              # l'erreur est envoyée au front, proprement
            yield f"data: {json.dumps({'type': 'error', 'text': f'{type(exc).__name__}: {exc}'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def create_app(settings: Settings | None = None, ctx: AppContext | None = None) -> FastAPI:
    ctx = ctx or AppContext(settings or load_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx.ensure_index()            # au démarrage : indexe le corpus d'exemple si l'index est vide
        yield

    app = FastAPI(title="TransfIA", version="0.1.0", lifespan=lifespan,
                  description="De la recherche en élevage aux contenus pédagogiques sourcés.")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.ctx = ctx

    @app.get("/api/health")
    def health() -> dict:
        from transfia import BUILD

        return {"status": "ok", "build": BUILD, "config": ctx.settings.name, "llm": ctx.settings.llm.model,
                "chunks": ctx.store.count(), "skills": list(ctx.skills.skills)}

    @app.post("/api/search")
    def search(req: SearchRequest) -> list[dict]:
        filters = {"species": req.espece} if req.espece else None
        # rerank SEULEMENT pour "hybrid" : sinon les 3 colonnes sont réordonnées par le même reranker et
        # BM25 = hybride (vu dans l'interface). Comparer des modes = les comparer BRUTS.
        hits = ctx.engine.search(req.query, k=req.k, filters=filters, mode=req.mode, rerank=req.mode == "hybrid")
        return [{"rank": h.rank + 1, "score": round(h.score, 4), "article_id": h.chunk.article_id,
                 "title": h.chunk.meta.get("title"), "label": h.chunk.meta.get("label"), "doi": h.chunk.meta.get("doi"),
                 "section": h.chunk.section, "page": h.chunk.page, "text": h.chunk.text,
                 "species": h.chunk.meta.get("species", []), "themes": h.chunk.meta.get("themes", [])} for h in hits]

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        ans = ctx.rag.answer(req.question, {"species": req.espece} if req.espece else None, req.history)
        d = ans.model_dump(exclude={"hits"})
        d["faithfulness"] = ans.faithfulness
        d["citations"] = [c.model_dump() | {"doi_url": c.doi_url} for c in ans.citations]
        return d

    @app.post("/api/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        return sse(ctx.rag.stream(req.question, {"species": req.espece} if req.espece else None, req.history))

    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> dict:
        from transfia.generation.generator import to_markdown

        try:
            res = ctx.generator.generate(req.kind, req.topic, req.niveau, n_questions=req.n_questions)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        d = res.to_dict()
        d["markdown"] = to_markdown(req.kind, res.content, d["sources"])
        return d

    @app.post("/api/agent/stream")
    def agent_stream(req: AgentRequest) -> StreamingResponse:
        def events():
            for ev in ctx.agent(req.mode).stream(req.task):
                if ev["type"] == "step":
                    yield {"type": "step", "step": asdict(ev["step"])}
                else:
                    yield ev

        return sse(events())

    @app.post("/api/editorial")
    def editorial(req: EditorialRequest) -> dict:
        passages = ctx.engine.search(req.topic, k=5)
        st = ctx.editorial().run(req.topic, req.niveau, passages)
        return {"history": st.history, "draft": st.draft, "final": st.final,
                "reviews": [r.model_dump() for r in st.reviews]}

    @app.get("/api/graph")
    def graph(max_nodes: int = 120) -> dict:
        if len(ctx.kg) == 0:
            from transfia.graph.kg import KnowledgeGraph

            kg = KnowledgeGraph().build_cooccurrence(ctx.store.all_chunks())
            kg.detect_communities()
            return kg.to_force_graph(max_nodes)
        return ctx.kg.to_force_graph(max_nodes)

    @app.get("/api/themes")
    def themes() -> dict:
        path = ctx.settings.path("index", "themes.json")
        if not path.exists():
            from transfia.ml.clustering import theme_map

            res = theme_map(ctx.store.all_chunks(), ctx.embedder)
            path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            return res
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/eval/reports")
    def eval_reports() -> list[dict]:
        out = []
        for p in sorted(ctx.settings.path("eval", ".keep").parent.glob("*.json"), reverse=True)[:30]:
            rep = json.loads(p.read_text(encoding="utf-8"))
            date = rep.get("date") or "{}-{}-{}T{}:{}:{}".format(p.name[:4], p.name[4:6], p.name[6:8], p.name[9:11],
                                                                 p.name[11:13], p.name[13:15])   # anciens rapports
            out.append({"file": p.name, "label": rep.get("label"), "date": date, "llm": rep.get("llm"),
                        "kind": rep.get("kind") or ("retrieval" if rep.get("configs") else "rag"),
                        "dataset": rep.get("dataset"), "note": rep.get("note"), "n_items": rep.get("n_items"),
                        "config": rep.get("config"), "fingerprint": rep.get("config_fingerprint"),
                        "summary": rep.get("summary"), "configs": {k: {m: v for m, v in r.items() if m != "rows"}
                                                                   for k, r in (rep.get("configs") or {}).items()}})
        return out

    @app.post("/api/feedback")
    def feedback(fb: Feedback) -> dict:
        """Évaluation ONLINE (11.3) : pouce haut/bas des utilisateurs, stocké pour analyse."""
        path = ctx.settings.path("feedback", "feedback.jsonl")
        with path.open("a", encoding="utf-8") as f:
            f.write(fb.model_dump_json() + "\n")
        return {"saved": True}

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str) -> list[dict]:
        from transfia.observability.tracing import get_tracer

        return [s.to_dict() for s in get_tracer().trace(trace_id)]

    return app


app = None  # pour `uvicorn transfia.api.app:create_app --factory`
