"""F13 — API FastAPI (sautée si FastAPI n'est pas installé)."""
import json

from helpers import make_ctx, require


def test_api_endpoints():
    require("fastapi")
    require("httpx")
    from fastapi.testclient import TestClient

    from transfia.api.app import create_app
    from transfia.llm.fake import FakeLLM

    ctx = make_ctx(FakeLLM())
    client = TestClient(create_app(ctx=ctx))
    assert client.get("/api/health").json()["chunks"] > 0
    hits = client.post("/api/search", json={"query": "colostrum", "k": 2}).json()
    assert hits[0]["article_id"] == "exemple-007"
    ans = client.post("/api/ask", json={"question": "Pourquoi le colostrum est-il important ?"}).json()
    assert ans["citations"] and ans["faithfulness"] == 1.0
    with client.stream("POST", "/api/ask/stream", json={"question": "Pourquoi le colostrum ?"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done"
    assert client.get("/api/graph").json()["nodes"]
    assert client.post("/api/feedback", json={"question": "q", "answer": "a", "rating": 1}).json()["saved"]
