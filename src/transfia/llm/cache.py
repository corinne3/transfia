"""Caches (levier de frugalité, 11.1).

1. ``CachedLLM`` : cache EXACT. Même modèle + mêmes messages + mêmes options -> même réponse,
   sans rappeler le modèle. Stocké dans SQLite (survit aux redémarrages).
2. ``SemanticCache`` : cache SÉMANTIQUE. Une question très proche d'une question déjà posée
   (similarité cosinus > seuil) réutilise la réponse. Risque : servir une réponse à une question
   subtilement différente -> seuil élevé, et à évaluer !
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from transfia.llm.base import LLMResponse, ToolCall, Usage
from transfia.observability.tracing import span


class CachedLLM:
    def __init__(self, inner, path: Path):
        self.inner = inner
        self.model = inner.model
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self.hits = 0
        self.misses = 0

    def _key(self, messages, tools, schema, temperature, max_tokens, model) -> str:
        raw = json.dumps({
            "model": model or self.model,
            "messages": [m.to_ollama() for m in messages],
            "tools": tools, "schema": schema, "t": temperature, "max": max_tokens,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def chat(self, messages, *, tools=None, schema=None, temperature=None, max_tokens=None, model=None) -> LLMResponse:
        key = self._key(messages, tools, schema, temperature, max_tokens, model)
        with self._lock, sqlite3.connect(self.path) as db:
            row = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            self.hits += 1
            d = json.loads(row[0])
            with span("llm.cache_hit", model=model or self.model):
                return LLMResponse(content=d["content"], model=d["model"],
                                   tool_calls=[ToolCall(**t) for t in d["tool_calls"]],
                                   usage=Usage(d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"], 0.0))
        self.misses += 1
        resp = self.inner.chat(messages, tools=tools, schema=schema, temperature=temperature,
                               max_tokens=max_tokens, model=model)
        value = json.dumps({
            "content": resp.content, "model": resp.model,
            "tool_calls": [{"name": t.name, "arguments": t.arguments, "id": t.id} for t in resp.tool_calls],
            "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens},
        }, ensure_ascii=False)
        with self._lock, sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
        return resp

    def stream(self, messages, **kwargs):
        # Le streaming n'est pas mis en cache (on veut voir les tokens arriver).
        return self.inner.stream(messages, **kwargs)


class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.95, max_items: int = 1000):
        self.embedder = embedder
        self.threshold = threshold
        self.max_items = max_items
        self.questions: list[str] = []
        self.vectors: list[np.ndarray] = []
        self.values: list[object] = []

    def get(self, question: str):
        if not self.vectors:
            return None
        q = self.embedder.embed_query(question)
        sims = np.stack(self.vectors) @ q
        i = int(np.argmax(sims))
        return self.values[i] if sims[i] >= self.threshold else None

    def put(self, question: str, value: object) -> None:
        self.questions.append(question)
        self.vectors.append(self.embedder.embed_query(question))
        self.values.append(value)
        if len(self.values) > self.max_items:
            self.questions.pop(0), self.vectors.pop(0), self.values.pop(0)
