"""Client Ollama écrit à la main (API HTTP), pour voir exactement ce qui circule.

API utilisée (https://github.com/ollama/ollama/blob/main/docs/api.md) :
- POST /api/chat   {model, messages, tools?, format?, stream, options}
  * ``format`` = un JSON schema -> décodage contraint (le JSON produit respecte le schéma)
  * ``tools``  = schémas des outils -> le modèle peut répondre par des ``tool_calls``
- POST /api/embed  {model, input}
- GET  /api/tags   liste des modèles installés

Les durées renvoyées par Ollama sont en nanosecondes : on en tire tokens/s et latence (F14).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

import requests

from transfia.llm.base import LLMResponse, Message, ToolCall, Usage
from transfia.observability.tracing import span


class OllamaError(RuntimeError):
    pass


# Modèles à « réflexion » (ils écrivent <think>…</think> avant de répondre). Pour nos tâches (JSON, RAG
# cité), la réflexion casse le format et allonge la réponse : on la désactive (champ "think": false d'Ollama),
# et on retire par sécurité tout bloc <think> restant.
THINKING_PREFIXES = ("qwen3", "deepseek-r1", "magistral")
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def is_thinking_model(model: str) -> bool:
    return model.lower().startswith(THINKING_PREFIXES)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip() if "<think>" in (text or "") else (text or "")


class OllamaClient:
    def __init__(self, model: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434",
                 temperature: float = 0.1, num_ctx: int = 4096, max_tokens: int = 700,
                 timeout_s: float = 300.0, session: Any = None, keep_alive: str = "10m"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.keep_alive = keep_alive

    # ------------------------------------------------------------------ utils
    def _options(self, temperature: float | None, max_tokens: int | None) -> dict:
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
            "num_predict": max_tokens or self.max_tokens,
        }

    def _post(self, path: str, payload: dict, stream: bool = False):
        try:
            r = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout_s, stream=stream)
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"Ollama injoignable sur {self.base_url}. Lance l'application Ollama puis réessaie."
            ) from exc
        if r.status_code == 404:
            raise OllamaError(f"Modèle introuvable : lance `ollama pull {payload.get('model')}`")
        if r.status_code >= 400:
            raise OllamaError(f"Erreur Ollama {r.status_code} : {r.text[:500]}")
        return r

    def list_models(self) -> list[str]:
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------- chat
    def chat(self, messages: list[Message], *, tools: list[dict] | None = None,
             schema: dict | None = None, temperature: float | None = None,
             max_tokens: int | None = None, model: str | None = None) -> LLMResponse:
        model = model or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_ollama() for m in messages],
            "stream": False,
            "options": self._options(temperature, max_tokens),
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = tools
        if schema:
            payload["format"] = schema
        if is_thinking_model(model):
            payload["think"] = False
        with span("llm.chat", model=model, n_messages=len(messages), tools=bool(tools), schema=bool(schema)) as s:
            t0 = time.perf_counter()
            data = self._post("/api/chat", payload).json()
            latency = time.perf_counter() - t0
            msg = data.get("message", {})
            calls = [
                ToolCall(name=c["function"]["name"], arguments=_as_dict(c["function"].get("arguments")),
                         id=f"call_{i}")
                for i, c in enumerate(msg.get("tool_calls") or [])
            ]
            usage = Usage(prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                          completion_tokens=data.get("eval_count", 0) or 0, latency_s=latency)
            s.set(tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens,
                  tok_per_s=round(usage.tokens_per_s or 0, 1), n_tool_calls=len(calls))
            return LLMResponse(content=strip_think(msg.get("content", "") or ""), tool_calls=calls, usage=usage,
                               model=model, raw=data)

    def stream(self, messages: list[Message], *, temperature: float | None = None,
               max_tokens: int | None = None, model: str | None = None) -> Iterator[str]:
        """Streaming : Ollama renvoie une ligne JSON par morceau de texte (NDJSON)."""
        model = model or self.model
        payload = {
            "model": model,
            "messages": [m.to_ollama() for m in messages],
            "stream": True,
            "options": self._options(temperature, max_tokens),
            "keep_alive": self.keep_alive,
        }
        if is_thinking_model(model):
            payload["think"] = False          # sinon <think>…</think> s'afficherait mot à mot dans l'interface
        with span("llm.stream", model=model) as s:
            t0 = time.perf_counter()
            ttft = None
            r = self._post("/api/chat", payload, stream=True)
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                        s.set(ttft_s=round(ttft, 3))
                    yield piece
                if chunk.get("done"):
                    s.set(tokens_in=chunk.get("prompt_eval_count", 0), tokens_out=chunk.get("eval_count", 0))
                    break

    # -------------------------------------------------------------- embeddings
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        data = self._post("/api/embed", {"model": model, "input": texts}).json()
        return data["embeddings"]


def _as_dict(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"_raw": args}
    return {}
