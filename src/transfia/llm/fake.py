"""LLM factice pour les tests (aucun réseau, aucun modèle, résultats déterministes).

Principe des tests d'applications LLM : on teste NOTRE code (boucles, parsing, validation,
garde-fous) avec un LLM simulé. La qualité du vrai modèle, elle, se mesure avec le banc d'éval.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Iterator

from transfia.llm.base import LLMResponse, Message, ToolCall, Usage

Responder = Callable[[list[Message], list[dict] | None, dict | None], "LLMResponse | str | dict"]


class FakeLLM:
    """Deux modes :
    - ``script`` : liste de réponses renvoyées dans l'ordre (str, dict JSON, ou LLMResponse) ;
    - ``responder`` : fonction (messages, tools, schema) -> réponse.
    Sans les deux : comportement "extractif" par défaut (cite le passage [1]).
    """

    def __init__(self, script: list | None = None, responder: Responder | None = None, model: str = "fake"):
        self.script = list(script or [])
        self.responder = responder
        self.model = model
        self.calls: list[dict] = []

    def chat(self, messages, *, tools=None, schema=None, temperature=None, max_tokens=None, model=None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "schema": schema, "model": model or self.model})
        if self.script:
            out = self.script.pop(0)
        elif self.responder:
            out = self.responder(messages, tools, schema)
        else:
            out = default_extractive(messages, schema)
        return _to_response(out, model or self.model, messages)

    def stream(self, messages, *, temperature=None, max_tokens=None, model=None) -> Iterator[str]:
        text = self.chat(messages, temperature=temperature, max_tokens=max_tokens, model=model).content
        for word in re.findall(r"\S+\s*", text):
            yield word


def _to_response(out, model: str, messages) -> LLMResponse:
    if isinstance(out, LLMResponse):
        return out
    if isinstance(out, ToolCall):
        out = [out]
    if isinstance(out, list) and out and isinstance(out[0], ToolCall):
        return LLMResponse(content="", tool_calls=out, model=model, usage=Usage(10, 5, 0.001))
    content = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    n_in = sum(len(m.content.split()) for m in messages)
    return LLMResponse(content=content, model=model, usage=Usage(n_in, len(content.split()), 0.001))


def default_extractive(messages: list[Message], schema: dict | None) -> str:
    """Réponse naïve : reprend la première phrase du passage [1] et la cite."""
    last = messages[-1].content if messages else ""
    m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\[2\]|\n\n|$)", last, re.S)
    if schema:
        return "{}"
    if not m:
        return "Je ne trouve pas cette information dans le corpus."
    first = re.split(r"(?<=[.!?])\s", m.group(1).strip())[0]
    return f"{first} [1]"
