"""Interface commune à tous les LLM (local, cloud, factice).

Un LLM "chat" reçoit une liste de messages (system, user, assistant, tool) et renvoie :
- du texte, OU
- des demandes d'appel d'outils (tool calls) : le LLM n'exécute rien, il DEMANDE (section 4).
Il peut aussi être contraint à produire un JSON conforme à un schéma (F7).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from pydantic import BaseModel, ValidationError


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class Message:
    role: str                                  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None                    # nom de l'outil pour role="tool"
    tool_call_id: str | None = None
    images: list[str] = field(default_factory=list)   # base64 (VLM)

    def to_ollama(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"function": {"name": t.name, "arguments": t.arguments}} for t in self.tool_calls]
        if self.name and self.role == "tool":
            d["tool_name"] = self.name
        if self.images:
            d["images"] = self.images
        return d

    def to_openai(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": t.id or f"call_{i}", "type": "function",
                 "function": {"name": t.name, "arguments": json.dumps(t.arguments, ensure_ascii=False)}}
                for i, t in enumerate(self.tool_calls)
            ]
        if self.role == "tool":
            d["tool_call_id"] = self.tool_call_id or "call_0"
            if self.name:
                d["name"] = self.name
        return d


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float | None = None                # temps jusqu'au premier token (streaming)

    @property
    def tokens_per_s(self) -> float | None:
        return self.completion_tokens / self.latency_s if self.latency_s else None


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: Any = None


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[Message], *, tools: list[dict] | None = None,
             schema: dict | None = None, temperature: float | None = None,
             max_tokens: int | None = None, model: str | None = None) -> LLMResponse: ...

    def stream(self, messages: list[Message], *, temperature: float | None = None,
               max_tokens: int | None = None, model: str | None = None) -> Iterator[str]: ...


# ---------------------------------------------------------------------------
# Sorties structurées : JSON schema -> objet Pydantic validé, avec retry
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def extract_json(text: str) -> Any:
    """Récupère le premier objet JSON d'un texte (utile si le modèle ajoute du blabla)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, attempts: list[str]):
        super().__init__(message)
        self.attempts = attempts


def chat_structured(llm: LLMClient, messages: list[Message], model_cls: type[BaseModel], *,
                    retries: int = 2, temperature: float | None = 0.0,
                    max_tokens: int | None = None, model: str | None = None) -> tuple[BaseModel, int]:
    """Appelle le LLM avec un JSON schema, valide avec Pydantic, et corrige en boucle.

    1. Le schéma est passé au moteur (constrained decoding chez Ollama : JSON toujours valide).
    2. Pydantic vérifie les règles métier (ex. bonne_reponse entre 0 et 3).
    3. En cas d'erreur, on renvoie l'erreur au LLM pour qu'il se corrige (self-correction).
    Renvoie (objet, nombre de tentatives).
    """
    schema = model_cls.model_json_schema()
    msgs = list(messages)
    attempts: list[str] = []
    for attempt in range(retries + 1):
        resp = llm.chat(msgs, schema=schema, temperature=temperature, max_tokens=max_tokens, model=model)
        attempts.append(resp.content)
        try:
            data = extract_json(resp.content)
            return model_cls.model_validate(data), attempt + 1
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            msgs = msgs + [
                Message("assistant", resp.content),
                Message("user", "Ta réponse n'est pas valide. Erreur :\n"
                        f"{str(exc)[:1500]}\nRéponds UNIQUEMENT avec un JSON corrigé conforme au schéma."),
            ]
    raise StructuredOutputError(f"Sortie invalide après {retries + 1} tentatives", attempts)
