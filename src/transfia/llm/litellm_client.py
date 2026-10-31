"""Adaptateur cloud via LiteLLM (chapitres 12 à 14) : Gemini, Azure OpenAI, Bedrock...

LiteLLM expose l'API "OpenAI" pour des dizaines de fournisseurs.
Changer de fournisseur = changer la chaîne ``model`` (ex. "gemini/gemini-2.5-flash-lite",
"azure/mon-deploiement", "ollama/qwen2.5:1.5b"). Les clés sont lues dans l'environnement.
"""
from __future__ import annotations

import json
import time
from typing import Iterator

from transfia.llm.base import LLMResponse, Message, ToolCall, Usage
from transfia.observability.tracing import span


class LiteLLMClient:
    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 700, timeout_s: float = 120):
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError("pip install -e \".[cloud]\" pour utiliser LiteLLM") from exc
        import logging

        # LiteLLM avertit à CHAQUE appel que Gemini 3 retirera un jour `temperature` : sans effet aujourd'hui,
        # mais ça noie l'affichage. On ne garde que ses erreurs.
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def chat(self, messages: list[Message], *, tools: list[dict] | None = None, schema: dict | None = None,
             temperature: float | None = None, max_tokens: int | None = None, model: str | None = None) -> LLMResponse:
        import litellm

        model = model or self.model
        kwargs: dict = dict(
            model=model,
            messages=[m.to_openai() for m in messages],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            timeout=self.timeout_s,
            num_retries=3,               # quotas par minute (API Gemini gratuite) : erreur 429 → nouvel essai
        )
        if tools:
            kwargs["tools"] = tools
        if schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "output", "schema": schema}}
        with span("llm.chat", model=model, provider="litellm") as s:
            t0 = time.perf_counter()
            resp = litellm.completion(**kwargs)
            latency = time.perf_counter() - t0
            choice = resp.choices[0].message
            calls = []
            for i, tc in enumerate(getattr(choice, "tool_calls", None) or []):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                calls.append(ToolCall(name=tc.function.name, arguments=args, id=tc.id or f"call_{i}"))
            u = getattr(resp, "usage", None)
            usage = Usage(prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                          completion_tokens=getattr(u, "completion_tokens", 0) or 0, latency_s=latency)
            s.set(tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens)
            return LLMResponse(content=choice.content or "", tool_calls=calls, usage=usage, model=model, raw=resp)

    def stream(self, messages: list[Message], *, temperature: float | None = None,
               max_tokens: int | None = None, model: str | None = None) -> Iterator[str]:
        import litellm

        resp = litellm.completion(model=model or self.model, messages=[m.to_openai() for m in messages],
                                  temperature=self.temperature if temperature is None else temperature,
                                  max_tokens=max_tokens or self.max_tokens, stream=True)
        for chunk in resp:
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield delta.content
