"""Construit le bon client LLM à partir de la configuration (port -> adaptateur)."""
from __future__ import annotations

from transfia.config import Settings


def build_llm(settings: Settings, model: str | None = None, cache: bool | None = None):
    cfg = settings.llm
    model = model or cfg.model
    if cfg.provider == "fake":
        from transfia.llm.fake import FakeLLM

        client = FakeLLM(model=model)
    elif cfg.provider == "litellm":
        from transfia.llm.litellm_client import LiteLLMClient

        client = LiteLLMClient(model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    else:
        from transfia.llm.ollama import OllamaClient

        client = OllamaClient(model, base_url=cfg.base_url, temperature=cfg.temperature,
                              num_ctx=cfg.num_ctx, max_tokens=cfg.max_tokens, timeout_s=cfg.timeout_s)
    if cfg.cache if cache is None else cache:
        from transfia.llm.cache import CachedLLM

        client = CachedLLM(client, settings.path("cache", "llm_cache.sqlite"))
    return client
