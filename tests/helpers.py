"""Outils communs aux tests : contexte isolé (dossier temporaire), corpus d'exemple, LLM factice."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from functools import lru_cache

from transfia.config import load_settings
from transfia.container import AppContext


def require(module: str) -> None:
    """Saute le test si une dépendance optionnelle manque (pytest gère unittest.SkipTest)."""
    if importlib.util.find_spec(module) is None:
        raise unittest.SkipTest(f"{module} non installé")


def make_ctx(llm=None) -> AppContext:
    settings = load_settings("config/test.yaml", data_dir=tempfile.mkdtemp(prefix="transfia-test-"))
    ctx = AppContext(settings, llm=llm)
    ctx.index_samples(log=lambda *a: None)
    return ctx


@lru_cache(maxsize=1)
def shared_ctx() -> AppContext:
    """Un contexte indexé partagé (lecture seule) pour accélérer les tests."""
    return make_ctx()
