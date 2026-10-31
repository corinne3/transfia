"""Prompts traités comme du code (section 5 du dossier).

- Les prompts vivent dans des fichiers ``prompts/*.j2`` versionnés par git, jamais en dur.
- Chaque prompt a une VERSION = empreinte de son contenu : on l'enregistre avec chaque résultat
  d'évaluation, pour savoir quel prompt a produit quel score.
- ``StrictUndefined`` : une variable oubliée fait planter le rendu (au lieu d'un prompt silencieusement faux).
- Les tests (tests/test_prompts.py) vérifient que chaque prompt se rend et contient ses consignes clés.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from transfia.config import REPO_ROOT

DEFAULT_DIR = REPO_ROOT / "prompts"


@lru_cache(maxsize=4)
def _env(directory: str) -> Environment:
    return Environment(loader=FileSystemLoader(directory), undefined=StrictUndefined,
                       trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)


def render(name: str, directory: Path | None = None, **variables) -> str:
    directory = directory or DEFAULT_DIR
    return _env(str(directory)).get_template(f"{name}.j2").render(**variables).strip()


def version(name: str, directory: Path | None = None) -> str:
    path = (directory or DEFAULT_DIR) / f"{name}.j2"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def all_versions(directory: Path | None = None) -> dict[str, str]:
    directory = directory or DEFAULT_DIR
    return {p.stem: version(p.stem, directory) for p in sorted(directory.glob("*.j2"))}
