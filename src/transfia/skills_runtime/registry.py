"""Skills (section 4 du dossier) : du SAVOIR-FAIRE packagé, chargé à la demande.

Une skill = un dossier ``skills/<nom>/`` contenant :
- ``SKILL.md`` : en-tête YAML (``name``, ``description``) + instructions détaillées ;
- des fichiers annexes facultatifs (exemples, grilles, modèles).

PROGRESSIVE DISCLOSURE (chargement progressif), pour économiser le contexte :
- Niveau 1 : seules les DESCRIPTIONS courtes de toutes les skills sont dans le prompt.
- Niveau 2 : quand une skill est pertinente, on charge le corps de son SKILL.md.
- Niveau 3 : le SKILL.md peut pointer vers des annexes, lues seulement si nécessaire.

Différence avec un TOOL : un tool est une fonction qu'on EXÉCUTE (il agit ou renvoie des données) ;
une skill est un texte qu'on LIT (il change la MANIÈRE dont le modèle travaille).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    description: str
    path: Path

    @property
    def body(self) -> str:
        text = (self.path / "SKILL.md").read_text(encoding="utf-8")
        return _split_frontmatter(text)[1]

    def resources(self) -> list[str]:
        return sorted(p.name for p in self.path.iterdir() if p.is_file() and p.name != "SKILL.md")

    def read_resource(self, filename: str) -> str:
        target = (self.path / filename).resolve()
        if self.path.resolve() not in target.parents:          # pas de "../../" (sécurité)
            raise ValueError("chemin interdit")
        return target.read_text(encoding="utf-8")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        return yaml.safe_load(fm) or {}, body.strip()
    return {}, text.strip()


class SkillRegistry:
    def __init__(self, directory: Path):
        self.directory = directory
        self.skills: dict[str, Skill] = {}
        if directory.exists():
            for skill_md in sorted(directory.glob("*/SKILL.md")):
                meta, _ = _split_frontmatter(skill_md.read_text(encoding="utf-8"))
                name = meta.get("name") or skill_md.parent.name
                self.skills[name] = Skill(name, meta.get("description", ""), skill_md.parent)

    def catalog(self) -> str:
        """Niveau 1 : quelques tokens par skill, toujours présents dans le prompt de l'agent."""
        return "\n".join(f"- {s.name} : {s.description}" for s in self.skills.values())

    def load(self, name: str) -> str:
        """Niveau 2 : le corps complet, + la liste des annexes disponibles (niveau 3)."""
        if name not in self.skills:
            raise KeyError(f"Skill inconnue : {name}. Disponibles : {', '.join(self.skills)}")
        s = self.skills[name]
        res = s.resources()
        extra = f"\n\nAnnexes disponibles (outil read_skill_resource) : {', '.join(res)}" if res else ""
        return s.body + extra

    def read_resource(self, name: str, filename: str) -> str:
        return self.skills[name].read_resource(filename)

    def __contains__(self, name: str) -> bool:
        return name in self.skills
