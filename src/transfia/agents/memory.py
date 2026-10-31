"""Mémoire d'agent (F8).

- COURT TERME : l'historique de la conversation. Problème : il grossit sans fin et coûte des tokens
  (et du KV cache). Solution : une FENÊTRE (les N derniers échanges) + un RÉSUMÉ de ce qui sort.
- LONG TERME : des faits durables sur l'utilisateur ("ce formateur veut des cas pratiques bovins"),
  stockés hors du modèle (fichier JSON ici ; base vectorielle si beaucoup de faits) et réinjectés
  dans le prompt système.
"""
from __future__ import annotations

import json
from pathlib import Path

from transfia.core.text import count_tokens
from transfia.llm.base import Message


class LongTermMemory:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def remember(self, user: str, fact: str) -> None:
        facts = self.data.setdefault(user, [])
        if fact not in facts:
            facts.append(fact)
            self._save()

    def forget(self, user: str, fact: str | None = None) -> None:
        if fact is None:
            self.data.pop(user, None)
        elif fact in self.data.get(user, []):
            self.data[user].remove(fact)
        self._save()

    def recall(self, user: str) -> list[str]:
        return list(self.data.get(user, []))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


class ConversationMemory:
    def __init__(self, max_tokens: int = 1500, llm=None):
        self.max_tokens = max_tokens
        self.llm = llm
        self.messages: list[Message] = []
        self.summary = ""

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))
        self._compact()

    def _compact(self) -> None:
        while sum(count_tokens(m.content) for m in self.messages) > self.max_tokens and len(self.messages) > 2:
            old = self.messages.pop(0)
            if self.llm is not None:
                prompt = f"Résumé actuel : {self.summary}\nNouvel élément ({old.role}) : {old.content}\n" \
                         "Mets à jour le résumé en 3 phrases maximum."
                self.summary = self.llm.chat([Message("user", prompt)], max_tokens=150).content.strip()
            else:
                self.summary = (self.summary + f" | {old.role}: {old.content[:120]}")[-600:]

    def as_messages(self) -> list[Message]:
        prefix = [Message("system", f"Résumé de la conversation précédente : {self.summary}")] if self.summary else []
        return prefix + list(self.messages)

    def history_text(self) -> list[str]:
        return [f"{m.role}: {m.content}" for m in self.messages[-6:]]
