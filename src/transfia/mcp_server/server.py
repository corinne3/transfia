"""Serveur MCP de TransfIA (section 4 du dossier).

MCP (Model Context Protocol) = une PRISE STANDARD entre des outils et n'importe quel client LLM
compatible (notre agent, mais aussi d'autres applications compatibles MCP).
Trois primitives exposées ici :
- TOOLS     : search_corpus, read_article, list_skills, get_skill (des actions à appeler)
              + ask_corpus : le pipeline RAG COMPLET (recherche, réponse citée, vérification, refus).
              Granularité d'un outil : bas niveau (search_corpus : le client fait le reste) ou HAUT niveau
              (ask_corpus : le serveur garantit la qualité). Un client faible (petit LLM) a besoin du haut niveau.
- RESOURCES : transfia://article/{article_id} (des données à lire)
- PROMPTS   : quiz_prompt (un modèle de prompt réutilisable)

Lancement (transport stdio : le client démarre ce processus et dialogue par stdin/stdout) :
    python -m transfia.mcp_server.server
⚠️ En stdio, on n'écrit JAMAIS sur stdout (réservé au protocole) : les logs vont sur stderr.
"""
from __future__ import annotations

import argparse
import sys

try:                                   # SDK MCP 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                    # SDK MCP 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from transfia.core.text import truncate_words


def build_server(ctx):
    server = _Server(name="transfia", instructions="Outils de recherche dans un corpus d'articles "
                     "scientifiques sur l'élevage (INRAE Productions Animales, CC BY) et skills pédagogiques.")

    @server.tool()
    def search_corpus(query: str, espece: str = "", top_k: int = 5) -> list[dict]:
        """Cherche des passages dans les articles du corpus sur l'élevage.

        query: la recherche en français ; espece: filtre optionnel (bovin, porcin...) ; top_k: 1 à 8
        """
        filters = {"species": espece} if espece else None
        hits = ctx.engine.search(query, k=max(1, min(top_k, 8)), filters=filters)
        return [{"article_id": h.chunk.article_id, "source": h.chunk.meta.get("label"),
                 "titre": h.chunk.meta.get("title"), "doi": h.chunk.meta.get("doi"),
                 "texte": truncate_words(h.chunk.text, 120)} for h in hits]

    @server.tool()
    def ask_corpus(question: str, preferences: str = "") -> str:
        """Répond à une question avec le pipeline RAG complet : réponse citée [n], sources, fidélité vérifiée.
        Refuse si le corpus ne contient pas l'information.

        question: la question en français
        preferences: préférences de l'utilisateur, séparées par " | " (optionnel)
        """
        prefs = [p.strip() for p in preferences.split("|") if p.strip()]
        text, summary = ctx._rag_fallback(question, prefs or None)
        return f"{text}\n\n({summary})"

    @server.tool()
    def read_article(article_id: str) -> str:
        """Renvoie le texte indexé d'un article (par son identifiant)."""
        chunks = [c for c in ctx.store.all_chunks() if c.article_id == article_id]
        if not chunks:
            return f"Article introuvable : {article_id}"
        return chunks[0].meta.get("title", "") + "\n" + truncate_words("\n".join(c.text for c in chunks), 900)

    @server.tool()
    def list_skills() -> str:
        """Liste les skills pédagogiques disponibles (nom : description)."""
        return ctx.skills.catalog()

    @server.tool()
    def get_skill(name: str) -> str:
        """Renvoie les instructions complètes d'une skill."""
        return ctx.skills.load(name)

    @server.resource("transfia://article/{article_id}")
    def article_resource(article_id: str) -> str:
        """Le texte d'un article du corpus."""
        return read_article(article_id)

    @server.prompt()
    def quiz_prompt(theme: str, niveau: str = "bts") -> str:
        """Prompt réutilisable pour demander un QCM sourcé."""
        return (f"Crée un QCM de 4 questions sur « {theme} » pour le niveau {niveau}. "
                "Utilise d'abord l'outil search_corpus, puis la skill quiz-qcm. Cite les sources.")

    return server


def main(argv: list[str] | None = None) -> None:
    from transfia.config import load_settings
    from transfia.container import AppContext

    ap = argparse.ArgumentParser(description="Serveur MCP TransfIA (stdio)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    try:                                   # Windows : stderr en cp1252 ne sait pas écrire "✓"
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ctx = AppContext(load_settings(args.config))
    ctx.ensure_index(log=lambda *a: print(*a, file=sys.stderr))
    build_server(ctx).run()


if __name__ == "__main__":
    main()
