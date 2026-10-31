"""Les outils concrets de TransfIA, exposés à l'agent (et au serveur MCP).

Règles de conception d'outils :
- PEU d'outils (< 10), aux descriptions TRÈS claires : trop d'outils = le modèle se trompe.
- Des sorties COURTES : chaque résultat d'outil consomme du contexte.
- Lecture seule par défaut (moindre privilège).
"""
from __future__ import annotations

from typing import Literal

from transfia.agents.tools import Tool, ToolRegistry, tool
from transfia.core.text import truncate_words


def build_corpus_tools(engine, skills=None, generator=None, kg=None, catalog=None) -> ToolRegistry:
    reg = ToolRegistry()

    @tool()
    def search_corpus(query: str, espece: str = "", top_k: int = 5) -> list[dict]:
        """Cherche des passages dans les articles scientifiques du corpus sur l'élevage. À utiliser pour toute question factuelle.

        query: la recherche, en français
        espece: filtre optionnel (bovin, porcin, ovin, caprin, volaille, poisson, lapin)
        top_k: nombre de passages (1 à 8)
        """
        filters = {"species": espece} if espece else None
        hits = engine.search(query, k=max(1, min(top_k, 8)), filters=filters)
        return [{"chunk_id": h.chunk.id, "article_id": h.chunk.article_id, "source": h.chunk.meta.get("label"),
                 "titre": h.chunk.meta.get("title"), "texte": truncate_words(h.chunk.text, 120)} for h in hits]

    @tool()
    def read_article(article_id: str, section: str = "") -> str:
        """Lit le texte d'un article du corpus (ou d'une de ses sections).

        article_id: identifiant de l'article (DOI ou id renvoyé par search_corpus)
        section: titre (ou début de titre) d'une section, optionnel
        """
        chunks = [c for c in engine.store.all_chunks() if c.article_id == article_id]
        if not chunks:
            return f"Article introuvable : {article_id}"
        if section:
            chunks = [c for c in chunks if c.section.lower().startswith(section.lower())] or chunks
        title = chunks[0].meta.get("title", "")
        body = "\n".join(f"[{c.section}] {c.text}" for c in chunks)
        return f"{title}\n{truncate_words(body, 900)}"

    for t in (search_corpus, read_article):
        reg.register(t)

    if skills is not None:
        @tool()
        def load_skill(name: str) -> str:
            """Charge les instructions détaillées d'une skill (savoir-faire) avant de réaliser la tâche correspondante.

            name: nom exact de la skill
            """
            return skills.load(name)

        @tool()
        def read_skill_resource(name: str, filename: str) -> str:
            """Lit un fichier annexe d'une skill (exemples, grille de qualité).

            name: nom de la skill
            filename: nom du fichier annexe
            """
            return skills.read_resource(name, filename)

        reg.register(load_skill)
        reg.register(read_skill_resource)

    if generator is not None:
        @tool(timeout_s=600, return_direct=True)
        def generate_content(kind: Literal["quiz", "fiche", "cas_pratique", "glossaire"], topic: str,
                             niveau: Literal["grand_public", "bac_pro", "bts", "master"] = "bts") -> str:
            """Génère un contenu pédagogique structuré et sourcé (quiz, fiche, cas pratique, glossaire).

            kind: type de contenu
            topic: thème précis
            niveau: public visé
            """
            from transfia.generation.generator import to_markdown

            res = generator.generate(kind, topic, niveau)
            md = to_markdown(kind, res.content, res.to_dict()["sources"])     # le livrable, tel quel
            return md + ("\n\n" + "\n".join(f"⚠️ {w}" for w in res.warnings) if res.warnings else "")

        reg.register(generate_content)

    if kg is not None:
        @tool()
        def graph_neighbors(entity: str, hops: int = 1) -> list[str]:
            """Donne les relations connues d'un concept dans le graphe de connaissances (ex. une pathologie).

            entity: nom du concept
            hops: profondeur d'exploration (1 ou 2)
            """
            return kg.local_facts(entity, hops=min(max(hops, 1), 2))[:20]

        reg.register(graph_neighbors)

    return reg


def tools_from_list(tools: list[Tool]) -> ToolRegistry:
    return ToolRegistry(tools)
