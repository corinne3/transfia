"""Ligne de commande : les commandes principales tournent de bout en bout (config de test)."""
import contextlib
import io

from transfia.cli import main


def run(*args) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["--config", "config/test.yaml", *args])
    return buf.getvalue()


def test_cli_end_to_end():
    assert "total_chunks" in run("ingest", "--samples")
    assert "Prévenir les mammites" in run("search", "mammites")
    assert "[1]" in run("ask", "Pourquoi le colostrum est-il important pour le veau ?")
    assert "hybrid" in run("eval", "retrieval")
    assert "poids_go" in run("size", "--params", "3")
    assert "rag_system" in run("prompts")
    assert "BFS" in run("foundations", "--only", "search_symbolic")
