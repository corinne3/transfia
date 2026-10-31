"""Prompts traités comme du code : chaque template se rend, contient ses consignes clés, et a une version."""
from pathlib import Path

from jinja2 import UndefinedError

from transfia.core.prompts import all_versions, render

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def test_every_prompt_has_a_version():
    versions = all_versions()
    assert set(versions) == {p.stem for p in PROMPTS.glob("*.j2")} and all(len(v) == 8 for v in versions.values())


def test_rag_system_rules_regression():
    txt = render("rag_system", style="claire", max_sentences=5)
    for rule in ("UNIQUEMENT", "[1]", "Je ne trouve pas cette information dans le corpus.", "DONNÉE"):
        assert rule in txt, rule


def test_missing_variable_fails_loudly():
    try:
        render("rag_user", passages=[])
        raise AssertionError("une variable manquante aurait dû lever une erreur")
    except UndefinedError:
        pass


def test_agent_prompt_text_mode_has_format():
    txt = render("agent_system", tools="- t", mode="text", skills="", memory=[])
    assert "Action Input" in txt and "Final Answer" in txt
    assert "Action Input" not in render("agent_system", tools="- t", mode="native", skills="", memory=[])


def test_generate_prompt_includes_level_and_passages():
    txt = render("generate", kind="quiz", niveau="bts", niveau_desc="étudiants de BTS", topic="colostrum",
                 n_questions=3, skill="", example="", passages=[{"n": 1, "label": "X, 2020", "text": "abc"}])
    assert "Nombre de questions : 3" in txt and "[1] (X, 2020) abc" in txt
