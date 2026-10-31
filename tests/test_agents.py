"""F7-F9 — génération structurée, skills, tools, agent ReAct (natif + textuel), plan-and-execute,
multi-agents, mémoire, serveur MCP."""
import os
import tempfile
from pathlib import Path
from typing import Literal

from helpers import make_ctx, require, shared_ctx

from transfia.agents.memory import ConversationMemory, LongTermMemory
from transfia.agents.multi_agent import EditorialWorkflow
from transfia.agents.planner import PlanAndExecute
from transfia.agents.react import ReActAgent
from transfia.agents.tools import ToolRegistry, tool
from transfia.generation.readability import count_syllables, flesch_kandel_moles
from transfia.generation.schemas import Quiz
from transfia.llm.base import ToolCall
from transfia.llm.fake import FakeLLM

QUIZ_OK = {"titre": "Colostrum", "niveau": "bts", "questions": [
    {"enonce": "Pourquoi donner le colostrum tôt ?", "choix": ["Absorption des anticorps", "Goût", "Couleur", "Prix"],
     "bonne_reponse": 0, "explication": "L'absorption des anticorps diminue vite.", "source": 1}]}


# ------------------------------------------------------------------ tools
def test_tool_decorator_builds_json_schema_and_validates():
    @tool()
    def f(query: str, mode: Literal["a", "b"] = "a", k: int = 3, tags: list[str] | None = None) -> str:
        """Cherche quelque chose.

        query: le texte
        k: combien
        """
        return f"{query}-{mode}-{k}"

    p = f.parameters
    assert p["required"] == ["query"] and p["properties"]["mode"]["enum"] == ["a", "b"]
    assert p["properties"]["k"]["type"] == "integer" and p["properties"]["query"]["description"] == "le texte"
    reg = ToolRegistry([f])
    assert reg.call("f", {"query": "x", "k": "5"}) == "x-a-5"                      # conversion de type
    assert reg.call("f", {"k": 2}).startswith("ERREUR : arguments invalides")       # argument manquant
    assert reg.call("inconnu", {}).startswith("ERREUR : outil inconnu")
    assert reg.schemas()[0]["function"]["name"] == "f"


def test_tool_timeout():
    import time

    @tool(timeout_s=0.2)
    def slow() -> str:
        """Lent."""
        time.sleep(2)
        return "fini"

    assert "dépassé" in ToolRegistry([slow]).call("slow", {})


# ----------------------------------------------------------------- skills
def test_skills_progressive_disclosure():
    skills = shared_ctx().skills
    assert "quiz-qcm" in skills and "description" not in skills.catalog()
    assert "quiz-qcm :" in skills.catalog()                            # niveau 1 : descriptions
    body = skills.load("quiz-qcm")                                      # niveau 2 : corps
    assert "4 propositions" in body and "grille-qualite.md" in body     # + annexes listées (niveau 3)
    assert "Grille qualité" in skills.read_resource("quiz-qcm", "grille-qualite.md")
    try:
        skills.read_resource("quiz-qcm", "../../pyproject.toml")
        raise AssertionError("traversée de chemin acceptée !")
    except ValueError:
        pass


# ------------------------------------------------------------- génération
def test_generator_structured_output_with_skill_and_retry():
    bad = dict(QUIZ_OK, questions=[dict(QUIZ_OK["questions"][0], choix=["A", "A", "B", "C"])])
    llm = FakeLLM(script=[bad, QUIZ_OK])
    ctx = make_ctx(llm)
    res = ctx.generator.generate("quiz", "le colostrum", "bts")
    assert isinstance(res.content, Quiz) and res.attempts == 2
    prompt = llm.calls[0]["messages"][0].content
    assert "Savoir-faire" in prompt and "Distracteurs" in prompt.replace("distracteurs", "Distracteurs")
    assert res.to_dict()["sources"][0]["article_id"]


def test_generator_warns_on_unknown_source():
    q = dict(QUIZ_OK, questions=[dict(QUIZ_OK["questions"][0], source=42)])
    ctx = make_ctx(FakeLLM(script=[q]))
    res = ctx.generator.generate("quiz", "le colostrum", "bts")
    assert any("n'existe pas" in w for w in res.warnings)


def test_readability():
    assert count_syllables("vache") == 1 and count_syllables("alimentation") >= 4
    easy = flesch_kandel_moles("La vache mange. Le veau boit. Le lait est bon.")
    hard = flesch_kandel_moles("L'immunoglobuline colostrale conditionne l'immunité passive néonatale des bovins.")
    assert easy > hard


# ------------------------------------------------------------------ agent
def test_react_agent_native_tool_calling():
    llm = FakeLLM(script=[
        [ToolCall("load_skill", {"name": "quiz-qcm"}, "c1")],
        [ToolCall("search_corpus", {"query": "colostrum veau", "top_k": 2}, "c2")],
        "Voici 1 question sur le colostrum (Exemple TransfIA, 2022).",
    ])
    ctx = make_ctx(llm)
    res = ctx.agent().run("Fais une question de QCM sur le colostrum")
    assert res.tools_used == ["load_skill", "search_corpus"] and res.stopped_reason == "final"
    assert "Colostrum" in res.steps[1].observation
    assert any(m.role == "tool" for m in llm.calls[-1]["messages"])                 # résultats renvoyés au LLM
    assert llm.calls[0]["tools"] and "quiz-qcm" in llm.calls[0]["messages"][0].content


def test_react_agent_text_mode():
    llm = FakeLLM(script=[
        'Thought: je cherche\nAction: search_corpus\nAction Input: {"query": "méthane rumen"}\n',
        "Thought: j'ai ce qu'il faut\nFinal Answer: Le méthane vient du rumen.",
    ])
    ctx = make_ctx(llm)
    res = ReActAgent(llm, ctx.tools(), mode="text").run("D'où vient le méthane ?")
    assert res.answer == "Le méthane vient du rumen." and res.tools_used == ["search_corpus"]


def test_react_agent_stops_repeated_calls_and_max_steps():
    call = [ToolCall("search_corpus", {"query": "x"}, "c")]
    llm = FakeLLM(script=[call, call, call, call, "fin forcée"])
    ctx = make_ctx(llm)
    res = ReActAgent(llm, ctx.tools(), max_steps=4).run("boucle")
    assert res.stopped_reason == "stuck" and "déjà été fait" in res.steps[-2].observation   # boucle détectée


def test_plan_and_execute_with_human_approval():
    plan = {"objectif": "Séquence colostrum", "etapes": [{"titre": "Chercher", "description": "corpus", "outil_suggere": "search_corpus"}]}
    llm = FakeLLM(script=[plan, "Résultat étape 1", "# Séquence finale"])
    ctx = make_ctx(llm)
    seen = []
    run = PlanAndExecute(llm, ctx.tools(), approver=lambda p: seen.append(p) or p).run("Séquence sur le colostrum")
    assert seen and run.final == "# Séquence finale" and run.results == ["Résultat étape 1"]
    refused = PlanAndExecute(FakeLLM(script=[plan]), ctx.tools(), approver=lambda p: None).run("x")
    assert not refused.approved


def test_editorial_workflow_loops_until_valid():
    bad = {"problemes": [{"affirmation": "X", "statut": "non_soutenue", "explication": "absent"}], "valide": True}
    good = {"problemes": [], "valide": True}
    writer = FakeLLM(script=["brouillon 1 [1]", "brouillon 2 [1]", "texte vulgarisé [1]"])
    reviewer = FakeLLM(script=[bad, good])
    ctx = make_ctx()
    st = EditorialWorkflow(writer, reviewer, skills=ctx.skills).run("colostrum", "grand_public",
                                                                    ctx.engine.search("colostrum", k=2))
    assert st.rounds == 2 and st.final == "texte vulgarisé [1]"
    assert st.reviews[0].valide is False                       # "valide" incohérent avec les problèmes -> corrigé
    assert st.history[:2] == ["rédacteur (tour 1)", "relecteur (1 problème(s) LLM + 0 automatique(s))"]
    assert st.history[-2:] == ["vulgarisateur", "contrôle final"]


def test_editorial_does_not_loop_without_problems_and_checks_final_text():
    """Cas réel : 3 tours avec « 0 problème », puis « Les vaches Charolaise ont des colostrums faibles »
    (la source dit 20 %)."""
    passage = "Une étude a montré que 20 % des colostrums de vaches Charolaise étaient trop faiblement concentrés."
    writer = FakeLLM(script=["Une étude a montré que 20 % des colostrums de vaches Charolaise étaient trop "
                             "faiblement concentrés [1].",
                             "Les vaches Charolaise ont des colostrums faibles."])
    reviewer = FakeLLM(script=[{"problemes": [], "valide": False}])          # drapeau faux, aucun problème
    ctx = make_ctx()
    hit = ctx.engine.search("colostrum", k=1)
    hit[0].chunk.text = passage
    st = EditorialWorkflow(writer, reviewer, skills=ctx.skills).run("colostrum", "grand_public", hit)
    assert st.rounds == 1                                        # pas de tour à vide
    assert any("PERDUS" in w and "20" in w for w in st.final_warnings)


# ---------------------------------------------------------------- mémoire
def test_memories():
    m = LongTermMemory(Path(tempfile.mkdtemp()) / "m.json")
    m.remember("enseignant", "préfère les cas pratiques bovins")
    assert LongTermMemory(m.path).recall("enseignant") == ["préfère les cas pratiques bovins"]
    m.forget("enseignant", "préfère les cas pratiques bovins")
    assert m.recall("enseignant") == []
    conv = ConversationMemory(max_tokens=30)
    for i in range(10):
        conv.add("user", f"message numéro {i} sur les vaches laitières")
    assert len(conv.messages) < 10 and conv.summary and conv.as_messages()[0].role == "system"


# -------------------------------------------------------------------- MCP
def test_mcp_server_tools_through_the_bridge():
    require("mcp")
    from transfia.agents.mcp_client import MCPToolBridge

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), TRANSFIA_CONFIG=str(root / "config" / "test.yaml"))
    with MCPToolBridge(env=env) as bridge:
        assert {"search_corpus", "read_article", "list_skills", "get_skill"} <= {t.name for t in bridge.tools}
        reg = bridge.registry()
        assert "Colostrum" in reg.call("search_corpus", {"query": "colostrum du veau", "top_k": 1})
        llm = FakeLLM(script=[[ToolCall("get_skill", {"name": "quiz-qcm"}, "c")], "ok"])
        res = ReActAgent(llm, reg).run("charge la skill")
        assert "Créer un QCM" in res.steps[0].observation
        # outil HAUT niveau : le pipeline RAG complet, côté serveur ; et plan B de l'agent via MCP
        assert "fidélité" in reg.call("ask_corpus", {"question": "Pourquoi le colostrum est-il indispensable au veau ?"})
        res = ReActAgent(FakeLLM(script=["De mémoire.", "Je vais vérifier."]), reg).run(
            "Pourquoi le colostrum est-il indispensable au veau ?")
        assert res.stopped_reason == "forced_search" and res.tools_used == ["ask_corpus"]


# ------------------------------------------- cohérence du QCM (cas réel, qwen2.5:1.5b)
QUIZ_REEL = {"titre": "Méthane entérique", "niveau": "bts", "questions": [
    {"enonce": "Quelles sont les principales raisons pour lesquelles la réduction des émissions de méthane entérique "
               "est un objectif important pour les pays concernés par le changement climatique?",
     "choix": ["Pour améliorer la santé des animaux", "Pour réduire les émissions de gaz à effet de serre",
               "Pour augmenter la production de lait et du lait", "Pour augmenter la production de viande"],
     "bonne_reponse": 2, "source": 1,
     "explication": "La réduction des émissions de méthane entérique est un objectif important car elle contribue à "
                    "réduire les émissions de gaz à effet de serre. Les émissions proviennent des ruminants et sont "
                    "un facteur clé de la production de lait et de la viande."},
    {"enonce": "Quelles sont les principales méthodes utilisées pour prédire les émissions de méthane entérique "
               "dans les élevages bovins?",
     "choix": ["Mesures directes des émissions de méthane entérique", "Prédictions via le lait et les fèces",
               "Évaluation des performances génétiques", "Évaluation des émissions de méthane génomique"],
     "bonne_reponse": 1, "source": 2,
     "explication": "Les principales méthodes sont les mesures directes des émissions de méthane entérique et les "
                    "prédictions via le lait et les fèces."},
    {"enonce": "Quels sont les principaux leviers d'intérêt pour réduire les émissions de méthane entérique dans "
               "les élevages bovins?",
     "choix": ["Alimentation et conduite des troupeaux", "Prévention des maladies",
               "Évaluation des performances génétiques", "Évaluation des émissions de méthane génomique"],
     "bonne_reponse": 1, "source": 3,
     "explication": "Les principaux leviers sont l'alimentation et la conduite des troupeaux. Ils réduisent les "
                    "émissions tout en préservant la production et la santé des animaux."}]}


def test_quiz_consistency_fixes_wrong_index_and_flags_ambiguity():
    from transfia.generation.consistency import check_quiz

    quiz = Quiz.model_validate(QUIZ_REEL)               # JSON valide : Pydantic ne voit RIEN
    warnings = check_quiz(quiz, fix=True)
    assert [q.bonne_reponse for q in quiz.questions] == [1, 1, 0]       # Q1 C->B, Q3 B->A
    assert "Q1" in warnings[0] and "corrigée en B" in warnings[0]
    assert "Q2" in warnings[1] and "ambiguë" in warnings[1] and "A, B" in warnings[1]
    assert "Q3" in warnings[2] and "corrigée en A" in warnings[2]


def test_consistent_quiz_has_no_warning():
    from transfia.generation.consistency import check_quiz

    assert check_quiz(Quiz.model_validate(QUIZ_OK)) == []


def test_schema_asks_explanation_before_answer():
    props = list(Quiz.model_json_schema()["$defs"]["QuestionQCM"]["properties"])
    assert props.index("explication") < props.index("bonne_reponse")


def test_generator_corrects_quiz_and_lists_only_cited_sources():
    from transfia.generation.generator import to_markdown

    ctx = make_ctx(FakeLLM(script=[QUIZ_REEL, QUIZ_REEL]))       # 2e appel = réparation, pas meilleure -> ignorée
    res = ctx.generator.generate("quiz", "méthane entérique", "bts")
    assert any("corrigée" in w for w in res.warnings)
    d = res.to_dict()
    md = to_markdown("quiz", res.content, d["sources"])
    assert "Réponse : B" in md and "Réponse : A" in md
    sources = md.split("## Sources")[1]
    assert "[4]" not in sources and "[5]" not in sources        # non cités -> non listés
    assert sum(1 for line in sources.splitlines() if line.startswith("[")) == len({s["article_id"] for s in d["sources"] if s["cited"]})


def _q(enonce, choix, bonne, expl, source=1):
    return {"enonce": enonce, "choix": choix, "bonne_reponse": bonne, "explication": expl, "source": source}


PRED = ["Mesures directes des émissions", "Prédictions via le lait ou les fèces",
        "Évaluation génomique des émissions", "Évaluation des interactions microbiennes du rumen"]
QUIZ_DOUBLON = {"titre": "Méthane", "niveau": "bts", "questions": [   # 2e essai réel : Q3 = copie des choix de Q2
    _q("Quelles méthodes pour prédire le méthane ?", PRED, 2,
       "On utilise les prédictions via le lait ou les fèces et l'évaluation génomique des émissions."),
    _q("Quelles stratégies pour réduire le méthane ?", PRED, 2,
       "On utilise les prédictions via le lait ou les fèces et l'évaluation génomique des émissions.")]}
QUIZ_REPARE = {"titre": "Méthane", "niveau": "bts", "questions": [
    _q("Quelle mesure prédit le méthane sans chambre respiratoire ?", PRED, 1,
       "Le méthane est estimé par des prédictions à partir du lait ou des fèces."),
    _q("Quel levier réduit directement le méthane ?", ["Additifs alimentaires", "Pâturage nocturne", "Tonte", "Éclairage"], 0,
       "Les additifs alimentaires réduisent la production de méthane dans le rumen.")]}


def test_duplicate_choices_are_detected():
    from transfia.generation.consistency import check_quiz

    w = check_quiz(Quiz.model_validate(QUIZ_DOUBLON))
    assert any("Q2 reprend les mêmes propositions que Q1" in x for x in w)


def test_generator_repairs_ambiguous_quiz_with_feedback():
    llm = FakeLLM(script=[QUIZ_DOUBLON, QUIZ_REPARE])
    res = make_ctx(llm).generator.generate("quiz", "méthane entérique", "bts")
    assert res.content.questions[1].choix[0] == "Additifs alimentaires"      # version réparée gardée
    assert res.warnings[0].startswith("réparé") and res.attempts == 2
    feedback = llm.calls[1]["messages"][-1].content
    assert "reprend les mêmes propositions" in feedback and "ambiguë" in feedback   # le LLM reçoit la critique


def test_unrepaired_questions_are_dropped():
    llm = FakeLLM(script=[QUIZ_REEL, QUIZ_REEL])         # la réparation ne change rien (vu avec qwen2.5:1.5b)
    res = make_ctx(llm).generator.generate("quiz", "méthane entérique", "bts")
    assert len(res.content.questions) == 2                     # Q2 ambiguë retirée, Q1 et Q3 corrigées gardées
    assert any("pas mieux" in w for w in res.warnings) and any("retirées car non réparées : Q2" in w for w in res.warnings)


def test_generation_model_is_used():
    llm = FakeLLM(script=[QUIZ_OK])
    gen = make_ctx(llm).generator
    gen.model = "qwen2.5:3b"
    gen.generate("quiz", "le colostrum", "bts")
    assert llm.calls[0]["model"] == "qwen2.5:3b"


# ------------------------------------------- ancrage de l'agent (cas réel : réponse "de mémoire")
def test_agent_answer_from_memory_is_sent_back_once():
    llm = FakeLLM(script=["Le méthane vient des bactéries de l'estomac.",                 # de mémoire
                          [ToolCall("search_corpus", {"query": "méthane ruminants"}, "c1")],
                          "Le méthane est produit dans le rumen (Martin, 2023)."])
    res = make_ctx(llm).agent().run("D'où vient le méthane des ruminants ?")
    assert [s.kind for s in res.steps][:2] == ["error", "tool"] and res.stopped_reason == "final"
    assert any("sans consulter le corpus" in m.content for m in llm.calls[1]["messages"] if m.role == "user")


def test_agent_that_insists_falls_back_to_rag_pipeline():
    # cas réel : après le rappel, "Je vais vérifier la source de la question." ... sans appel d'outil
    def responder(messages, tools, schema):
        if tools:                                     # l'agent (avec outils) : répond toujours de mémoire
            return "Le méthane vient des bactéries de l'estomac."
        return "Le colostrum apporte au veau les anticorps dont il a besoin à la naissance [1]."   # le pipeline RAG
    llm = FakeLLM(responder=responder)
    res = make_ctx(llm).agent().run("Pourquoi le colostrum est-il indispensable au veau ?")
    assert res.stopped_reason == "forced_search" and res.tools_used == ["pipeline_rag"]
    assert "anticorps" in res.answer and "Sources" in res.answer and "bactéries" not in res.answer


def test_plan_b_without_rag_uses_fresh_context():
    llm = FakeLLM(script=["De mémoire.", "Je vais vérifier la source.", "Le méthane vient du rumen (Martin, 2023)."])
    agent = make_ctx(llm).agent()
    agent.fallback = None                             # ex. mode MCP : pas de pipeline RAG dans ce processus
    res = agent.run("D'où vient le méthane ?")
    assert res.stopped_reason == "forced_search" and res.tools_used == ["search_corpus"]
    last = llm.calls[-1]["messages"]
    assert len(last) == 1 and "UNIQUEMENT sur ces passages" in last[0].content     # contexte NEUF : pas d'historique


def test_agent_without_forced_search_is_flagged_ungrounded():
    llm = FakeLLM(script=["De mémoire.", "Toujours de mémoire."])
    agent = make_ctx(llm).agent()
    agent.force_search = False
    res = agent.run("D'où vient le méthane ?")
    assert res.stopped_reason == "ungrounded" and res.answer == "Toujours de mémoire."



# ------------------------------------------- mode texte (cas réel : 8 fois le même load_skill fautif)
def test_misnamed_argument_is_repaired_and_said():
    reg = make_ctx(FakeLLM()).tools()
    out = reg.call("load_skill", {"skill": "quiz-qcm"})              # "skill" au lieu de "name"
    assert out.startswith("(note : argument 'skill' interprété comme 'name')") and "QCM" in out


def test_invalid_arguments_error_is_readable():
    reg = make_ctx(FakeLLM()).tools()
    out = reg.call("search_corpus", {"top_k": 3})
    assert "Arguments attendus : query (string, obligatoire)" in out and 'Exemple : {"query": "..."}' in out


def test_looping_text_agent_is_stopped_then_plan_b():
    loop = 'Thought: je charge la skill\nAction: read_skill_resource\nAction Input: {"name": "x", "filename": "y"}'
    def responder(messages, tools, schema):
        if "Passages" in messages[-1].content or "<passages>" in messages[-1].content:
            return "Le colostrum apporte au veau les anticorps dont il a besoin à la naissance [1]."
        return loop
    llm = FakeLLM(responder=responder)
    res = make_ctx(llm).agent(mode="text").run("Pourquoi le colostrum est-il indispensable au veau ?")
    assert res.stopped_reason == "forced_search" and res.tools_used.count("read_skill_resource") == 4
    assert any("tourne en rond" in st.content for st in res.steps if st.kind == "error")


def test_final_answer_is_cleaned_of_react_format():
    from transfia.agents.react import _clean_final

    assert _clean_final("Thought: j'ai fini\nAction: load_skill\nAction Input: {}\nLe rumen.") == "Le rumen."
    assert _clean_final("Thought: je vais charger\nAction: load_skill\nAction Input: {}") == ""   # un plan, pas une réponse
    assert _clean_final("Thought: ok\nFinal Answer: Le rumen.") == "Le rumen."


def test_almost_right_enum_values_are_repaired():
    from transfia.agents.tools import repair_arguments

    t = make_ctx(FakeLLM()).tools().tools["generate_content"]
    args, note = repair_arguments(t, {"kind": "cas-pratique", "topic": "méthane", "niveau": "Bac Pro"})
    assert args["kind"] == "cas_pratique" and args["niveau"] == "bac_pro" and "'cas-pratique' interprété" in note


# ------------------------------------------- mémoire long terme (cas réel : préférence perdue au plan B)
def test_memory_follows_the_handoff_to_rag():
    def responder(messages, tools, schema):
        if tools:
            return "Je vais vérifier."                     # l'agent n'appelle pas d'outil -> plan B
        return "Le colostrum apporte au veau les anticorps dont il a besoin à la naissance [1]."
    llm = FakeLLM(responder=responder)
    ctx = make_ctx(llm)
    res = ctx.agent(memory_facts=["préfère les cas pratiques en élevage bovin"]).run(
        "Pourquoi le colostrum est-il indispensable au veau ?")
    rag_system = llm.calls[-1]["messages"][0].content
    assert res.stopped_reason == "forced_search" and "préfère les cas pratiques en élevage bovin" in rag_system
    assert "préférences appliquées" in res.steps[-2].observation


def test_rag_sources_grouped_by_article():
    ctx = make_ctx(FakeLLM(responder=lambda m, t, s: "Le colostrum apporte des anticorps au veau [1][2]."))
    text, _ = ctx._rag_fallback("Pourquoi le colostrum est-il indispensable au veau ?")
    assert text.count("Sources") == 1


# ------------------------------------------- plan-and-execute (cas réel : livrable "[Rédiger ici]")
PLAN_COLO = {"objectif": "Séquence colostrum", "etapes": [{"titre": "Rédiger", "description": "séance", "outil_suggere": ""}]}


def test_planner_stops_when_corpus_does_not_cover_topic():
    llm = FakeLLM(script=[PLAN_COLO])
    ctx = make_ctx(llm)
    ctx.settings.retrieval.min_score = 100.0           # rien n'atteint le seuil
    run = PlanAndExecute(llm, ctx.tools(), rag=ctx.rag).run("Séquence sur la tonte des alpagas du Pérou")
    assert not run.covered and "plan non lancé" in run.final and llm.calls == []     # 0 appel LLM


def test_planner_flags_placeholders_and_builds_sources_from_citations():
    llm = FakeLLM(script=[PLAN_COLO, "Le colostrum est vital [1].",
                          "```markdown\n# Séquence\nLe colostrum apporte des anticorps [1].\n- Cas : [Rédiger ici]\n```"])
    ctx = make_ctx(llm)
    steps = []
    run = PlanAndExecute(llm, ctx.tools(), rag=ctx.rag, on_step=lambda *a: steps.append(a)).run(
        "Séquence sur le colostrum du veau")
    assert run.covered and run.final.startswith("# Séquence") and "## Sources\n[1]" in run.final
    assert any("INCOMPLET" in w for w in run.warnings)
    assert len(steps) == 2 and steps[1][3] == "Le colostrum est vital [1]."          # progression affichée
    assert "Passages du corpus" in llm.calls[1]["messages"][-1].content               # preuves données à l'étape


def test_deliverable_without_citation_is_flagged():
    from transfia.agents.planner import check_deliverable

    assert check_deliverable("Texte sans source.", 5) == ["aucune citation [n] : le contenu n'est relié à aucune source."]


def test_planner_step_with_evidence_is_one_plain_llm_call():
    llm = FakeLLM(script=[PLAN_COLO, "Le colostrum est vital [1].", "# Séquence\nLe colostrum est vital [1]."])
    ctx = make_ctx(llm)
    run = PlanAndExecute(llm, ctx.tools(), rag=ctx.rag).run("Séquence sur le colostrum du veau")
    assert len(llm.calls) == 3 and llm.calls[1]["tools"] is None          # plan, 1 appel par étape, synthèse
    assert "UNIQUEMENT sur les passages" in llm.calls[1]["messages"][-1].content and run.warnings == []


def test_glossary_terms_absent_from_their_source_are_flagged():
    """Cas réel (interface) : glossaire « colostrum » dont 3-NOP, méthane… cités [1] mais absents du passage."""
    gloss = {"titre": "Glossaire", "niveau": "bts", "termes": [
        {"terme": "colostrum", "definition": "Premier lait riche en anticorps.", "source": 1},
        {"terme": "3-NOP", "definition": "Complément qui réduit les émissions de méthane.", "source": 1},
        {"terme": "engraissement", "definition": "Processus par lequel un animal augmente son poids.", "source": 1}]}
    ctx = make_ctx(FakeLLM(script=[gloss]))
    hit = ctx.engine.search("colostrum anticorps veau", k=1)
    res = ctx.generator.generate("glossaire", "le colostrum", "bts", passages=hit)
    assert any("ABSENT" in w and "3-NOP" in w and "engraissement" in w for w in res.warnings)


def test_generate_content_result_is_returned_directly():
    """Cas réel : l'agent réécrivait la fiche générée en texte libre (structure, sources et contrôles perdus)."""
    llm = FakeLLM(script=[[ToolCall("generate_content", {"kind": "quiz", "topic": "le colostrum"}, "c1")], QUIZ_OK])
    res = make_ctx(llm).agent().run("Prépare un QCM sur le colostrum")
    assert res.stopped_reason == "return_direct" and res.answer.startswith("# Colostrum") and "## Sources" in res.answer


def test_text_agent_cannot_invent_observations():
    from transfia.agents.react import ReActAgent

    fake = 'Thought: je cherche\nAction: search_corpus\nAction Input: {"query": "méthane"}\nObservation: Je trouve plein de choses inventées.'
    llm = FakeLLM(script=[fake, "Thought: fini\nFinal Answer: Le méthane vient du rumen."])
    res = ReActAgent(llm, make_ctx(llm).tools(), mode="text").run("D'où vient le méthane ?")
    assert "inventées" not in llm.calls[1]["messages"][-2].content and res.tools_used == ["search_corpus"]



def test_level_synonyms_are_repaired():
    from transfia.agents.tools import repair_arguments

    t = make_ctx(FakeLLM()).tools().tools["generate_content"]
    args, note = repair_arguments(t, {"kind": "fiche-synthese", "topic": "caprins", "niveau": "bac+2"})   # cas réel
    assert args["niveau"] == "bts" and args["kind"] == "fiche" and "bac+2" in note



def test_editorial_prunes_persistent_problems_ignores_years_and_falls_back_reviewer():
    """Cas réel : relecteur llama illisible 3 fois ; « substances protectrices » signalée 3 fois, jamais corrigée ;
    fausse alerte « chiffres perdus : 2012, 2021, 2024 » (années des références)."""
    passage = "Une étude (Lagarde, 2024) a montré que 20 % des colostrums de vaches Charolaise étaient trop pauvres."
    draft = ("Le colostrum contient des substances protectrices magiques et merveilleuses [1]. Une étude (Lagarde, 2024) "
             "a montré que 20 % des colostrums de vaches Charolaise étaient trop pauvres [1].")
    writer = FakeLLM(script=[draft, {"problemes": [], "valide": True}, draft, {"problemes": [], "valide": True},
                             "20 % des colostrums de vaches Charolaise sont trop pauvres."])
    reviewer = FakeLLM(responder=lambda m, t, sc: "pas du JSON")                 # relecteur illisible
    ctx = make_ctx()
    hit = ctx.engine.search("colostrum", k=1)
    hit[0].chunk.text = passage
    st = EditorialWorkflow(writer, reviewer, skills=ctx.skills, max_rounds=2).run("colostrum", "grand_public", hit)
    assert any("repli sur le modèle du rédacteur" in h for h in st.history)
    assert any(h.startswith("élagage (1") for h in st.history) and "magiques" not in st.draft
    assert not any("PERDUS" in w for w in st.final_warnings)                    # 2024 = une année, pas une perte


def test_finetuned_model_is_called_with_its_training_prompt():
    """Cas réel : transfia-qcm, appelé avec le prompt habituel (skill + exemple + 5 passages), a recopié
    les choix de l'exemple de style. Un modèle fine-tuné s'appelle comme à l'entraînement."""
    from transfia.generation.generator import FINETUNE_QCM_SYSTEM

    llm = FakeLLM(script=[QUIZ_OK])
    gen = make_ctx(llm).generator
    gen.model = "transfia-qcm:latest"
    gen.generate("quiz", "le colostrum", "bts")
    msgs = llm.calls[0]["messages"]
    assert msgs[0].role == "system" and msgs[0].content == FINETUNE_QCM_SYSTEM
    assert msgs[1].content.startswith("Niveau : bts\nPassage [1] : ") and "Savoir-faire" not in msgs[1].content
