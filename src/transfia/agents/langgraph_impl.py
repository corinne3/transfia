"""Les mêmes agents, réécrits avec LangGraph (F8, F9) — pour COMPARER avec la version from scratch.

Ce que LangGraph apporte par rapport à nos boucles maison :
- un graphe d'états EXPLICITE (nœuds, arêtes, arêtes conditionnelles), visualisable ;
- des CHECKPOINTS : l'état est sauvegardé à chaque étape (reprise après erreur, "voyage dans le temps") ;
- l'INTERRUPTION humaine (human-in-the-loop) : on met le graphe en pause, un humain valide, on reprend.

Prérequis : pip install -e ".[agents]".
⚠️ Ce module n'a pas pu être exécuté dans l'environnement de création du projet (LangGraph absent) :
l'API de LangGraph évolue vite ; en cas d'erreur, compare avec la documentation officielle.
"""
from __future__ import annotations

import json
from typing import Annotated, TypedDict


def _require():
    try:
        import langgraph  # noqa: F401
    except ImportError as exc:
        raise ImportError('pip install -e ".[agents]" pour utiliser LangGraph') from exc


# --------------------------------------------------------------------- ReAct
class AgentState(TypedDict):
    messages: Annotated[list, "historique (liste de transfia.llm.base.Message)"]
    steps: int


def build_react_graph(llm, registry, system_prompt: str, max_steps: int = 8):
    _require()
    from langgraph.graph import END, START, StateGraph

    from transfia.llm.base import Message

    def call_model(state: AgentState) -> dict:
        resp = llm.chat(state["messages"], tools=registry.schemas())
        msg = Message("assistant", resp.content, tool_calls=resp.tool_calls)
        return {"messages": state["messages"] + [msg], "steps": state["steps"] + 1}

    def call_tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        new = [Message("tool", registry.call(c.name, c.arguments), name=c.name, tool_call_id=c.id)
               for c in last.tool_calls]
        return {"messages": state["messages"] + new, "steps": state["steps"]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if last.tool_calls and state["steps"] < max_steps else END

    g = StateGraph(AgentState)
    g.add_node("model", call_model)
    g.add_node("tools", call_tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    graph = g.compile()

    def run(task: str) -> str:
        out = graph.invoke({"messages": [Message("system", system_prompt), Message("user", task)], "steps": 0})
        return out["messages"][-1].content

    return graph, run


# ------------------------------------------------- atelier éditorial + pause humaine
class EditorialGraphState(TypedDict, total=False):
    topic: str
    niveau: str
    passages: str
    draft: str
    problems: list
    rounds: int
    final: str
    approved: bool


def build_editorial_graph(workflow, with_human_review: bool = True):
    """Réutilise les nœuds de ``EditorialWorkflow`` (multi_agent.py) dans un graphe LangGraph.

    Avec ``with_human_review`` : le graphe s'INTERROMPT avant la vulgarisation ; un humain lit le
    brouillon validé et reprend l'exécution avec ``Command(resume=True/False)``.
    """
    _require()
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    from transfia.agents.multi_agent import EditorialState

    def to_state(s: EditorialGraphState, passages) -> EditorialState:
        st = EditorialState(s["topic"], s["niveau"], passages)
        st.draft, st.rounds = s.get("draft", ""), s.get("rounds", 0)
        return st

    def make(passages):
        def write(s):
            st = workflow.node_write(to_state(s, passages))
            return {"draft": st.draft, "rounds": st.rounds}

        def review(s):
            st = workflow.node_review(to_state(s, passages))
            return {"problems": [p.model_dump() for p in st.open_problems]}

        def route(s):
            if not s.get("problems"):
                return "human" if with_human_review else "simplify"
            return "write" if s.get("rounds", 0) < workflow.max_rounds else ("human" if with_human_review else "simplify")

        def human(s):
            decision = interrupt({"draft": s["draft"], "question": "Valider ce brouillon avant vulgarisation ?"})
            return {"approved": bool(decision)}

        def simplify(s):
            if with_human_review and not s.get("approved", True):
                return {"final": "Brouillon refusé par l'humain."}
            st = workflow.node_simplify(to_state(s, passages))
            return {"final": st.final}

        g = StateGraph(EditorialGraphState)
        for name, fn in [("write", write), ("review", review), ("human", human), ("simplify", simplify)]:
            g.add_node(name, fn)
        g.add_edge(START, "write")
        g.add_edge("write", "review")
        g.add_conditional_edges("review", route, {"write": "write", "human": "human", "simplify": "simplify"})
        g.add_edge("human", "simplify")
        g.add_edge("simplify", END)
        return g.compile(checkpointer=MemorySaver())

    return make


def demo_human_in_the_loop(workflow, topic: str, niveau: str, passages) -> dict:
    """Exemple d'utilisation : lancer, s'arrêter à la pause, reprendre."""
    from langgraph.types import Command

    graph = build_editorial_graph(workflow)(passages)
    cfg = {"configurable": {"thread_id": "demo-1"}}
    first = graph.invoke({"topic": topic, "niveau": niveau, "rounds": 0}, cfg)
    print("⏸️  Pause humaine :", json.dumps(first.get("__interrupt__", ""), default=str)[:400])
    return graph.invoke(Command(resume=True), cfg)
