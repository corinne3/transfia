"""TOOLS (section 4 du dossier) : des fonctions que le LLM peut DEMANDER d'appeler.

Le LLM ne voit qu'un SCHÉMA par outil : nom + description + JSON schema des paramètres.
Il répond par une demande structurée {"name": ..., "arguments": {...}} ; c'est NOTRE code
qui valide les arguments, exécute la fonction et renvoie le résultat au modèle.

Ce module génère le schéma automatiquement à partir des annotations de type Python
(comme le font les frameworks), et VALIDE les arguments avec Pydantic avant exécution :
un petit modèle produit souvent des arguments mal formés -> on renvoie l'erreur pour qu'il corrige.
"""
from __future__ import annotations

import concurrent.futures
import inspect
import json
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

from pydantic import ValidationError, create_model

from transfia.observability.tracing import span


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict
    arg_model: Any = None
    read_only: bool = True          # moindre privilège (11.5) : on sait quels outils modifient quelque chose
    timeout_s: float = 120.0
    tags: list[str] = field(default_factory=list)
    # return_direct : le résultat de l'outil EST la réponse finale (pas de réécriture par le LLM).
    # Vu en vrai : l'agent appelait generate_content (fiche structurée, vérifiée, sourcée), puis la RÉÉCRIVAIT
    # en texte libre : structure perdue, vocabulaire vide, sources « Bluet et al., 2026 » ×3 sans titre ni DOI.
    return_direct: bool = False

    def schema(self) -> dict:
        """Format "function calling" commun à Ollama, OpenAI, LiteLLM..."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _param_docs(doc: str) -> dict[str, str]:
    """Lit les descriptions de paramètres dans la docstring : lignes "nom: description"."""
    out = {}
    for line in doc.splitlines():
        m = re.match(r"\s*[-*]?\s*(\w+)\s*:\s*(.+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _json_type(tp) -> dict:
    origin = get_origin(tp)
    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}
    if origin is Literal:
        return {"type": "string", "enum": list(get_args(tp))}
    if origin in (list, typing.List):
        (item,) = get_args(tp) or (str,)
        return {"type": "array", "items": _json_type(item)}
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", object)):
        args = [a for a in get_args(tp) if a is not type(None)]
        return _json_type(args[0]) if args else {"type": "string"}
    return {"type": "string"}


def tool(name: str | None = None, read_only: bool = True, timeout_s: float = 120.0, return_direct: bool = False):
    """Décorateur : transforme une fonction Python typée et documentée en Tool."""

    def deco(fn: Callable) -> Tool:
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
        doc = inspect.getdoc(fn) or ""
        summary = doc.split("\n\n")[0].strip()
        pdocs = _param_docs(doc)
        props, required, fields = {}, [], {}
        for pname, p in sig.parameters.items():
            tp = hints.get(pname, str)
            js = _json_type(tp)
            if pname in pdocs:
                js["description"] = pdocs[pname]
            if p.default is inspect.Parameter.empty:
                required.append(pname)
                fields[pname] = (tp, ...)
            else:
                js["default"] = p.default
                fields[pname] = (tp, p.default)
            props[pname] = js
        params = {"type": "object", "properties": props, "required": required}
        model = create_model(f"{fn.__name__}_args", **fields)
        return Tool(name or fn.__name__, summary, fn, params, model, read_only, timeout_s,
                    return_direct=return_direct)

    return deco


def signature(t: "Tool") -> str:
    props, req = t.parameters.get("properties", {}), set(t.parameters.get("required", []))
    return ", ".join(f"{k} ({v.get('type', 'any')}{', obligatoire' if k in req else ''})" for k, v in props.items())


def example_call(t: "Tool") -> str:
    props, req = t.parameters.get("properties", {}), t.parameters.get("required", [])
    demo = {"string": "...", "integer": 1, "number": 1.0, "boolean": True, "array": []}
    return json.dumps({k: (props[k].get("enum") or [demo.get(props[k].get("type"), "...")])[0] for k in req},
                      ensure_ascii=False)


def repair_arguments(t: "Tool", args: dict) -> tuple[dict, str]:
    """TOLÉRANCE (1) : valeur d'enum presque juste (casse, accents, tiret/espace/_) -> la bonne valeur.
    TOLÉRANCE (2) : il manque UN argument obligatoire et il y a UN argument inconnu -> c'est presque sûrement
    le même, mal nommé (vu : load_skill({"skill": ...}) au lieu de {"name": ...}). On renomme, et on le DIT."""
    props, req = t.parameters.get("properties", {}), t.parameters.get("required", [])
    notes = []
    # valeurs de liste fermée (enum) presque justes : "cas-pratique" -> "cas_pratique", "bac pro" -> "bac_pro"
    for k, v in list(args.items()):
        enum = props.get(k, {}).get("enum")
        if enum and isinstance(v, str) and v not in enum:
            match = [e for e in enum if _norm(e) == _norm(v)]
            if not match and _norm(v) in SYNONYMS and SYNONYMS[_norm(v)] in enum:
                match = [SYNONYMS[_norm(v)]]          # vu : "bac+2" (= BTS), "fiche-synthese" (= fiche)
            if len(match) == 1:
                args = {**args, k: match[0]}
                notes.append(f"'{v}' interprété comme '{match[0]}'")
    missing = [k for k in req if k not in args]
    unknown = [k for k in args if k not in props]
    if len(missing) == 1 and len(unknown) == 1:
        args = dict(args)
        args[missing[0]] = args.pop(unknown[0])
        notes.insert(0, f"argument '{unknown[0]}' interprété comme '{missing[0]}'")
    return args, (f"(note : {' ; '.join(notes)})\n" if notes else "")


# Synonymes métier fréquents (valeurs normalisées par _norm) -> valeur attendue par l'outil.
SYNONYMS = {
    "bac+2": "bts", "bac_2": "bts", "licence": "master", "bac+3": "master", "bac+5": "master", "bac_5": "master",
    "lycee": "bac_pro", "bacpro": "bac_pro", "tout_public": "grand_public", "grandpublic": "grand_public",
    "fiche_synthese": "fiche", "fiche_de_synthese": "fiche", "synthese": "fiche", "qcm": "quiz",
    "questionnaire": "quiz", "cas": "cas_pratique", "lexique": "glossaire", "vocabulaire": "glossaire",
}


def _norm(x: str) -> str:
    import unicodedata

    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[\s_\-]+", "_", x).strip("_")


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self.tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        self.tools[t.name] = t

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        return [t.schema() for n, t in self.tools.items() if names is None or n in names]

    def describe(self) -> str:
        return "\n".join(f"- {t.name}({', '.join(t.parameters['properties'])}) : {t.description}"
                         for t in self.tools.values())

    def call(self, name: str, arguments: dict) -> str:
        """Exécute un outil et renvoie TOUJOURS une chaîne (le LLM ne lit que du texte).

        Les erreurs sont renvoyées comme des observations : le modèle peut alors se corriger.
        """
        with span("tool.call", tool=name, arguments=arguments) as s:
            if name not in self.tools:
                s.set(error="unknown_tool")
                return f"ERREUR : outil inconnu '{name}'. Outils disponibles : {', '.join(self.tools)}"
            t = self.tools[name]
            args = {k: v for k, v in (arguments or {}).items() if not k.startswith("_")}
            note = ""
            if t.arg_model:
                args, note = repair_arguments(t, args)
            try:
                validated = t.arg_model(**args).model_dump() if t.arg_model else args
            except ValidationError as exc:
                s.set(error="invalid_arguments")
                # message LISIBLE par un petit modèle : la signature attendue + un exemple (pas un dump Pydantic)
                problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or '?'} : {e['msg']}"
                                     for e in exc.errors(include_url=False))
                return (f"ERREUR : arguments invalides pour {name} ({problems}). "
                        f"Arguments attendus : {signature(t)}. Exemple : {example_call(t)}")
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                result = pool.submit(t.fn, **validated).result(timeout=t.timeout_s)
            except concurrent.futures.TimeoutError:
                s.set(error="timeout")
                return f"ERREUR : l'outil {name} a dépassé {t.timeout_s} s."
            except Exception as exc:
                s.set(error=type(exc).__name__)
                return f"ERREUR pendant {name} : {type(exc).__name__}: {exc}"
            finally:
                pool.shutdown(wait=False)
            out = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            s.set(result_chars=len(out), repaired=bool(note))
            return note + out
