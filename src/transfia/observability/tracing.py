"""Traçage "from scratch" (F12), compatible OpenTelemetry en option.

Concept : une requête = une TRACE, composée de SPANS emboîtés (rewriting, retrieval, rerank,
génération, appels d'outils...). Chaque span a un début, une fin, des attributs (entrées,
sorties, tokens) et un parent. C'est exactement le modèle d'OpenTelemetry, que Phoenix,
Langfuse, Cloud Trace ou Application Insights savent afficher.

Usage :
    with span("retrieval", query=q) as s:
        hits = ...
        s.set(n_hits=len(hits))
"""
from __future__ import annotations

import contextvars
import functools
import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("transfia_span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    @property
    def duration_ms(self) -> float | None:
        return None if self.end is None else (self.end - self.start) * 1000

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start": self.start,
            "duration_ms": self.duration_ms,
            "attributes": _jsonable(self.attributes),
            "status": self.status,
            "error": self.error,
        }


def _jsonable(obj: Any, limit: int = 2000) -> Any:
    """Les attributs doivent rester petits et sérialisables (ne pas logger des documents entiers)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, limit) for v in obj[:50]]
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


class Tracer:
    """Collecte les spans en mémoire et les écrit en JSONL (une ligne par span)."""

    def __init__(self, path: Path | None = None, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.finished: list[Span] = []
        self._lock = threading.Lock()
        self._otel = None

    def enable_otel(self, endpoint: str, service_name: str = "transfia") -> bool:
        """Pont optionnel vers OpenTelemetry (Phoenix, Cloud Trace...). Renvoie False si absent."""
        try:
            from opentelemetry import trace as ot_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            return False
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        ot_trace.set_tracer_provider(provider)
        self._otel = ot_trace.get_tracer("transfia")
        return True

    @contextmanager
    def span(self, name: str, **attrs: Any):
        if not self.enabled:
            dummy = Span(name, "-", "-", None, time.perf_counter())
            yield dummy
            return
        parent = _current.get()
        trace_id = parent.trace_id if parent else uuid.uuid4().hex[:16]
        s = Span(name, trace_id, uuid.uuid4().hex[:8], parent.span_id if parent else None,
                 time.perf_counter(), attributes=dict(attrs))
        token = _current.set(s)
        otel_cm = self._otel.start_as_current_span(name) if self._otel else None
        otel_span = otel_cm.__enter__() if otel_cm else None
        try:
            yield s
        except Exception as exc:  # on trace l'erreur puis on la relance
            s.status, s.error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.end = time.perf_counter()
            try:
                _current.reset(token)
            except ValueError:
                # Streaming (SSE) : le serveur web fait avancer le générateur dans un AUTRE contexte
                # (un thread différent à chaque morceau). Le jeton n'y est pas valable : on remet le parent.
                _current.set(parent)
            if otel_span is not None:
                for k, v in _jsonable(s.attributes).items():
                    if isinstance(v, (str, int, float, bool)):
                        otel_span.set_attribute(k, v)
                otel_cm.__exit__(None, None, None)
            self._finish(s)

    def _finish(self, s: Span) -> None:
        with self._lock:
            self.finished.append(s)
            if len(self.finished) > 5000:
                self.finished = self.finished[-2500:]
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    def trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.finished if s.trace_id == trace_id]

    def tree(self, trace_id: str) -> str:
        """Affichage texte de l'arbre des spans (comme dans Phoenix, en plus simple)."""
        spans = self.trace(trace_id)
        children: dict[str | None, list[Span]] = {}
        for s in spans:
            children.setdefault(s.parent_id, []).append(s)
        lines: list[str] = []

        def walk(parent: str | None, depth: int) -> None:
            for s in sorted(children.get(parent, []), key=lambda x: x.start):
                extra = {k: v for k, v in s.attributes.items() if k in ("tokens_in", "tokens_out", "n_hits", "model", "tool")}
                lines.append(f"{'  ' * depth}- {s.name} [{s.duration_ms:.0f} ms] {extra if extra else ''}")
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)


_TRACER = Tracer(enabled=True)


def get_tracer() -> Tracer:
    return _TRACER


def configure_tracer(path: Path | None, enabled: bool = True) -> Tracer:
    _TRACER.path, _TRACER.enabled = path, enabled
    return _TRACER


def span(name: str, **attrs: Any):
    return _TRACER.span(name, **attrs)


def current_trace_id() -> str | None:
    s = _current.get()
    return s.trace_id if s else None


def traced(name: str | None = None) -> Callable:
    """Décorateur : trace automatiquement une fonction."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with span(name or fn.__qualname__):
                return fn(*args, **kwargs)

        return wrapper

    return deco
