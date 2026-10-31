"""Clients LLM : Ollama (HTTP simulé), sorties structurées + retry, cache, conversions de messages."""
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from transfia.llm.base import Message, StructuredOutputError, ToolCall, chat_structured, extract_json
from transfia.llm.cache import CachedLLM, SemanticCache
from transfia.llm.fake import FakeLLM
from transfia.llm.ollama import OllamaClient
from transfia.embeddings.embedders import HashingEmbedder


class FakeHTTP:
    """Simule le serveur Ollama : on vérifie EXACTEMENT ce que notre client envoie."""

    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.sent = None

    def post(self, url, json=None, timeout=None, stream=False):
        self.sent = {"url": url, "json": json}
        return self

    def json(self):
        return self.payload

    def iter_lines(self):
        for piece in ["Bon", "jour"]:
            yield (__import__("json").dumps({"message": {"content": piece}, "done": False})).encode()
        yield b'{"message": {"content": ""}, "done": true, "eval_count": 2}'


def test_ollama_payload_tools_schema_and_usage():
    http = FakeHTTP({"message": {"content": "", "tool_calls": [{"function": {"name": "search_corpus",
                                                                              "arguments": {"query": "veau"}}}]},
                     "prompt_eval_count": 120, "eval_count": 15})
    c = OllamaClient("qwen2.5:1.5b", session=http, num_ctx=2048)
    resp = c.chat([Message("user", "salut")], tools=[{"type": "function"}], schema={"type": "object"})
    sent = http.sent["json"]
    assert http.sent["url"].endswith("/api/chat") and sent["format"] == {"type": "object"}
    assert sent["options"]["num_ctx"] == 2048 and sent["stream"] is False
    assert resp.tool_calls[0].name == "search_corpus" and resp.tool_calls[0].arguments == {"query": "veau"}
    assert resp.usage.prompt_tokens == 120 and resp.usage.completion_tokens == 15


def test_ollama_streaming():
    c = OllamaClient(session=FakeHTTP({}))
    assert "".join(c.stream([Message("user", "x")])) == "Bonjour"


def test_message_conversions():
    m = Message("assistant", "", tool_calls=[ToolCall("f", {"a": 1}, "id1")])
    assert m.to_ollama()["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert json.loads(m.to_openai()["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


class Out(BaseModel):
    reponse: str
    note: int = Field(ge=1, le=5)


def test_extract_json_from_noisy_text():
    assert extract_json('Voici : ```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('blabla {"a": 2} fin') == {"a": 2}


def test_structured_output_retries_with_the_error():
    llm = FakeLLM(script=['{"reponse": "ok", "note": 9}', '{"reponse": "ok", "note": 4}'])
    out, attempts = chat_structured(llm, [Message("user", "note")], Out)
    assert out.note == 4 and attempts == 2
    assert "Erreur" in llm.calls[1]["messages"][-1].content           # l'erreur est renvoyée au modèle
    assert llm.calls[0]["schema"]["properties"]["note"]["maximum"] == 5


def test_structured_output_gives_up():
    llm = FakeLLM(script=["pas du json"] * 3)
    try:
        chat_structured(llm, [Message("user", "x")], Out, retries=2)
        raise AssertionError("aurait dû échouer")
    except StructuredOutputError as exc:
        assert len(exc.attempts) == 3


def test_exact_cache_avoids_second_call():
    inner = FakeLLM(script=["réponse 1", "réponse 2"])
    c = CachedLLM(inner, Path(tempfile.mkdtemp()) / "c.sqlite")
    a = c.chat([Message("user", "même question")])
    b = c.chat([Message("user", "même question")])
    assert a.content == b.content == "réponse 1" and len(inner.calls) == 1 and c.hits == 1


def test_semantic_cache_threshold():
    sc = SemanticCache(HashingEmbedder(256), threshold=0.9)
    sc.put("Quels sont les facteurs de risque des mammites ?", "réponse A")
    assert sc.get("Quels sont les facteurs de risque des mammites ?") == "réponse A"
    assert sc.get("Comment réduire le méthane ?") is None


def test_thinking_models_have_thinking_disabled():
    """qwen3:4b (Apache 2.0, gros modèle open source) écrit <think>…</think> : ça casserait le JSON."""
    from transfia.llm.base import Message
    from transfia.llm.ollama import OllamaClient, is_thinking_model, strip_think

    assert is_thinking_model("qwen3:4b") and not is_thinking_model("qwen2.5:3b")
    assert strip_think("<think>je réfléchis…</think>\n{\"a\": 1}") == '{"a": 1}'

    class Resp:
        status_code = 200
        def json(self):
            return {"message": {"content": "<think>hmm</think>Le rumen."}, "prompt_eval_count": 1, "eval_count": 1}

    class Session:
        def __init__(self):
            self.payloads = []
        def post(self, url, json=None, timeout=None, stream=False):
            self.payloads.append(json)
            return Resp()

    sess = Session()
    out = OllamaClient("qwen3:4b", session=sess).chat([Message("user", "?")])
    assert sess.payloads[0]["think"] is False and out.content == "Le rumen."
    OllamaClient("qwen2.5:1.5b", session=sess).chat([Message("user", "?")])
    assert "think" not in sess.payloads[1]
