"""Client MCP -> outils de l'agent (section 4).

Le pont démarre un serveur MCP (processus séparé, transport stdio), découvre ses outils
(``list_tools``) et les convertit en ``Tool`` pour notre ``ToolRegistry`` : l'agent ReAct
les utilise sans savoir qu'ils passent par MCP. C'est tout l'intérêt du protocole :
brancher un nouveau serveur = de nouveaux outils, sans changer le code de l'agent.

Le SDK MCP est asynchrone (asyncio) ; notre agent est synchrone. On fait donc tourner la session
MCP dans une boucle asyncio, dans un thread dédié, et on lui envoie les appels par une file.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading

from transfia.agents.tools import Tool, ToolRegistry


class MCPToolBridge:
    def __init__(self, command: str = sys.executable, args: list[str] | None = None, env: dict | None = None,
                 timeout_s: float = 180.0):
        self.command = command
        self.args = args if args is not None else ["-m", "transfia.mcp_server.server"]
        self.env = env
        self.timeout_s = timeout_s
        self.tools: list = []
        self._loop = asyncio.new_event_loop()
        self._ready = concurrent.futures.Future()
        self._queue: asyncio.Queue | None = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.result(timeout=timeout_s)          # lève l'erreur de connexion éventuelle

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.tools = (await session.list_tools()).tools
                    self._queue = asyncio.Queue()
                    self._ready.set_result(True)
                    while True:
                        item = await self._queue.get()
                        if item is None:
                            return
                        name, arguments, fut = item
                        try:
                            res = await session.call_tool(name, arguments)
                            fut.set_result(_result_text(res))
                        except Exception as exc:
                            fut.set_exception(exc)
        except Exception as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)

    def call(self, name: str, arguments: dict) -> str:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (name, arguments, fut))
        return fut.result(timeout=self.timeout_s)

    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        for t in self.tools:
            def fn(_name=t.name, **kwargs):
                return self.call(_name, kwargs)

            # v1 : inputSchema ; v2 : input_schema
            raw = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
            schema = dict(raw or {"type": "object", "properties": {}})
            reg.register(Tool(name=t.name, description=t.description or "", fn=fn, parameters=schema,
                              arg_model=None, timeout_s=self.timeout_s, tags=["mcp"]))
        return reg

    def close(self) -> None:
        if self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        self._thread.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _result_text(res) -> str:
    parts = []
    for c in getattr(res, "content", []) or []:
        text = getattr(c, "text", None)
        if text is not None:
            parts.append(text)
    structured = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if not parts and structured is not None:
        import json

        return json.dumps(structured, ensure_ascii=False)
    return "\n".join(parts)
