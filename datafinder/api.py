"""FastAPI HTTP layer."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from datafinder.agent import Agent, AgentConfig, ChatClient, FakeChatClient
from datafinder.schema import AgentRun
from datafinder.seed import populate
from datafinder.store import MemoryStore, Store


class AskBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = None


def build_app(
    store: Store | None = None,
    chat: ChatClient | None = None,
    config: AgentConfig | None = None,
) -> FastAPI:
    state: dict[str, Any] = {
        "store": store or _default_store(),
        "chat":  chat  or _default_chat(),
        "config": config or AgentConfig(),
        "runs":   {},  # in-memory run cache; production writes to Postgres
    }
    populated = populate(state["store"])

    app = FastAPI(title="DataFinder", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("DATAFINDER_CORS", "http://localhost:5173").split(","),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "datasets": populated}

    @app.get("/v1/datasets")
    def list_datasets() -> dict[str, Any]:
        ids = state["store"].all_ids()
        items = []
        for did in ids:
            ds = state["store"].get(did)
            if ds is None:
                continue
            items.append({
                "id": ds.id, "title": ds.title,
                "modality": ds.modality, "anatomy": ds.anatomy,
                "subjects": ds.subjects,
            })
        return {"items": items}

    @app.get("/v1/datasets/{ds_id}")
    def get_dataset(ds_id: str) -> dict[str, Any]:
        d = state["store"].detail(ds_id)
        if d is None:
            raise HTTPException(404, "dataset not found")
        return d.model_dump()

    @app.post("/v1/ask")
    def ask(body: AskBody) -> dict[str, Any]:
        agent = Agent(
            store=state["store"],
            chat=state["chat"],
            config=state["config"],
        )
        run: AgentRun = agent.run(body.query, session_id=body.session_id)
        state["runs"][run.id] = run
        return run.model_dump(mode="json")

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        r = state["runs"].get(run_id)
        if r is None:
            raise HTTPException(404, "run not found")
        return r.model_dump(mode="json")

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return _ROOT_HTML

    return app


def _default_store() -> MemoryStore:
    return MemoryStore()


def _default_chat() -> ChatClient:
    """If OPENAI_API_KEY is set, return a real client; otherwise a
    deterministic fake that picks a tool by route + emits a final
    answer that mentions the top hit's id."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from datafinder.openai_client import OpenAIChatClient
            return OpenAIChatClient(model=os.environ.get("DATAFINDER_MODEL", "gpt-4o-mini"))
        except Exception:
            pass
    return _StubChatClient()


class _StubChatClient:
    """Deterministic stand-in: routes based on the user message,
    calls one tool, then emits a final answer naming the top hit."""

    def __init__(self) -> None:
        self._step = 0
        self._cached_top: str | None = None

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Round 1: pick a tool based on a keyword hint in the system msg.
        sys = next((m for m in messages if m["role"] == "system"), None)
        sys_text = sys["content"] if sys else ""
        if self._step == 0:
            self._step += 1
            if "metadata_filter first" in sys_text:
                tool = "metadata_filter"
                args: dict[str, Any] = {"modality": "MRI"}
            else:
                tool = "semantic_search"
                args = {"query": _last_user(messages), "k": 5}
            import json as _j
            return {
                "choices": [{"message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "tc_stub",
                        "type": "function",
                        "function": {"name": tool, "arguments": _j.dumps(args)},
                    }],
                }}],
            }
        # Round 2: read the tool result we just appended; pick a top
        # dataset id and emit a final answer that mentions it.
        tool_msg = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
        top = ""
        if tool_msg:
            try:
                import json as _j
                payload = _j.loads(tool_msg.get("content", "{}"))
                hits = payload.get("hits", [])
                if hits:
                    top = hits[0].get("dataset_id", "")
            except Exception:
                pass
        if top:
            return {"choices": [{"message": {
                "role": "assistant",
                "content": f"Top match: {top}. See dataset preview for details.",
            }}]}
        return {"choices": [{"message": {"role": "assistant", "content": "No matching dataset."}}]}


def _last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


_ROOT_HTML = """<!doctype html>
<html><body style="font-family: ui-monospace, monospace; max-width: 720px; margin: 40px auto; padding: 0 16px;">
<h1>DataFinder API</h1>
<p>POST <code>/v1/ask</code> with <code>{"query": "..."}</code>.</p>
<p>The React UI runs at <a href="http://localhost:5173">localhost:5173</a>.</p>
</body></html>"""


# A module-level app for `uvicorn datafinder.api:app`.
app = build_app()


# Re-export for tests that want the fake bare.
__all__ = ["AskBody", "FakeChatClient", "app", "build_app"]
