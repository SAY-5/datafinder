"""v2: agent.stream() and the SSE endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from datafinder.agent import Agent, AgentConfig, FakeChatClient, fake_final, fake_tool_call
from datafinder.api import build_app
from datafinder.seed import populate
from datafinder.store import MemoryStore


def _agent(responses: list[dict]) -> Agent:
    store = MemoryStore()
    populate(store)
    return Agent(
        store=store,
        chat=FakeChatClient(responses=responses),
        config=AgentConfig(max_refinements=0),
    )


def test_stream_emits_normalize_route_then_tool_events():
    a = _agent([
        fake_tool_call("semantic_search", {"query": "knee", "k": 3}),
        fake_final("ds_oai is the best fit."),
    ])
    events = list(a.stream("knee MRI cohort"))
    types = [e["type"] for e in events]
    assert types[0] == "normalize"
    assert types[1] == "route"
    # tool_call must precede tool_result.
    tc_idx = types.index("tool_call")
    tr_idx = types.index("tool_result")
    assert tc_idx < tr_idx
    # answer + done at the end, in order.
    assert types[-2] == "answer"
    assert types[-1] == "done"


def test_stream_normalize_carries_hints():
    a = _agent([
        fake_tool_call("metadata_filter", {"min_subjects": 100}),
        fake_final("ds_oai"),
    ])
    events = list(a.stream("knee MRI, at least 100 patients"))
    n = next(e for e in events if e["type"] == "normalize")
    assert n["hints"].get("min_subjects") == 100


def test_stream_emits_refine_event():
    a = _agent([
        fake_tool_call("semantic_search", {"query": "x"}),
        fake_final("Not sure."),
        fake_tool_call("semantic_search", {"query": "y"}),
        fake_final("ds_oai matches."),
    ])
    a.config.max_refinements = 1
    events = list(a.stream("vague"))
    assert any(e["type"] == "refine" for e in events)
    refine = next(e for e in events if e["type"] == "refine")
    assert refine["attempt"] == 1


def test_sse_endpoint_returns_event_stream():
    store = MemoryStore()
    chat = FakeChatClient(responses=[
        fake_tool_call("semantic_search", {"query": "knee"}),
        fake_final("ds_oai best fit."),
    ])
    app = build_app(store=store, chat=chat, config=AgentConfig(max_refinements=0))
    client = TestClient(app)
    with client.stream("POST", "/v1/ask/stream", json={"query": "knee MRI"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        # Collect the body and parse SSE frames.
        body = b"".join(resp.iter_bytes())
    text = body.decode()
    # Verify each event type appears as an SSE 'event:' line.
    for ev in ("normalize", "route", "tool_call", "tool_result", "answer", "done"):
        assert f"event: {ev}" in text, f"missing event {ev}"
    # End sentinel.
    assert "event: end" in text


def test_done_event_carries_run_id_and_grounded_flag():
    a = _agent([
        fake_tool_call("semantic_search", {"query": "knee"}),
        fake_final("ds_oai is the best match."),
    ])
    events = list(a.stream("knee"))
    done = next(e for e in events if e["type"] == "done")
    assert done["run_id"].startswith("r_")
    assert done["grounded"] is True
    assert done["refinements"] == 0
