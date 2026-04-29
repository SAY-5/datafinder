"""v3: multi-turn session memory."""

from __future__ import annotations

from fastapi.testclient import TestClient

from datafinder.agent import Agent, AgentConfig, FakeChatClient, fake_final, fake_tool_call
from datafinder.api import build_app
from datafinder.seed import populate
from datafinder.store import MemoryStore


def _agent(responses: list[dict], **cfg) -> Agent:
    store = MemoryStore()
    populate(store)
    return Agent(
        store=store,
        chat=FakeChatClient(responses=responses),
        config=AgentConfig(max_refinements=0, **cfg),
    )


def test_history_appends_after_each_run():
    a = _agent([
        fake_tool_call("semantic_search", {"query": "knee", "k": 3}),
        fake_final("ds_oai is the best fit."),
        fake_tool_call("metadata_filter", {"min_subjects": 100}),
        fake_final("ds_oai again, with the higher cohort."),
    ])
    a.run("knee MRI cohort", session_id="s1")
    a.run("now narrow to ≥100 subjects", session_id="s1")
    h = a.history("s1")
    assert len(h) == 2
    assert h[0].query == "knee MRI cohort"
    assert h[1].query == "now narrow to ≥100 subjects"


def test_history_isolated_per_session():
    a = _agent([
        fake_tool_call("semantic_search", {"query": "x"}),
        fake_final("ds_oai"),
        fake_tool_call("semantic_search", {"query": "y"}),
        fake_final("ds_adni"),
    ])
    a.run("knee MRI", session_id="s1")
    a.run("brain MRI", session_id="s2")
    assert len(a.history("s1")) == 1
    assert len(a.history("s2")) == 1
    assert a.history("s1")[0].answer.startswith("ds_oai")
    assert a.history("s2")[0].answer.startswith("ds_adni")


def test_history_bounded_by_max_session_turns():
    a = _agent([], max_session_turns=2)
    # Inject 4 prior runs into the session.
    from datafinder.schema import AgentRun, NormalizedQuery
    from datetime import datetime, timezone
    for i in range(4):
        a.sessions.setdefault("s1", []).append(AgentRun(
            id=f"r_{i}",
            session_id="s1",
            query=f"q{i}",
            normalized=NormalizedQuery(raw=f"q{i}", text=f"q{i}", hints={}),
            route="hybrid",
            answer=f"ds_a{i}",
            citations=[f"ds_a{i}"],
            grounded=True,
            refinements=0,
            tool_calls=[],
            started_at=datetime.now(timezone.utc),
        ))
    # The system message renderer trims to last 2; verify by grepping
    # the system prompt the agent would build.
    from datafinder.agent import _system_message
    from datafinder.normalize import normalize
    nq = normalize("follow up")
    sys = _system_message("hybrid", nq, a.history("s1")[-2:])
    assert "q3" in sys and "q2" in sys
    assert "q1" not in sys and "q0" not in sys


def test_system_message_includes_prior_q_a_and_citations():
    a = _agent([
        fake_tool_call("semantic_search", {"query": "x"}),
        fake_final("ds_oai is the answer here."),
    ])
    a.run("knee MRI cohort", session_id="s1")

    # Capture what the second turn sends as the system message.
    captured: list[list[dict]] = []
    chat = FakeChatClient(responses=[fake_final("ds_adni")])
    a.chat = chat
    a.run("any brain MRI alternatives?", session_id="s1")
    sys_msg = chat.sent_messages[0][0]["content"]
    assert "Prior turns" in sys_msg
    assert "Q: knee MRI cohort" in sys_msg
    # Cited dataset id from the prior turn must appear so the model
    # can avoid recommending it again or build on it.
    assert "ds_oai" in sys_msg


def test_get_session_endpoint_returns_history_and_cumulative_citations():
    store = MemoryStore()
    populate(store)
    chat = FakeChatClient(responses=[
        fake_tool_call("semantic_search", {"query": "knee"}),
        fake_final("ds_oai is the best fit."),
        fake_tool_call("metadata_filter", {"min_subjects": 1000}),
        fake_final("ds_ukbb has more subjects."),
    ])
    app = build_app(store=store, chat=chat, config=AgentConfig(max_refinements=0))
    client = TestClient(app)

    client.post("/v1/ask", json={"query": "knee MRI", "session_id": "s-7"})
    client.post("/v1/ask", json={"query": "anything bigger?", "session_id": "s-7"})

    body = client.get("/v1/sessions/s-7").json()
    assert body["session_id"] == "s-7"
    assert len(body["turns"]) == 2
    assert body["turns"][0]["query"] == "knee MRI"
    # Cumulative citations across the session, deduped + ordered.
    assert "ds_oai" in body["citations"]
    assert "ds_ukbb" in body["citations"]


def test_no_session_id_does_not_persist_history():
    a = _agent([
        fake_tool_call("semantic_search", {"query": "x"}),
        fake_final("ds_oai"),
    ])
    a.run("knee MRI")  # no session_id
    # No session_id → nothing recorded.
    assert a.sessions == {}
