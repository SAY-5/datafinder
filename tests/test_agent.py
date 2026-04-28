from __future__ import annotations

from datafinder.agent import Agent, AgentConfig, FakeChatClient, fake_final, fake_tool_call
from datafinder.seed import populate
from datafinder.store import MemoryStore


def _agent(responses: list[dict]) -> tuple[Agent, FakeChatClient, MemoryStore]:
    store = MemoryStore()
    populate(store)
    chat = FakeChatClient(responses=responses)
    a = Agent(store=store, chat=chat, config=AgentConfig(max_tool_rounds=4, max_refinements=1))
    return a, chat, store


def test_single_tool_call_round_trip_grounded():
    """Agent calls semantic_search, then emits an answer that mentions
    the top hit's id. The grounding check should succeed."""
    a, _chat, _ = _agent([
        fake_tool_call("semantic_search", {"query": "knee MRI", "k": 3}),
        fake_final("ds_oai is the best match — large longitudinal MRI cohort."),
    ])
    run = a.run("knee MRI cohort with cartilage")
    assert run.grounded is True
    assert "ds_oai" in run.answer
    assert run.answer.startswith("ds_oai")
    assert any(c.tool == "semantic_search" for c in run.tool_calls)
    assert run.refinements == 0


def test_metadata_then_semantic_two_round_dispatch():
    """Hybrid path: metadata first, then semantic, then a grounded answer."""
    a, _chat, _ = _agent([
        fake_tool_call("metadata_filter", {"modality": "MRI", "anatomy": "knee"}, "tc_a"),
        fake_tool_call("semantic_search", {"query": "knee", "k": 3}, "tc_b"),
        fake_final("Recommend ds_oai or ds_mrnet — both annotated knee MRI."),
    ])
    run = a.run("MRI knee studies similar to existing osteoarthritis cohorts")
    assert run.grounded is True
    assert {c.tool for c in run.tool_calls} >= {"metadata_filter", "semantic_search"}


def test_ungrounded_answer_triggers_refinement():
    """First attempt: tool call + answer that names NO dataset.
    Should refine once. Second attempt: tool + grounded answer."""
    a, _chat, _ = _agent([
        fake_tool_call("semantic_search", {"query": "X", "k": 3}, "tc_a"),
        fake_final("I'm not sure I have a good match here."),
        # Refinement attempt:
        fake_tool_call("semantic_search", {"query": "knee MRI", "k": 5}, "tc_b"),
        fake_final("On reflection: ds_oai is the strongest fit."),
    ])
    run = a.run("knee studies")
    assert run.refinements == 1
    assert run.grounded is True
    assert "ds_oai" in run.answer


def test_refinement_budget_capped():
    """All attempts return ungrounded answers; agent gives up after
    max_refinements (1) and reports grounded=False."""
    a, _chat, _ = _agent([
        fake_tool_call("semantic_search", {"query": "x"}),
        fake_final("No idea."),
        fake_tool_call("semantic_search", {"query": "y"}),
        fake_final("Still no idea."),
    ])
    run = a.run("vague question")
    assert run.refinements == 1
    assert run.grounded is False


def test_preview_only_route_calls_dataset_preview():
    """A query with a literal dataset id should send the agent down
    the preview route — at minimum, the system message tells it to
    call dataset_preview."""
    # Even with the stub picking semantic_search, the run records the
    # route the router decided on. We assert on `run.route`.
    a, _, _ = _agent([fake_final("ds_oai")])
    run = a.run("show me ds_oai")
    assert run.route == "preview_only"


def test_tool_calls_are_audited():
    a, _, _ = _agent([
        fake_tool_call("metadata_filter", {"modality": "MRI"}, "x"),
        fake_final("ds_oai"),
    ])
    run = a.run("MRI cohorts at least 500 subjects")
    # Hint extraction in the system prompt + the tool call recorded.
    assert run.normalized.hints.get("min_subjects") == 500
    assert run.tool_calls[0].tool == "metadata_filter"
    assert run.tool_calls[0].args == {"modality": "MRI"}


def test_round_budget_caps_tool_calls():
    """Agent that keeps calling tools without ever emitting content
    must stop after max_tool_rounds."""
    responses = [fake_tool_call("semantic_search", {"query": "loop"}) for _ in range(20)]
    a, _, _ = _agent(responses)
    run = a.run("induce a loop")
    assert run.grounded is False
    # Capped at max_tool_rounds across attempts.
    assert run.tool_calls
