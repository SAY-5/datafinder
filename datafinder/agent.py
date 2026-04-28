"""Agent loop with three autonomous decision points.

  decide()          → (route, system_message)
  run()             → loop over OpenAI tool-call rounds, dispatch,
                      collect, ground, optionally refine, return.

The OpenAI client is injected — production wires the real one,
tests pass a `FakeChatClient` that scripts the conversation. The
loop has zero direct imports of the openai package so the rest of
the code runs offline and in CI.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from datafinder.normalize import normalize
from datafinder.router import route as route_query
from datafinder.schema import AgentRun, NormalizedQuery, Route
from datafinder.store import Store
from datafinder.tools import TOOL_SCHEMAS, ToolRunner

# ---- chat-client protocol --------------------------------------------

class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


# ---- agent ------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    max_tool_rounds: int = 6
    max_refinements: int = 2


@dataclass
class Agent:
    store: Store
    chat: ChatClient
    config: AgentConfig = field(default_factory=AgentConfig)

    def run(self, query: str, session_id: str | None = None) -> AgentRun:
        nq = normalize(query)
        r = route_query(nq)
        return self._run_with_normalized(query, nq, r, session_id)

    def _run_with_normalized(
        self,
        original_query: str,
        nq: NormalizedQuery,
        route: Route,
        session_id: str | None,
    ) -> AgentRun:
        run = AgentRun(
            id=f"r_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            query=original_query,
            normalized=nq,
            route=route,
            started_at=datetime.now(UTC),
        )
        runner = ToolRunner(self.store)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_message(route, nq)},
            {"role": "user", "content": nq.text},
        ]

        # Up to max_refinements + 1 attempts. Each attempt may itself
        # round-trip with OpenAI multiple times for tool calls.
        attempts = 0
        while attempts <= self.config.max_refinements:
            answer, citations = self._one_attempt(messages, runner)
            if _is_grounded(answer, runner) or attempts == self.config.max_refinements:
                run.answer = answer
                run.citations = citations
                run.grounded = _is_grounded(answer, runner)
                run.refinements = attempts
                run.tool_calls = runner.calls
                run.ended_at = datetime.now(UTC)
                return run
            # Refine: tighter system message + same user query.
            attempts += 1
            messages = [
                {"role": "system", "content": _refinement_system(route, nq, attempts)},
                {"role": "user", "content": nq.text},
            ]

        # Unreachable but mypy-pleasing.
        run.tool_calls = runner.calls
        run.ended_at = datetime.now(UTC)
        return run

    def _one_attempt(
        self,
        messages: list[dict[str, Any]],
        runner: ToolRunner,
    ) -> tuple[str, list[str]]:
        """One round of OpenAI calls until the model emits a final
        answer. Returns (answer_text, citations)."""
        cited: set[str] = set()
        rounds = 0
        local_messages = list(messages)
        while rounds < self.config.max_tool_rounds:
            resp = self.chat.complete(messages=local_messages, tools=TOOL_SCHEMAS)
            choice = resp["choices"][0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                content = (msg.get("content") or "").strip()
                # Citations: every dataset_id that appeared in any
                # tool result is a possible citation; the model is
                # encouraged to mention them by id in its answer.
                for tc in runner.calls:
                    if tc.tool in ("semantic_search", "metadata_filter"):
                        try:
                            payload = json.loads(tc.result_summary.rstrip("…"))
                        except json.JSONDecodeError:
                            continue
                        for h in payload.get("hits", []):
                            did = h.get("dataset_id", "")
                            if did and did in content:
                                cited.add(did)
                    elif tc.tool == "dataset_preview":
                        did = tc.args.get("dataset_id")
                        if did and did in content:
                            cited.add(did)
                return content, sorted(cited)

            # Append the assistant's tool-call message verbatim, then
            # one tool message per call.
            local_messages.append(msg)
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = runner.dispatch(name, args)
                local_messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", f"tc_{rounds}"),
                    "name": name,
                    "content": result,
                })
            rounds += 1

        # Hit the round budget with no final content — return what we
        # have. The grounding check will catch this and (maybe) retry.
        return "(agent: tool-round budget exhausted)", sorted(cited)


# ---- helpers ----------------------------------------------------------

_BASE_SYSTEM = (
    "You are a research-dataset discovery assistant. Use the tools to "
    "find datasets that satisfy the user's query. Cite each dataset "
    "you mention by its id. Prefer one or two strong recommendations "
    "with a short justification over a long list of weak matches."
)


def _system_message(route: Route, nq: NormalizedQuery) -> str:
    hint_block = ""
    if nq.hints:
        hint_block = (
            "\n\nExtracted constraints from the query (pass directly "
            "to metadata_filter):\n"
            + json.dumps(nq.hints, indent=2)
        )
    route_block = {
        "semantic":     "Start with semantic_search. Use metadata_filter only if you need to disambiguate.",
        "metadata":     "Start with metadata_filter. Use semantic_search only if metadata returns ≥3 candidates.",
        "hybrid":       "Call metadata_filter first to narrow the candidate set, then semantic_search to rank.",
        "preview_only": "The user referenced a dataset by id. Call dataset_preview directly.",
    }[route]
    return _BASE_SYSTEM + "\n\n" + route_block + hint_block


def _refinement_system(route: Route, nq: NormalizedQuery, attempt: int) -> str:
    return (
        _system_message(route, nq)
        + f"\n\n[refinement #{attempt}] The previous answer wasn't grounded "
        "in any tool result. Try a different tool sequence or broaden "
        "your filters."
    )


def _is_grounded(answer: str, runner: ToolRunner) -> bool:
    """An answer is grounded if it references at least one dataset id
    that appeared in some tool result. The agent is instructed to
    cite by id; if it didn't, retry with a tighter system message."""
    if not answer or answer.startswith("(agent:"):
        return False
    seen: set[str] = set()
    for tc in runner.calls:
        if tc.tool == "dataset_preview":
            did = tc.args.get("dataset_id")
            if isinstance(did, str):
                seen.add(did)
            continue
        try:
            payload = json.loads(tc.result_summary.rstrip("…"))
        except json.JSONDecodeError:
            continue
        for h in payload.get("hits", []):
            did = h.get("dataset_id")
            if isinstance(did, str):
                seen.add(did)
    return any(did in answer for did in seen)


# ---- a ChatClient stub used by tests ----------------------------------

@dataclass
class FakeChatClient:
    """Scripted ChatClient: each call pops the next response off a
    list. Lets tests script multi-round conversations without OpenAI."""

    responses: list[dict[str, Any]] = field(default_factory=list)
    sent_messages: list[list[dict[str, Any]]] = field(default_factory=list)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.sent_messages.append(list(messages))
        if not self.responses:
            return {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        return self.responses.pop(0)


# Convenience factory: build a tool-call response.
def fake_tool_call(tool: str, args: dict[str, Any], call_id: str = "tc_1") -> dict[str, Any]:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool,
                        "arguments": json.dumps(args),
                    },
                }],
            },
        }],
    }


def fake_final(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# Used by tests; expose for ergonomic imports.
__all__ = [
    "Agent",
    "AgentConfig",
    "ChatClient",
    "FakeChatClient",
    "fake_final",
    "fake_tool_call",
]


# Suppress unused-symbol warnings.
_ = (Callable, time)
