"""The three tools the agent dispatches to.

Each tool is a Python callable plus a JSON schema (for OpenAI
function-calling). The dispatch layer below maps a tool name +
arg JSON to the right call and packages the result back into a
short string the agent can read.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from typing import Any

from datafinder.schema import DatasetHit, ToolCall
from datafinder.store import Store

# ---- JSON schemas -- shipped to OpenAI as tools= ----------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Find datasets whose description is semantically close "
                "to the query text. Use for free-form / conceptual "
                "questions. Returns up to k hits ranked by similarity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "metadata_filter",
            "description": (
                "Filter datasets by structured metadata. All filters "
                "are AND'd. Returns every match (no limit). Use for "
                "spec-shaped queries with hard constraints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "enum": ["MRI", "CT", "X-ray", "ultrasound", "PET",
                                 "histology", "fundus", "OCT", "EEG", "ECG"],
                    },
                    "anatomy": {"type": "string"},
                    "min_subjects": {"type": "integer", "minimum": 1},
                    "age_min": {"type": "integer", "minimum": 0, "maximum": 120},
                    "annotations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dataset_preview",
            "description": (
                "Fetch full metadata + a few sample rows for one "
                "dataset. Call this on top candidates before answering "
                "so the answer can cite specific column names and "
                "value ranges."
            ),
            "parameters": {
                "type": "object",
                "properties": {"dataset_id": {"type": "string"}},
                "required": ["dataset_id"],
            },
        },
    },
]


# ---- dispatcher --------------------------------------------------------

class ToolRunner:
    """Maps tool name + args to a Store call. Records every dispatch
    as a ToolCall so the agent run is fully audited.

    If `on_event` is supplied, fires `tool_call` before dispatch and
    `tool_result` after, so streaming consumers (the SSE endpoint)
    render the trace live."""

    def __init__(
        self,
        store: Store,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.store = store
        self._calls: list[ToolCall] = []
        self._on_event = on_event

    @property
    def calls(self) -> list[ToolCall]:
        return list(self._calls)

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        idx = len(self._calls)
        if self._on_event is not None:
            self._on_event({"type": "tool_call", "idx": idx, "tool": name, "args": args})
        t0 = time.perf_counter()
        if name == "semantic_search":
            hits = self.store.semantic_search(
                query=str(args.get("query", "")),
                k=int(args.get("k", 5)),
            )
            result = _hits_summary(hits)
        elif name == "metadata_filter":
            hits = self.store.metadata_filter(
                modality=args.get("modality"),
                anatomy=args.get("anatomy"),
                min_subjects=args.get("min_subjects"),
                age_min=args.get("age_min"),
                annotations=args.get("annotations"),
            )
            result = _hits_summary(hits)
        elif name == "dataset_preview":
            d = self.store.detail(str(args.get("dataset_id", "")))
            if d is None:
                result = json.dumps({"error": "not_found"})
            else:
                result = json.dumps({
                    "id": d.dataset.id,
                    "title": d.dataset.title,
                    "modality": d.dataset.modality,
                    "anatomy": d.dataset.anatomy,
                    "subjects": d.dataset.subjects,
                    "age_range": [d.dataset.age_min, d.dataset.age_max],
                    "annotations": d.dataset.annotations,
                    "columns": d.dataset.columns,
                    "sample_rows": d.sample_rows,
                    "citation": d.dataset.citation,
                })
        else:
            result = json.dumps({"error": f"unknown tool: {name}"})

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        summary = _truncate(result, 800)
        self._calls.append(ToolCall(
            idx=idx,
            tool=name,
            args=args,
            result_summary=summary,
            elapsed_ms=elapsed_ms,
        ))
        if self._on_event is not None:
            self._on_event({
                "type": "tool_result", "idx": idx,
                "summary": summary, "elapsed_ms": elapsed_ms,
            })
        return result


def _hits_summary(hits: Iterable[DatasetHit]) -> str:
    items = [h.model_dump() for h in hits]
    return json.dumps({"count": len(items), "hits": items})


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
