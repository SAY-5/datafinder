"""Pydantic types shared across the agent, tools, store, and API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---- domain ---------------------------------------------------------

Modality = Literal[
    "MRI", "CT", "X-ray", "ultrasound", "PET", "histology",
    "fundus", "OCT", "EEG", "ECG",
]


class Dataset(BaseModel):
    id: str
    title: str
    description: str
    modality: Modality | None = None
    anatomy: str | None = None
    subjects: int = 0
    age_min: int | None = None
    age_max: int | None = None
    annotations: list[str] = Field(default_factory=list)
    citation: str | None = None
    columns: list[str] = Field(default_factory=list)


class DatasetHit(BaseModel):
    dataset_id: str
    title: str
    score: float
    why: str  # short, why this matched (semantic distance, filter spec, …)


class DatasetDetail(BaseModel):
    dataset: Dataset
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)


# ---- agent runtime --------------------------------------------------

Route = Literal["semantic", "metadata", "hybrid", "preview_only"]


class NormalizedQuery(BaseModel):
    raw: str
    text: str
    hints: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    idx: int
    tool: str
    args: dict[str, Any]
    result_summary: str
    elapsed_ms: int = 0


class AgentRun(BaseModel):
    id: str
    session_id: str | None = None
    query: str
    normalized: NormalizedQuery
    route: Route
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    grounded: bool = False
    refinements: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime | None = None
