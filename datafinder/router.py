"""Query-route classifier.

Given a normalized query + its extracted hints, decide which
retrieval strategy the agent should use:

  semantic    : free-form / conceptual ("emphysema markers in lung CT")
  metadata    : strict spec-shaped ("MRI, knee, ≥ 100 subjects, age 40+")
  hybrid      : both required (the most common case for real research
                queries — vague concept + at least one hard constraint)
  preview_only: explicit dataset-id reference; we just go fetch it

The classifier is intentionally rule-based here. A few-shot LLM
classifier was the original v0 approach but introduced 200ms of
latency per query for a decision we can make in microseconds with
~30 lines of regex. The agent's *answer* still goes through the
LLM, of course; the routing layer is just the dispatch.
"""

from __future__ import annotations

import re

from datafinder.schema import NormalizedQuery, Route

# Heuristics — order matters for tie-breaks.

# A literal dataset id reference (e.g. "show me ds_abc123" or
# "preview ds-001") shortcuts to preview_only.
_RE_DATASET_ID = re.compile(r"\bds[_-][a-zA-Z0-9]{3,}\b")

# Phrases that strongly suggest the user wants similarity over
# free-text descriptions rather than structured filtering.
_SEMANTIC_PHRASES = (
    "similar to", "like the", "related to", "emphysema markers",
    "looks like", "describing", "about", "concept of",
)


def route(nq: NormalizedQuery) -> Route:
    text = nq.text.lower()

    if _RE_DATASET_ID.search(text):
        return "preview_only"

    has_metadata_hints = bool(nq.hints) or _has_metadata_keywords(text)
    has_semantic_signal = any(p in text for p in _SEMANTIC_PHRASES) or _looks_conceptual(text)

    if has_metadata_hints and has_semantic_signal:
        return "hybrid"
    if has_metadata_hints:
        return "metadata"
    if has_semantic_signal:
        return "semantic"
    # Default to hybrid — the agent will pick whichever tool yields
    # signal first. This is the safest default for ambiguous queries.
    return "hybrid"


def _has_metadata_keywords(text: str) -> bool:
    """Cheap detector for words that map directly to metadata cols."""
    keywords = (
        "mri", "ct ", "x-ray", "ultrasound", "pet", "histology",
        "fundus", "oct ", "eeg", "ecg",
        "knee", "lung", "brain", "heart", "spine", "retina",
        "subjects", "patients", "participants",
        "annotation", "label", "annotated",
    )
    return any(k in text for k in keywords)


def _looks_conceptual(text: str) -> bool:
    """Anything that's clearly a question rather than a spec."""
    if "?" in text:
        return True
    if any(text.startswith(w) for w in ("find", "show", "list", "give")):
        return False  # commands; favor metadata route if anything
    if any(w in text for w in ("study", "research", "investigate", "explore")):
        return True
    # Short queries (< 4 tokens) without any metadata keyword are
    # usually free-text "MRI knee" style — let semantic handle.
    return len(text.split()) <= 3 and not _has_metadata_keywords(text)
