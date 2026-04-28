from __future__ import annotations

from datafinder.normalize import normalize
from datafinder.router import route


def _r(q: str) -> str:
    return route(normalize(q))


def test_dataset_id_reference_routes_to_preview():
    assert _r("show me ds_oai") == "preview_only"
    assert _r("preview ds-001") == "preview_only"


def test_pure_metadata_query_routes_to_metadata():
    # Spec-shaped: hard constraints, no conceptual signal.
    assert _r("MRI knee, ≥100 subjects, age 60+") == "metadata"


def test_pure_semantic_query_routes_to_semantic():
    assert _r("what's similar to the OAI cohort?") in {"semantic", "hybrid"}
    # A short conceptual phrase with no constraints.
    r = _r("emphysema markers in lung CT")
    # has lung+CT keywords AND a conceptual phrase ('emphysema markers')
    assert r in {"semantic", "hybrid"}


def test_mixed_query_routes_to_hybrid():
    # Constraints + conceptual phrase together.
    assert _r("studies similar to the OAI cohort with at least 200 subjects") == "hybrid"


def test_default_routing_is_hybrid_for_ambiguous():
    # No specific cues — default policy says hybrid.
    assert _r("xenobiology") in {"hybrid", "semantic"}


def test_route_uses_extracted_hints_as_metadata_signal():
    # Ages alone should be enough to flag metadata signal.
    nq = normalize("subjects aged 50 to 70")
    assert nq.hints
    r = route(nq)
    assert r in {"metadata", "hybrid"}
