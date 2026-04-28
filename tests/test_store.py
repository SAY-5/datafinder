from __future__ import annotations

from datafinder.embed import DeterministicEmbedder, cosine
from datafinder.schema import Dataset
from datafinder.store import MemoryStore


def _seed_minimal() -> MemoryStore:
    s = MemoryStore()
    s.upsert(Dataset(
        id="a", title="Knee MRI", description="MRI of the knee with cartilage labels",
        modality="MRI", anatomy="knee", subjects=500, age_min=40, age_max=70,
        annotations=["cartilage", "WOMAC"],
    ))
    s.upsert(Dataset(
        id="b", title="Lung CT", description="Low-dose CT for lung nodules",
        modality="CT", anatomy="lung", subjects=10000, age_min=55, age_max=75,
        annotations=["nodules"],
    ))
    s.upsert(Dataset(
        id="c", title="Brain MRI", description="Structural brain MRI in dementia",
        modality="MRI", anatomy="brain", subjects=2000, age_min=60, age_max=90,
        annotations=["hippocampus_volume"],
    ))
    return s


def test_metadata_filter_modality():
    s = _seed_minimal()
    ids = {h.dataset_id for h in s.metadata_filter(modality="MRI")}
    assert ids == {"a", "c"}


def test_metadata_filter_anatomy_substring():
    s = _seed_minimal()
    assert {h.dataset_id for h in s.metadata_filter(anatomy="knee")} == {"a"}


def test_metadata_filter_min_subjects():
    s = _seed_minimal()
    ids = {h.dataset_id for h in s.metadata_filter(min_subjects=1000)}
    assert ids == {"b", "c"}


def test_metadata_filter_age_min_lower_bound():
    """Dataset must include subjects at-or-below the requested floor."""
    s = _seed_minimal()
    ids = {h.dataset_id for h in s.metadata_filter(age_min=60)}
    # a covers 40-70 → its lower bound is 40 < 60 → REJECT
    # b covers 55-75 → 55 < 60 → REJECT
    # c covers 60-90 → 60 ≥ 60 → keep
    assert ids == {"c"}


def test_metadata_filter_annotations_subset():
    s = _seed_minimal()
    ids = {h.dataset_id for h in s.metadata_filter(annotations=["cartilage"])}
    assert ids == {"a"}
    # Multi-annotation: must contain both.
    ids = {h.dataset_id for h in s.metadata_filter(annotations=["cartilage", "WOMAC"])}
    assert ids == {"a"}
    # One missing → no result.
    assert s.metadata_filter(annotations=["cartilage", "nodules"]) == []


def test_semantic_search_ranks_relevant_dataset_first():
    s = _seed_minimal()
    hits = s.semantic_search("knee MRI cartilage", k=3)
    assert hits[0].dataset_id == "a"


def test_semantic_search_returns_at_most_k():
    s = _seed_minimal()
    assert len(s.semantic_search("MRI", k=2)) == 2


def test_detail_returns_sample_rows():
    s = _seed_minimal()
    d = s.detail("a")
    assert d is not None
    assert d.dataset.id == "a"
    assert len(d.sample_rows) > 0


def test_detail_unknown_returns_none():
    s = _seed_minimal()
    assert s.detail("does-not-exist") is None


def test_embedder_determinism_and_cosine():
    e = DeterministicEmbedder(dim=128)
    a, b = e.embed(["knee MRI", "knee MRI"])
    assert a == b
    c, d = e.embed(["knee MRI", "lung CT"])
    # Same-input cosine == 1.0 (within floating-point); different
    # inputs should be < 1.0 and finite.
    assert abs(cosine(a, b) - 1.0) < 1e-9
    assert -1.0 <= cosine(c, d) < 1.0
