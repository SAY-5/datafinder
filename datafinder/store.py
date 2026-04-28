"""Store protocol + an in-memory implementation.

The protocol is what the tools call; production wires PgvectorStore,
tests + CI use MemoryStore. Both implement the same surface so a
flag flip moves the service onto Postgres without touching tools/
agent code.
"""

from __future__ import annotations

import threading
from typing import Protocol

from datafinder.embed import DeterministicEmbedder, Embedder, cosine
from datafinder.schema import Dataset, DatasetDetail, DatasetHit, Modality


class Store(Protocol):
    embedder: Embedder

    def upsert(self, ds: Dataset) -> None: ...

    def get(self, dataset_id: str) -> Dataset | None: ...

    def detail(self, dataset_id: str) -> DatasetDetail | None: ...

    def semantic_search(self, query: str, k: int = 5) -> list[DatasetHit]: ...

    def metadata_filter(
        self,
        modality: Modality | None = None,
        anatomy: str | None = None,
        min_subjects: int | None = None,
        age_min: int | None = None,
        annotations: list[str] | None = None,
    ) -> list[DatasetHit]: ...

    def all_ids(self) -> list[str]: ...


class MemoryStore:
    """Reference in-memory backend. Thread-safe for the read paths
    we exercise; writes are guarded by a coarse lock."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or DeterministicEmbedder()
        self._mu = threading.RLock()
        self._datasets: dict[str, Dataset] = {}
        self._vectors: dict[str, list[float]] = {}

    # ----- writes ----------------------------------------------------

    def upsert(self, ds: Dataset) -> None:
        with self._mu:
            self._datasets[ds.id] = ds
            text = " ".join(filter(None, [
                ds.title, ds.description,
                ds.modality or "", ds.anatomy or "",
                " ".join(ds.annotations),
            ]))
            self._vectors[ds.id] = self.embedder.embed([text])[0]

    # ----- reads -----------------------------------------------------

    def get(self, dataset_id: str) -> Dataset | None:
        with self._mu:
            return self._datasets.get(dataset_id)

    def detail(self, dataset_id: str) -> DatasetDetail | None:
        ds = self.get(dataset_id)
        if ds is None:
            return None
        # Synthetic preview rows — production fetches first 5 rows
        # from S3 / lab storage. The shape is what citations use.
        sample = []
        for i in range(min(3, ds.subjects)):
            row = {col: f"<{col}_{i}>" for col in (ds.columns or ["subject_id", "value"])}
            sample.append(row)
        return DatasetDetail(dataset=ds, sample_rows=sample)

    def semantic_search(self, query: str, k: int = 5) -> list[DatasetHit]:
        if k <= 0:
            return []
        q = self.embedder.embed([query])[0]
        with self._mu:
            scored = [
                (self._datasets[did], cosine(q, vec))
                for did, vec in self._vectors.items()
            ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            DatasetHit(
                dataset_id=ds.id, title=ds.title, score=round(score, 4),
                why=f"semantic similarity {score:.3f}",
            )
            for ds, score in scored[:k]
        ]

    def metadata_filter(
        self,
        modality: Modality | None = None,
        anatomy: str | None = None,
        min_subjects: int | None = None,
        age_min: int | None = None,
        annotations: list[str] | None = None,
    ) -> list[DatasetHit]:
        with self._mu:
            results: list[DatasetHit] = []
            for ds in self._datasets.values():
                if modality and ds.modality != modality:
                    continue
                if anatomy and (not ds.anatomy or anatomy.lower() not in ds.anatomy.lower()):
                    continue
                if min_subjects is not None and ds.subjects < min_subjects:
                    continue
                if age_min is not None:
                    if ds.age_min is None or ds.age_min < age_min:
                        # Dataset must include subjects in the requested age band:
                        # ds covers [age_min..age_max]; reject if its lower bound
                        # is below what the user asked.
                        continue
                if annotations:
                    have = {a.lower() for a in ds.annotations}
                    want = {a.lower() for a in annotations}
                    if not want.issubset(have):
                        continue
                results.append(DatasetHit(
                    dataset_id=ds.id, title=ds.title, score=1.0,
                    why=_filter_why(modality, anatomy, min_subjects, age_min, annotations),
                ))
        return results

    def all_ids(self) -> list[str]:
        with self._mu:
            return list(self._datasets)


def _filter_why(
    modality: str | None,
    anatomy: str | None,
    min_subjects: int | None,
    age_min: int | None,
    annotations: list[str] | None,
) -> str:
    bits = []
    if modality:
        bits.append(f"modality={modality}")
    if anatomy:
        bits.append(f"anatomy~={anatomy}")
    if min_subjects:
        bits.append(f"≥{min_subjects} subjects")
    if age_min:
        bits.append(f"age≥{age_min}")
    if annotations:
        bits.append(f"annot⊇{set(annotations)}")
    return "metadata " + (", ".join(bits) if bits else "(no filter)")
