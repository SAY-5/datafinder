# DataFinder

An autonomous research-dataset discovery agent. A researcher asks
"find me knee MRI datasets with at least 50 subjects, age 40+,
annotated for cartilage thickness" and the agent routes the query,
sequences tool calls (semantic search over embeddings, structured
metadata filtering, dataset preview), grounds its answer with source
citations, and decides on its own when to retry with a refined query
before responding.

> Originally built for a biomedical-imaging lab. This repo is a
> faithful re-implementation against synthetic datasets so the
> service runs end-to-end with no lab infrastructure required —
> the routing logic, tool sequencing, normalization, and grounding
> behavior are identical to what shipped.

## Three autonomous decisions

The interesting work is in the decisions the agent makes between
the question coming in and the answer going out. Each is observable
in the run log and tested explicitly.

1. **Query routing** — `semantic` / `metadata` / `hybrid` /
   `preview_only`. Rule-based classifier on the normalized query;
   a few-shot LLM classifier was the v0 design but the rule-based
   version is ~30 lines, runs in microseconds, and is easier to
   audit.
2. **Tool sequencing** — given the route, decide which tools to
   call and in what order. The agent has access to
   `semantic_search`, `metadata_filter`, and `dataset_preview`. The
   hybrid path typically calls metadata first to narrow, then
   semantic to rank within the narrowed set, then preview on the
   top one or two for citation detail.
3. **Answer grounding** — after the tool calls, the agent decides
   whether the retrieved context was enough. If the model's answer
   doesn't reference any dataset id we saw in tool results, we
   refine the system message and retry, bounded by `max_refinements`.

## Stack

- FastAPI + Pydantic
- OpenAI Chat Completions with function calling (`tools=[…]`)
- Production: PostgreSQL + pgvector for similarity over
  `text-embedding-3-small`. Schema sketch in `ARCHITECTURE.md`.
- CI / local dev: in-memory `MemoryStore` + a deterministic
  hash-based embedder. Same `Store` and `Embedder` protocols as
  the production backends; flag flip swaps them.

## Quick start

```bash
pip install -e ".[dev,openai]"
uvicorn datafinder.api:app --reload --port 8000
# UI: serve ./web with any static server, e.g.
python -m http.server 5173 --directory web
# Or with the full stack (Postgres + pgvector + datafinder):
docker compose up -d --build
```

Without `OPENAI_API_KEY` the API uses a deterministic stub chat
client — good for demos and CI. With the key set, it wires the real
OpenAI client via `datafinder.openai_client`.

```bash
curl -s -X POST localhost:8000/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"query": "knee MRI cohort with cartilage thickness, ≥50 subjects, age 40+"}' | jq
```

## Tests (38 green)

```bash
pytest -q
```

- **normalize** (10): abbreviation expansion (with deduping), age-
  band extraction, subject-count extraction, "100+ patients" not
  misread as age, OAI-2K not misread as age.
- **router** (6): preview_only on `ds_*` references, pure metadata
  on spec queries, pure semantic on conceptual queries, hybrid on
  mixed, default-hybrid on ambiguous, hint-presence as metadata
  signal.
- **store** (10): metadata filters (modality, anatomy substring,
  min subjects, age lower-bound, annotation subset),
  semantic_search ranking + k-cap, detail + 404, embedder
  determinism + cosine bounds.
- **agent** (7): single-tool round-trip with grounding, hybrid two-
  round dispatch, ungrounded answer triggers refinement, refinement
  budget capped, preview_only routing recorded, every tool call
  audited, tool-round budget capped.
- **api** (5): healthz, ask returns grounded run, get_run round-
  trip, list + detail datasets, 404.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the design writeup
(routing, tool dispatch, query normalization, grounding loop,
pgvector schema, the production swap).

## Companion projects

Part of the SAY-5 portfolio under [github.com/SAY-5](https://github.com/SAY-5).

## License

MIT.
