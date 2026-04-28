# DataFinder — Architecture

> An autonomous research-dataset discovery agent. A researcher asks
> "find me knee MRI datasets with at least 50 subjects, age 40+,
> annotated for cartilage thickness" and the agent routes the query,
> sequences tool calls (semantic search over embeddings, structured
> metadata filtering, dataset previews), grounds its answer with
> source citations, and decides on its own when to retry with a
> refined query before responding. Built originally for a biomedical
> imaging lab; the tool surface here is a faithful re-implementation
> with synthetic datasets so the repo runs end-to-end without lab
> infrastructure.

## Three autonomous decisions

The agent's value is in the decisions it makes between input and
response. Each is observable in the run log and tested explicitly.

1. **Query routing** — classify whether a question needs:
   - `semantic` (similarity over `description` embeddings — good for
     vague conceptual queries like "lung CT with emphysema markers"),
   - `metadata` (structured filters — good for precise spec-shaped
     queries like "subjects ≥ 100, modality = MRI, anatomy = knee"),
   - `hybrid` (both, then merge — the common case for real research
     questions).
2. **Tool sequencing** — given the route, decide which tools to call
   and in what order. The agent has access to `semantic_search`,
   `metadata_filter`, and `dataset_preview`. For a hybrid query it
   typically calls `metadata_filter` first to narrow the candidate
   set, then `semantic_search` to rank within it, then
   `dataset_preview` on the top 1-2 to surface schema details for
   the citation.
3. **Answer grounding** — after the tool calls, the agent decides
   whether the retrieved context is enough to answer. If not, it
   normalizes the query (strip filler, expand abbreviations, add
   inferred constraints from chat history) and retries — bounded by
   a `max_refinements` budget so it can't loop forever.

## Stack

| Piece            | Tech |
|------------------|------|
| Agent runtime    | OpenAI Chat Completions with function calling (`tools=[…]`) |
| Embeddings       | OpenAI text-embedding-3-small (1536-d) — swappable via the `Embedder` protocol |
| Vector store     | PostgreSQL + pgvector extension (`<->` cosine distance, IVFFlat index) |
| Metadata store   | Same Postgres database; structured columns + JSONB for free-form facets |
| HTTP API         | FastAPI |
| Frontend         | React + TypeScript (chat UI with the tool-call trace visible) |
| Session state    | Postgres-backed; one `agent_runs` row per query, one `tool_calls` per call |

For local dev and CI we ship an **in-memory backend** (`store.MemoryStore`)
that satisfies the same `Store` protocol the pgvector backend does.
Tests run against MemoryStore; production wires `PgvectorStore`.

## Directory layout

```
datafinder/
├── datafinder/
│   ├── schema.py           # Pydantic models: Dataset, ToolCall, AgentRun
│   ├── store.py            # Store protocol + MemoryStore
│   ├── pgvector_store.py   # Production pg backend (sketched)
│   ├── embed.py            # Embedder protocol + DeterministicEmbedder
│   ├── tools.py            # The three tools wired against a Store
│   ├── normalize.py        # Query normalization layer
│   ├── router.py           # Query-route classifier
│   ├── agent.py            # Function-calling agent loop
│   ├── api.py              # FastAPI app
│   └── seed.py             # Synthetic biomedical-imaging datasets
├── tests/
└── web/                    # React UI
```

## The three tools

Each tool is a Python function with a JSON schema the agent receives.

```python
@tool(schema={...})
def semantic_search(query: str, k: int = 5) -> list[Hit]:
    """Find datasets whose description is semantically close to `query`.
    Returns Hits sorted by cosine distance ascending."""

@tool(schema={...})
def metadata_filter(modality: str | None = None,
                    anatomy: str | None = None,
                    min_subjects: int | None = None,
                    age_min: int | None = None,
                    annotations: list[str] | None = None) -> list[Hit]:
    """Filter datasets by structured fields. Returns ALL matches."""

@tool(schema={...})
def dataset_preview(dataset_id: str) -> DatasetDetail:
    """Fetch full metadata + first-5-rows schema for a single dataset.
    The agent calls this on its top candidates so the answer can cite
    specific column names and value ranges."""
```

## Query normalization

`normalize.py` runs before the agent sees the question. It:

- Lowercases, collapses whitespace, strips trailing punctuation.
- Expands a small bio-imaging abbreviation table (`MRI` → "magnetic
  resonance imaging", `CT` → "computed tomography", `OAI` →
  "Osteoarthritis Initiative", etc.) — duplicates not added if the
  long form is already present.
- Detects implicit numeric constraints in natural-language form
  ("at least 100 subjects", "older than 40", "between 20 and 60")
  and emits structured hints the agent can pass directly to
  `metadata_filter` without inferring them itself.

The output is a `NormalizedQuery(text, hints)` envelope the agent
sees as its system message context. Every run records the
normalization pass in the audit log so you can see exactly what the
agent was actually asked.

## Agent loop

```
NormalizedQuery → system+user messages
  ↓
OpenAI call with tools=[semantic_search, metadata_filter, dataset_preview]
  ↓
For each tool call returned:
   execute → append result message → call OpenAI again
  ↓
Agent emits a finish_reason="stop" with content
  ↓
Grounding check: does the answer cite at least one tool-result?
   yes → return
   no  → retry with refined system message (+1 to refinement count)
   refinement_count > max_refinements → return as-is, mark
     `grounded=false` in the run record
```

## Session state

Two Postgres tables (in production; MemoryStore mirrors the shape):

```
agent_runs (id, session_id, query, normalized_query, route,
            answer, grounded, refinements, started_at, ended_at)
tool_calls (id, run_id, idx, tool, args_json, result_summary,
            elapsed_ms)
```

`session_id` lets the UI show conversation context; the agent passes
recent runs into its system message so follow-ups ("can you also
check ages 60+") have continuity.

## Embeddings

`Embedder` is a protocol with one method: `embed(texts) -> list[list[float]]`.

Two implementations:

- `OpenAIEmbedder` (production) — wraps the OpenAI client. Batches
  inputs to 100, retries on 429.
- `DeterministicEmbedder` (tests) — hashes each text into a fixed-
  dimension float vector via stable PRNG. Returns identical vectors
  for identical inputs, cosine-similar for inputs sharing tokens.
  No network, no nondeterminism. Used by every test.

## pgvector index

Production-only; the schema sketch:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE datasets (
    id           text PRIMARY KEY,
    title        text NOT NULL,
    description  text NOT NULL,
    modality     text,
    anatomy      text,
    subjects     int,
    age_min      int,
    age_max      int,
    annotations  jsonb,
    embedding    vector(1536)
);
CREATE INDEX ON datasets USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX ON datasets (modality, anatomy);
CREATE INDEX ON datasets USING gin (annotations);
```

## Performance & non-goals

- Latency: a typical run is 3-5 OpenAI calls (1 routing + 1-2 tool
  rounds + 1 answer); end-to-end p50 ≈ 2-3 s when OpenAI is healthy.
- Concurrency: the agent loop is async; FastAPI handles fan-out;
  Postgres pool is shared.
- **Out of scope**: training the embedding model (we use OpenAI's),
  cross-institution federation (the lab dataset was single-site),
  fine-tuning the routing classifier (a system prompt + few-shots
  is more than enough for ~17 question archetypes).
