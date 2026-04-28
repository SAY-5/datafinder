from __future__ import annotations

from fastapi.testclient import TestClient

from datafinder.agent import AgentConfig, FakeChatClient, fake_final, fake_tool_call
from datafinder.api import build_app
from datafinder.store import MemoryStore


def _client(responses):
    store = MemoryStore()
    chat = FakeChatClient(responses=responses)
    app = build_app(store=store, chat=chat, config=AgentConfig(max_refinements=0))
    return TestClient(app), chat


def test_healthz():
    c, _ = _client([])
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["datasets"] >= 5


def test_ask_returns_grounded_run():
    c, _ = _client([
        fake_tool_call("semantic_search", {"query": "knee", "k": 3}),
        fake_final("ds_oai — best fit."),
    ])
    r = c.post("/v1/ask", json={"query": "knee MRI cohort"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["answer"].startswith("ds_oai")
    assert body["route"] in {"semantic", "metadata", "hybrid"}
    assert body["tool_calls"]


def test_get_run_after_ask():
    c, _ = _client([
        fake_tool_call("semantic_search", {"query": "x"}),
        fake_final("ds_adni"),
    ])
    body = c.post("/v1/ask", json={"query": "brain MRI"}).json()
    rid = body["id"]
    detail = c.get(f"/v1/runs/{rid}")
    assert detail.status_code == 200
    assert detail.json()["id"] == rid


def test_list_and_detail_datasets():
    c, _ = _client([])
    items = c.get("/v1/datasets").json()["items"]
    assert any(d["id"] == "ds_oai" for d in items)
    detail = c.get("/v1/datasets/ds_oai")
    assert detail.status_code == 200
    body = detail.json()
    assert body["dataset"]["title"].startswith("OAI")
    assert body["sample_rows"]


def test_unknown_dataset_404():
    c, _ = _client([])
    assert c.get("/v1/datasets/does-not-exist").status_code == 404
