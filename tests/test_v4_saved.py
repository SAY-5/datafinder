from __future__ import annotations

from datafinder.saved import (
    SavedSearch,
    SavedSearchStore,
    parse_filters,
    render_filter_url,
)


def _s(user: str, name: str, query: str = "x", filters=()) -> SavedSearch:
    return SavedSearch(user_id=user, name=name, query=query, filters=filters)


def test_save_then_get_round_trips() -> None:
    store = SavedSearchStore()
    s = _s("alice", "north-america", filters=(("region", "us-west"),))
    store.save(s)
    got = store.get("alice", "north-america")
    assert got == s


def test_list_for_filters_by_user() -> None:
    store = SavedSearchStore()
    store.save(_s("alice", "a"))
    store.save(_s("alice", "b"))
    store.save(_s("bob", "c"))
    assert [s.name for s in store.list_for("alice")] == ["a", "b"]
    assert [s.name for s in store.list_for("bob")] == ["c"]


def test_delete_removes_and_returns_true() -> None:
    store = SavedSearchStore()
    store.save(_s("alice", "x"))
    assert store.delete("alice", "x")
    assert store.get("alice", "x") is None
    assert not store.delete("alice", "x")  # second delete returns false


def test_render_filter_url_preserves_order() -> None:
    s = _s("alice", "n", query="users",
           filters=(("region", "us-west"), ("active", "true")))
    url = render_filter_url(s)
    assert url == "?q=users&region=us-west&active=true"


def test_parse_filters_returns_tuple() -> None:
    out = parse_filters([("a", "1"), ("b", "2")])
    assert out == (("a", "1"), ("b", "2"))


def test_save_overwrites_existing_with_same_name() -> None:
    store = SavedSearchStore()
    store.save(_s("alice", "n", query="old"))
    store.save(_s("alice", "n", query="new"))
    got = store.get("alice", "n")
    assert got is not None
    assert got.query == "new"
