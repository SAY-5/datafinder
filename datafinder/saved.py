"""v4: saved searches with pinned filters.

Users repeat the same search ('rows where region=us-west and date
> last week') across sessions. v4 adds a SavedSearch registry: a
named query + filter set the user can recall by name. The
registry is keyed on user_id so multi-tenant deployments don't
leak across users.

We persist via a pluggable Store; the in-memory default is for
tests + small deployments. Production swaps in a Postgres-backed
adapter that satisfies the same contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SavedSearch:
    user_id: str
    name: str
    query: str
    filters: tuple[tuple[str, str], ...] = ()  # ordered (key, value) pairs


@dataclass
class SavedSearchStore:
    """In-memory keyed store. (user_id, name) → SavedSearch."""

    _items: dict[tuple[str, str], SavedSearch] = field(default_factory=dict)

    def save(self, s: SavedSearch) -> None:
        self._items[(s.user_id, s.name)] = s

    def get(self, user_id: str, name: str) -> SavedSearch | None:
        return self._items.get((user_id, name))

    def list_for(self, user_id: str) -> list[SavedSearch]:
        return sorted(
            (s for (uid, _), s in self._items.items() if uid == user_id),
            key=lambda s: s.name,
        )

    def delete(self, user_id: str, name: str) -> bool:
        return self._items.pop((user_id, name), None) is not None


def render_filter_url(s: SavedSearch) -> str:
    """Build a query string the API can re-issue. Filters are
    serialized in declaration order so the URL is stable across
    invocations (no random dict ordering)."""

    parts = [f"q={s.query}"]
    parts.extend(f"{k}={v}" for k, v in s.filters)
    return "?" + "&".join(parts)


def parse_filters(items: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Normalize a filter iterable into the canonical tuple form.
    Production CLI / API layers consume user-typed filter strings
    and pass them through here."""
    return tuple((k, v) for k, v in items)
