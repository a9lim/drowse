"""The probe roster: namespace routing, digest caches, one invalidation point."""

from __future__ import annotations

from typing import Any

import pytest

from drowse.readout.roster import ProbeRoster


class _Family:
    """A read family fake recording attaches, detaches and digest calls."""

    def __init__(self, family: str, *, owns: tuple[str, ...] = ()) -> None:
        self.family = family
        self.attached: list[tuple[str, dict[str, Any]]] = []
        self.owned = list(owns)
        self.hash_calls: list[str] = []
        self.detached: list[str] = []
        self.swaps: list[Any] = []

    @property
    def names(self) -> list[str]:
        return list(self.owned)

    def specs(self) -> dict[str, dict[str, Any]]:
        return {name: {"family": self.family} for name in self.owned}

    def manifolds(self) -> dict[str, Any]:
        return {name: f"manifold:{name}" for name in self.owned}

    def attach(self, selector: str, **kwargs: Any) -> str:
        self.attached.append((selector, kwargs))
        name = kwargs.get("as_name") or selector
        self.owned.append(name)
        return name

    def try_detach(self, name: str) -> bool:
        if name not in self.owned:
            return False
        self.owned.remove(name)
        self.detached.append(name)
        return True

    def detach(self, name: str) -> None:
        self.owned.remove(name)
        self.detached.append(name)

    def probe_hash(self, name: str) -> str | None:
        self.hash_calls.append(name)
        return f"{self.family}:{name}" if name in self.owned else None

    def refresh_swapped(self, swaps: Any) -> list[str]:
        self.swaps.append(swaps)
        return list(self.owned)


def _roster(**owns: tuple[str, ...]) -> tuple[ProbeRoster, dict[str, _Family], list[int]]:
    families = {
        family: _Family(family, owns=owns.get(family, ()))
        for family in ("geometry", "lens", "sae")
    }
    changes: list[int] = []
    roster = ProbeRoster(
        geometry=families["geometry"],
        lens=families["lens"],
        sae=families["sae"],
        on_change=lambda: changes.append(1),
    )
    return roster, families, changes


@pytest.mark.parametrize(("selector", "family", "kwargs"), [
    ("jlens/word", "lens", {"as_name": None}),
    ("sae/12", "sae", {"as_name": None}),
    ("mood", "geometry", {"as_name": None, "top_n": 5}),
    ("local/mood:raw", "geometry", {"as_name": None, "top_n": 5}),
])
def test_attach_routes_by_namespace(selector: str, family: str, kwargs: dict[str, Any]) -> None:
    roster, families, changes = _roster()
    assert roster.attach(selector, top_n=5) == selector
    assert families[family].attached == [(selector, kwargs)]
    assert all(not f.attached for name, f in families.items() if name != family)
    assert changes == [1]


def test_detach_asks_the_readout_families_before_geometry() -> None:
    roster, families, changes = _roster(geometry=("g",), lens=("jlens/x",), sae=("sae/1",))
    roster.detach("jlens/x")
    roster.detach("sae/1")
    roster.detach("g")
    assert families["lens"].detached == ["jlens/x"]
    assert families["sae"].detached == ["sae/1"]
    assert families["geometry"].detached == ["g"]
    assert changes == [1, 1, 1]


def test_digests_are_cached_per_family_until_forgotten() -> None:
    roster, families, changes = _roster(geometry=("g",), lens=("jlens/x",))
    assert roster.probe_hash("jlens/x") == "lens:jlens/x"
    assert roster.probe_hash("g") == "geometry:g"
    assert roster.probe_hash("ghost") is None
    calls = {name: list(f.hash_calls) for name, f in families.items()}
    assert roster.probe_hash("jlens/x") == "lens:jlens/x"
    assert roster.probe_hash("g") == "geometry:g"
    assert {name: f.hash_calls for name, f in families.items()} == calls

    roster.forget(["g"])
    assert roster.probe_hash("g") == "geometry:g"
    assert families["geometry"].hash_calls.count("g") == 2
    assert changes == []

    roster.changed(["jlens/x"])
    assert changes == [1]
    roster.probe_hash("jlens/x")
    assert families["lens"].hash_calls.count("jlens/x") == 2


def test_probe_hashes_walk_every_family() -> None:
    roster, _families, _changes = _roster(
        geometry=("g",), lens=("jlens/x",), sae=("sae/1",),
    )
    assert roster.probe_hashes() == {
        "g": "geometry:g", "jlens/x": "lens:jlens/x", "sae/1": "sae:sae/1",
    }


def test_a_registry_swap_refreshes_geometry_and_drops_its_digests() -> None:
    roster, families, changes = _roster(geometry=("g",))
    roster.probe_hash("g")
    swaps = {1: None}
    roster.swap(swaps)
    assert families["geometry"].swaps == [swaps]
    roster.probe_hash("g")
    assert families["geometry"].hash_calls.count("g") == 2
    assert changes == []


def test_specs_merge_the_three_families() -> None:
    roster, _families, _changes = _roster(geometry=("g",), lens=("jlens/x",), sae=("sae/1",))
    assert roster.specs() == {
        "g": {"manifold": "manifold:g"},
        "jlens/x": {"lens": {"family": "lens"}},
        "sae/1": {"sae": {"family": "sae"}},
    }


def test_clear_drops_every_digest() -> None:
    roster, families, _changes = _roster(geometry=("g",), sae=("sae/1",))
    roster.probe_hashes()
    roster.clear()
    roster.probe_hashes()
    assert families["geometry"].hash_calls.count("g") == 2
    assert families["sae"].hash_calls.count("sae/1") == 2


def test_a_session_remove_probe_drops_the_digest_but_keeps_the_prefix(
    make_tiny_session: Any,
) -> None:
    from tests.support.sessions import register_vectors

    session, _script = make_tiny_session.scripted("ok")
    register_vectors(session, "mood")
    session.add_probe("mood")
    first = session.probe_hashes()["mood"]
    session.cache_prefix([{"role": "user", "content": "hello there, a shared prefix"}])
    entry = session._engine.prefix_cache.entry
    assert entry is not None
    session.remove_probe("mood")
    assert session._engine.prefix_cache.entry is entry
    assert "mood" not in session.probe_hashes()
    session.add_probe("mood")
    assert session.probe_hashes()["mood"] == first
