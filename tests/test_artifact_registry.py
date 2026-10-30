"""``ArtifactRegistry``: read-only views, invalidating mutations, adopt/evict
key matching, and the analytics snapshot cache."""
from __future__ import annotations

import threading
from typing import Any

import pytest
import torch

from drowse.session.registry import (
    ArtifactRegistry,
    ModelBinding,
    artifact_key_matches,
)


def _registry() -> ArtifactRegistry:
    return ArtifactRegistry(ModelBinding(
        model=None, tokenizer=None, model_id="stub/model",
        device=torch.device("cpu"), dtype=torch.float32,
    ))


class _Recorder:
    def __init__(self, registry: ArtifactRegistry) -> None:
        self.changes = 0
        self.swaps: list[tuple[set[int], Any]] = []
        registry.on_change(self._changed)
        registry.on_swap(lambda ids, replacement: self.swaps.append((ids, replacement)))

    def _changed(self) -> None:
        self.changes += 1


def test_views_are_read_only() -> None:
    registry = _registry()
    registry.put_profile("a", {0: torch.ones(2)})
    with pytest.raises(TypeError):
        registry.profiles["b"] = {0: torch.ones(2)}  # type: ignore[index]
    with pytest.raises(TypeError):
        registry.manifolds["m"] = object()  # type: ignore[index]
    assert set(registry.profiles) == {"a"}


def test_every_mutation_fires_the_change_listeners() -> None:
    registry = _registry()
    seen = _Recorder(registry)
    registry.put_profile("jlens/a", {0: torch.ones(2)})
    registry.put_profile("jlens/b", {0: torch.ones(2)})
    registry.put_profile("x", {0: torch.ones(2)})
    assert seen.changes == 3
    assert registry.evict_prefix("jlens/") == ["jlens/a", "jlens/b"]
    assert set(registry.profiles) == {"x"}
    assert registry.pop_profile("x") is not None
    registry.clear()
    assert seen.changes == 6


def test_lazy_loads_do_not_fire_the_change_listeners(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolution-time fold memoizes what is on disk; it must not drop the
    prefix cache the way a user registration does."""
    registry = _registry()
    seen = _Recorder(registry)
    monkeypatch.setattr(registry, "fold_manifold", lambda name, role="vector": {0: torch.ones(2)})
    registry.ensure_profile_registered("default/concept")
    assert "default/concept" in registry.profiles
    assert seen.changes == 0


@pytest.mark.parametrize(
    ("key", "feature", "expected"),
    [
        ("mood", None, True),
        ("local/mood", None, True),
        ("local/mood:from-src", None, True),
        ("local/mood", "raw", True),
        ("local/mood:role-pirate", "raw", True),
        ("local/mood:sae-rel", "raw", False),
        ("local/mood:sae-rel", "sae-rel", True),
        ("local/mood:from-src", "raw", False),
        ("other/mood", None, False),
        ("local/moody", None, False),
    ],
)
def test_artifact_key_matching(key: str, feature: str | None, expected: bool) -> None:
    assert artifact_key_matches(key, "local", "mood", feature) is expected


def test_evict_artifact_drops_every_consumer_and_reports_the_swap() -> None:
    registry = _registry()
    seen = _Recorder(registry)
    raw, sae, other = object(), object(), object()
    registry.put_manifold("local/mood", raw)  # type: ignore[arg-type]
    registry.put_manifold("mood:sae-rel", sae)  # type: ignore[arg-type]
    registry.put_manifold("other/mood", other)  # type: ignore[arg-type]
    registry.put_profile("local/mood", {0: torch.ones(2)})
    registry.put_profile("other/mood", {0: torch.ones(2)})

    registry.evict_artifact("local", "mood", feature="raw")
    assert set(registry.manifolds) == {"mood:sae-rel", "other/mood"}
    assert set(registry.profiles) == {"other/mood"}
    assert seen.swaps == [({id(raw)}, None)]

    registry.evict_artifact("local", "mood")
    assert set(registry.manifolds) == {"other/mood"}
    assert seen.swaps[-1] == ({id(sae)}, None)


def test_analytics_snapshot_is_cached_until_a_mutation() -> None:
    registry = _registry()
    builds: list[int] = []

    def live() -> dict[int, torch.Tensor]:
        builds.append(1)
        return {0: torch.ones(2)}

    lock = threading.Lock()
    first = registry.analytics_snapshot("a", live=live, lock=lock)
    assert registry.analytics_snapshot("a", live=live, lock=lock) is first
    assert len(builds) == 1
    registry.put_profile("b", {0: torch.ones(2)})
    assert registry.analytics_snapshot("a", live=live, lock=lock) is not first
    assert len(builds) == 2


def test_analytics_snapshot_defers_while_the_gpu_is_busy() -> None:
    registry = _registry()
    lock = threading.Lock()
    lock.acquire()
    try:
        assert registry.analytics_snapshot(
            "a", live=lambda: {0: torch.ones(2)}, lock=lock,
        ) is None
    finally:
        lock.release()


def test_snapshot_built_across_a_mutation_is_not_cached() -> None:
    """C5: a mutation that lands while a snapshot is being built must not
    leave the stale snapshot cached (it used to be stored after the lock was
    released, past the mutation's invalidation)."""
    registry = _registry()
    registry.put_profile("a", {0: torch.ones(2)})
    builds: list[torch.Tensor] = []

    def live() -> dict[int, torch.Tensor]:
        tensors = dict(registry.profiles["a"])
        builds.append(tensors[0])
        if len(builds) == 1:
            # A concurrent re-registration while the first build runs.
            registry.put_profile("a", {0: 2.0 * torch.ones(2)})
        return tensors

    lock = threading.Lock()
    stale = registry.analytics_snapshot("a", live=live, lock=lock)
    assert stale is not None and torch.equal(stale[0], torch.ones(2))
    fresh = registry.analytics_snapshot("a", live=live, lock=lock)
    assert fresh is not None and torch.equal(fresh[0], 2.0 * torch.ones(2))
    assert len(builds) == 2
