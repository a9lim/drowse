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
        self.swaps: list[dict[int, Any]] = []
        registry.on_change(self._changed)
        registry.on_swap(lambda swaps: self.swaps.append(dict(swaps)))

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
    assert seen.swaps == [{id(raw): None}]

    registry.evict_artifact("local", "mood")
    assert set(registry.manifolds) == {"other/mood"}
    assert seen.swaps[-1] == {id(sae): None}


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


class _Fold:
    """A stand-in probe fold: carries the tensors it was folded from."""

    def __init__(self, key: str, tensors: Any) -> None:
        self.key = key
        self.tensors = dict(tensors)


def _ray(direction: torch.Tensor, *, share: float = 1.0) -> Any:
    from drowse.geometry.domains import CustomDomain
    from drowse.geometry.manifold import Manifold
    from drowse.geometry.subspace import LayerSubspace

    unit = (direction / direction.norm()).reshape(1, -1)
    layer = LayerSubspace.affine(
        torch.zeros(direction.numel()), unit,
        node_coords=torch.tensor([[1.0]]),
    )
    return Manifold(
        name="mood", domain=CustomDomain(1), node_labels=["+"],
        node_coords=torch.tensor([[1.0]]), layers={0: layer},
        feature_space="raw", node_roles=[None], node_kinds=[None],
        mahalanobis_share={0: share},
    )


def test_evict_artifact_reports_tracked_probe_folds() -> None:
    """C4: a probe folded from a registered profile never held a loaded
    manifold, so eviction has to find it through the registry's fold index."""
    registry = _registry()
    registry.set_probe_folder(_Fold)  # type: ignore[arg-type]
    seen = _Recorder(registry)
    registry.put_profile("local/mood", {0: torch.ones(2)})
    fold = registry.fold_probe("local/mood", registry.profiles["local/mood"])
    alias = registry.fold_probe("mood", {0: torch.ones(2)})
    other = registry.fold_probe("other/mood", {0: torch.ones(2)})

    registry.evict_artifact("local", "mood")
    assert seen.swaps == [{id(fold): None, id(alias): None}]
    registry.evict_artifact("other", "mood")
    assert seen.swaps[-1] == {id(other): None}


def test_adopt_fitted_refolds_tracked_probe_folds() -> None:
    registry = _registry()
    registry.set_probe_folder(_Fold)  # type: ignore[arg-type]
    seen = _Recorder(registry)
    first = registry.fold_probe("local/mood", {0: torch.ones(2)})
    second = registry.fold_probe("local/mood", {0: torch.ones(2)})

    direction = torch.tensor([3.0, 4.0])
    registry.adopt_fitted("local/mood", _ray(direction, share=2.0))
    (swaps,) = seen.swaps
    refold = swaps[id(first)]
    assert swaps == {id(first): refold, id(second): refold}
    assert isinstance(refold, _Fold) and refold.key == "local/mood"
    assert torch.allclose(refold.tensors[0], 2.0 * direction / 5.0)

    # The refold is tracked in turn: the next eviction reaches it.
    registry.evict_artifact("local", "mood")
    assert seen.swaps[-1] == {id(refold): None}


def test_adopt_fitted_drops_probe_folds_when_the_fit_no_longer_folds() -> None:
    registry = _registry()
    registry.set_probe_folder(_Fold)  # type: ignore[arg-type]
    seen = _Recorder(registry)
    fold = registry.fold_probe("local/mood", {0: torch.ones(2)})
    rank2 = _ray(torch.tensor([1.0, 0.0]))
    rank2.layers[0] = type(rank2.layers[0]).affine(
        torch.zeros(2), torch.eye(2), node_coords=torch.tensor([[1.0, 0.0]]),
    )
    registry.adopt_fitted("local/mood", rank2)
    assert seen.swaps == [{id(fold): None}]


def test_detached_probe_folds_drop_out_of_the_index() -> None:
    import gc

    registry = _registry()
    registry.set_probe_folder(_Fold)  # type: ignore[arg-type]
    seen = _Recorder(registry)
    registry.fold_probe("local/mood", {0: torch.ones(2)})
    gc.collect()
    registry.evict_artifact("local", "mood")
    assert seen.swaps == []
