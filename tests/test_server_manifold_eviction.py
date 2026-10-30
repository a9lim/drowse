"""C4: manifold folders changed over HTTP must not leave stale in-memory steering.

Runs the real server over a real tiny session: fit a 2-node manifold, steer
with it and attach it as a probe, then change the folder through the native
routes and check what the session still holds.
"""
from __future__ import annotations

from typing import Any

import pytest

from drowse.base.paths import manifold_dir
from drowse.errors import ManifoldNotRegisteredError, ProfileNotRegisteredError
from drowse.store.manifold_authoring import create_discover_manifold_folder
from tests.support.sessions import baseline_corpus

pytestmark = pytest.mark.usefixtures("isolated_home")


@pytest.fixture
def fitted(tiny_session: Any) -> Any:
    create_discover_manifold_folder(
        "local", "mood", "test axis", fit_mode="pca",
        node_corpora={
            "calm": baseline_corpus("I feel calm and settled today."),
            "angry": baseline_corpus("I am furious about all of this!"),
        },
        hyperparams={"max_dim": 1},
    )
    tiny_session.fit(manifold_dir("local", "mood"))
    # A push term loads the manifold, the probe attaches that loaded object,
    # and an ablation folds it into the profile registry — three in-memory
    # consumers of the folder.
    with tiny_session.steering("0.5 local/mood"):
        pass
    tiny_session.add_probe("local/mood")
    with tiny_session.steering("0.2 !local/mood"):
        pass
    assert tiny_session.monitor.manifolds["local/mood"] is tiny_session.manifolds["local/mood"]
    assert "local/mood" in tiny_session.profiles
    assert "local/mood" in tiny_session.manifolds
    return tiny_session


def test_delete_route_evicts_every_in_memory_consumer(fitted: Any, app_client: Any) -> None:
    client = app_client(fitted)
    response = client.delete("/drowse/v1/manifolds/local/mood")
    assert response.status_code == 200, response.text

    assert "local/mood" not in fitted.profiles
    assert "local/mood" not in fitted.manifolds
    assert "local/mood" not in fitted.monitor.probe_names
    with pytest.raises(ProfileNotRegisteredError):
        fitted.steering("0.2 !local/mood")
    with pytest.raises((ManifoldNotRegisteredError, ProfileNotRegisteredError)):
        fitted.steering("0.5 local/mood")


def test_fit_route_keeps_the_fit_it_adopted(fitted: Any, app_client: Any) -> None:
    """The route used to evict the manifold ``session.fit`` had just promoted,
    leaving the attached probe on one object and the next load on another."""
    client = app_client(fitted)
    response = client.post(
        "/drowse/v1/manifolds/local/mood/fit", json={"force": True},
    )
    assert response.status_code == 200, response.text

    loaded = fitted.manifolds["local/mood"]
    assert fitted.monitor.manifolds["local/mood"] is loaded
    assert "local/mood" in fitted.profiles
    with fitted.steering("0.5 local/mood"):
        pass


def test_evict_artifact_refuses_during_a_model_op(fitted: Any) -> None:
    from drowse.errors import ConcurrentExtractionError

    acquired = fitted.gen_lock.acquire(blocking=False)
    assert acquired
    try:
        import threading

        errors: list[BaseException] = []

        def _evict() -> None:
            try:
                fitted.evict_artifact("local", "mood")
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        worker = threading.Thread(target=_evict)
        worker.start()
        worker.join()
    finally:
        fitted.gen_lock.release()
    assert len(errors) == 1 and isinstance(errors[0], ConcurrentExtractionError)
    assert "local/mood" in fitted.manifolds
