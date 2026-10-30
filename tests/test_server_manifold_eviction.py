"""C4: manifold folders changed over HTTP must not leave stale in-memory steering.

Runs the real server over a real tiny session: fit a 2-node manifold, steer
with it and attach it as a probe, then change the folder through the native
routes and check what the session still holds.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from drowse.base.paths import manifold_dir
from drowse.errors import ManifoldNotRegisteredError, ProfileNotRegisteredError
from drowse.store.manifold_authoring import create_discover_manifold_folder
from tests.support.sessions import baseline_corpus

pytestmark = pytest.mark.usefixtures("isolated_home")


def _fit_mood(session: Any) -> None:
    create_discover_manifold_folder(
        "local", "mood", "test axis", fit_mode="pca",
        node_corpora={
            "calm": baseline_corpus("I feel calm and settled today."),
            "angry": baseline_corpus("I am furious about all of this!"),
        },
        hyperparams={"max_dim": 1},
    )
    session.fit(manifold_dir("local", "mood"))


@pytest.fixture
def fitted(tiny_session: Any) -> Any:
    _fit_mood(tiny_session)
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


@pytest.fixture
def folded_probe(tiny_session: Any) -> Any:
    """Probes attached *after* an ablation registered the folded profile read a
    fold of that profile, not the registry's manifold object."""
    _fit_mood(tiny_session)
    with tiny_session.steering("0.2 !local/mood"):
        pass
    assert "local/mood" in tiny_session.profiles
    tiny_session.add_probe("local/mood")
    tiny_session.add_probe("local/mood", as_name="mood-alias")
    probes = tiny_session.monitor.manifolds
    assert probes["local/mood"] is not tiny_session.manifolds["local/mood"]
    return tiny_session


def test_delete_route_drops_probes_folded_from_the_profile(
    folded_probe: Any, app_client: Any,
) -> None:
    client = app_client(folded_probe)
    response = client.delete("/drowse/v1/manifolds/local/mood")
    assert response.status_code == 200, response.text

    assert "local/mood" not in folded_probe.profiles
    assert "local/mood" not in folded_probe.monitor.probe_names
    assert "mood-alias" not in folded_probe.monitor.probe_names


def test_fit_route_refolds_probes_folded_from_the_profile(
    folded_probe: Any, app_client: Any,
) -> None:
    before = dict(folded_probe.monitor.manifolds)
    client = app_client(folded_probe)
    response = client.post(
        "/drowse/v1/manifolds/local/mood/fit", json={"force": True},
    )
    assert response.status_code == 200, response.text

    after = folded_probe.monitor.manifolds
    for name in ("local/mood", "mood-alias"):
        assert name in after
        assert after[name] is not before[name]
        # Still the profile's 1-node fold, rebuilt from the new fit.
        assert after[name].node_labels == before[name].node_labels
    refit = folded_probe.profiles["local/mood"]
    probe_layer = next(iter(after["local/mood"].layers))
    basis = after["local/mood"].layers[probe_layer].basis.reshape(-1)
    direction = refit[probe_layer].to(basis).reshape(-1)
    assert torch.allclose(basis, direction / direction.norm(), atol=1e-5)


def _assert_evicted(session: Any) -> None:
    assert "local/mood" not in session.profiles
    assert "local/mood" not in session.manifolds
    assert "local/mood" not in session.monitor.probe_names


def test_forced_install_evicts_the_replaced_folder(
    fitted: Any, app_client: Any, tmp_path: Any,
) -> None:
    import shutil

    source = tmp_path / "src" / "mood"
    shutil.copytree(manifold_dir("local", "mood"), source)
    client = app_client(fitted)
    response = client.post("/drowse/v1/manifolds/install", json={
        "target": str(source), "as": "local/mood", "force": True,
    })
    assert response.status_code == 201, response.text
    _assert_evicted(fitted)


_WEEKDAY = {
    "name": "weekday",
    "slot": "[DAY]",
    "values": ["Monday", "Tuesday"],
    "contexts": [{
        "turns": [{"role": "user", "content": "which day?"}],
        "assistant": "today is [DAY]",
    }],
}


def test_forced_from_template_evicts_the_replaced_folder(
    fitted: Any, app_client: Any,
) -> None:
    client = app_client(fitted)
    assert client.post("/drowse/v1/templates", json=_WEEKDAY).status_code == 201
    response = client.post("/drowse/v1/manifolds/from-template", json={
        "name": "mood", "template_ref": "local/weekday", "force": True,
    })
    assert response.status_code == 201, response.text
    _assert_evicted(fitted)


def test_forced_from_template_refuses_before_writing_while_busy(
    fitted: Any, app_client: Any,
) -> None:
    """The overwrite used to land first and the eviction then fail on the busy
    engine, leaving every in-memory consumer on a folder that no longer
    exists."""
    client = app_client(fitted)
    assert client.post("/drowse/v1/templates", json=_WEEKDAY).status_code == 201
    manifest = manifold_dir("local", "mood") / "manifold.json"
    before = manifest.read_bytes()
    assert fitted.gen_lock.acquire(blocking=False)
    try:
        response = client.post("/drowse/v1/manifolds/from-template", json={
            "name": "mood", "template_ref": "local/weekday", "force": True,
        })
    finally:
        fitted.gen_lock.release()
    assert response.status_code == 409, response.text
    assert manifest.read_bytes() == before
    assert "local/mood" in fitted.manifolds
