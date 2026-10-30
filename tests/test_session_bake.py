"""``Extraction.bake`` (behind ``DrowseSession.bake``): the model-scoped merge,
its fingerprint guard, and registration."""

# pyright: reportUnusedVariable=false

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from drowse.model.fingerprint import loaded_model_fingerprint
from tests.support.server import mock_session
from tests.support.extraction import make_extraction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_session(monkeypatch: pytest.MonkeyPatch):
    session = mock_session()

    monitor = MagicMock()
    monitor.probe_names = []
    monitor.profiles = {}
    session._monitor = monitor
    session._tokenizer = MagicMock()
    session._layers = []
    session.events = MagicMock()
    session.events.subscribe = lambda cb: (lambda: None)
    # A real Extraction over the mock model handle, so the contract
    # (including the loaded-weight fingerprint forwarded into the merge) is
    # under test rather than a mock's return value; ``session.bake`` is the
    # extraction's bake, which ``DrowseSession.bake`` delegates to.
    session.model = session._model
    session.bake = partial(
        make_extraction(
            monkeypatch, handle=session,
            register_profile=session._registry.put_profile,
        ).extraction.bake,
        force=False, strict=False,
    )
    return session


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Any:
    return _mock_session(monkeypatch)


def _publish_noble(session: Any, tmp_path: Path) -> tuple[Path, Any]:
    """A real baked ``local/noble`` folder for ``session``'s model, and its fold."""
    from drowse.geometry.fold import (
        fold_directions_to_subspace, folded_directions,
    )
    from drowse.store.manifold_tensors import load_manifold
    from drowse.store.manifold_authoring import BakedTensor, publish_baked_manifold
    from drowse.base.paths import tensor_filename

    dirs = {0: torch.tensor([1.0, 0.0, 0.0, 0.0]), 5: torch.tensor([0.0, 2.0, 0.0, 0.0])}
    from tests.support.whitener import isotropic_whitener
    means = {layer: torch.zeros_like(direction) for layer, direction in dirs.items()}
    manifold = fold_directions_to_subspace(
        "noble", dirs, means,
        whitener=isotropic_whitener(dirs, 4), label="merged",
    )
    from drowse.base.integrity import hash_file

    honest_source = tmp_path / "honest-source.safetensors"
    warm_source = tmp_path / "warm-source.safetensors"
    honest_source.write_bytes(b"manifest-proven honest source")
    warm_source.write_bytes(b"manifest-proven warm source")
    merged_folder = publish_baked_manifold(
        "local", "noble", "merged",
        [BakedTensor(
            model_id=session.model_id, manifold=manifold,
            model_fingerprint=loaded_model_fingerprint(
                session._model, session.model_id,
            ),
            components={
            "0": {
                "selector": "default/honest",
                "alpha": 0.3,
                "tensor_sha256": hash_file(honest_source),
            },
            "1": {
                "selector": "default/warm",
                "alpha": 0.4,
                "tensor_sha256": hash_file(warm_source),
            },
        },
        )],
        method="merge",
    )
    tensor_path = merged_folder / tensor_filename(session.model_id)
    expected = folded_directions(load_manifold(tensor_path))

    return merged_folder, expected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMergeVector:
    def test_happy_path(self, session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # The merge now lands a corpus-less ``fit_mode="baked"`` manifold; the
        # route loads + folds it back to a steering Profile.  Build a real one
        # on disk so the route's ``load_manifold`` + fold has a valid target
        # via the mocked ``merge_into_manifold`` output.
        monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
        merged_folder, expected = _publish_noble(session, tmp_path)

        with patch(
            "drowse.fit.bake.merge_into_manifold",
            return_value=merged_folder,
        ) as m:
            name, profile = session.bake("noble", "0.3 default/honest + 0.4 default/warm")
        assert name == "noble"
        assert profile.layers == [0, 5]
        # Confirm we forwarded model + expression, the non-destructive
        # ``force=False`` default, and the loaded-weight fingerprint guard.
        kwargs = m.call_args.kwargs
        args = m.call_args.args
        assert args[0] == "noble"
        assert args[1] == "0.3 default/honest + 0.4 default/warm"
        assert args[2] == "test/model"
        assert kwargs["force"] is False
        assert kwargs["expected_model_fingerprint"] == loaded_model_fingerprint(
            session._model, session.model_id,
        )
        # Auto-register on the session with the folded merged direction.
        register = cast(MagicMock, session._registry.put_profile)
        register.assert_called_once()
        call_name, call_profile = register.call_args.args
        assert call_name == "noble"
        assert call_profile.layers == [0, 5]
        assert torch.allclose(call_profile[0].float(), expected[0].float(), atol=1e-5)
        assert torch.allclose(call_profile[5].float(), expected[5].float(), atol=1e-5)

    @pytest.mark.parametrize("message", [
        "merge requires at least one component",
        "component default/missing not installed",
    ])
    def test_merge_errors_propagate_as_400(self, session: Any, message: str):
        from drowse.errors import MergeError
        with patch("drowse.fit.bake.merge_into_manifold", side_effect=MergeError(message)):
            with pytest.raises(MergeError) as error:
                session.bake("x", "0.3 default/missing")
        assert error.value.user_message() == (400, message)

    def test_fingerprint_mismatch_400(self, session: Any):
        """Components fitted against other weights are refused.

        The session forwards its live fingerprint into the merge, so the
        guard fires rather than baking a tensor the loaded model never
        produced.
        """
        from drowse.errors import MergeError
        live = loaded_model_fingerprint(session._model, session.model_id)
        seen: dict[str, Any] = {}

        def _merge(*_args: Any, **kwargs: Any) -> Path:
            seen.update(kwargs)
            expected = kwargs.get("expected_model_fingerprint")
            if expected is not None and expected != "stale-fingerprint":
                raise MergeError(
                    "cannot bake 'x' for test/model: components were fitted "
                    "for different loaded weights; refit them for this "
                    "session first"
                )
            raise AssertionError("guard did not fire")

        with patch("drowse.fit.bake.merge_into_manifold", side_effect=_merge):
            with pytest.raises(MergeError, match="different loaded weights"):
                session.bake("x", "0.3 default/a")
        assert seen["expected_model_fingerprint"] == live

    def test_no_tensor_for_model_400(self, session: Any, tmp_path: Path):
        """If merge_into_manifold returns a folder with no tensor for our model.

        Defensive — merge_into_manifold should always produce one when given a
        model arg, but if it doesn't (e.g. silent skip), the session raises
        ``MergeError``.
        """
        from drowse.errors import MergeError
        empty_folder = tmp_path / "local" / "x"
        empty_folder.mkdir(parents=True)

        with patch(
            "drowse.fit.bake.merge_into_manifold",
            return_value=empty_folder,
        ):
            with pytest.raises(MergeError, match="produced no tensor"):
                session.bake("x", "0.3 default/a")

    def test_replacing_an_existing_bake_needs_explicit_force(self, session: Any):
        """C6: ``bake`` used to default to ``force=True`` and silently replace
        an existing ``local/<name>`` manifold (authored corpora included)."""
        from drowse.errors import MergeError

        seen: list[Any] = []

        def _merge(*_args: Any, **kwargs: Any) -> Path:
            seen.append(kwargs["force"])
            raise MergeError("stop after the forwarded flags")

        with patch("drowse.fit.bake.merge_into_manifold", side_effect=_merge):
            with pytest.raises(MergeError):
                session.bake("x", "0.3 default/a")
            with pytest.raises(MergeError):
                session.bake("x", "0.3 default/a", force=True)
        assert seen == [False, True]


class TestForcedBakeReplacesInMemoryConsumers:
    def test_previous_local_artifact_is_evicted_under_the_model_lock(
        self, session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A forced bake over ``local/<name>`` used to register only the bare
        profile, leaving a loaded ``local/<name>`` manifold (and the probes and
        folds reading it) on the old tensors.  It now evicts every consumer of
        the folder under the exclusive-model lock before the new profile lands."""
        from contextlib import contextmanager

        from drowse.session.registry import ArtifactRegistry, ModelBinding
        from tests.support.extraction import folded_vector_manifold, make_extraction

        monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
        merged_folder, expected = _publish_noble(session, tmp_path)
        registry = ArtifactRegistry(ModelBinding(
            model=None, tokenizer=None, model_id=session.model_id,
            device=torch.device("cpu"), dtype=torch.float32,
        ))
        registry.put_manifold("local/noble", folded_vector_manifold("noble"))
        registry.put_profile("noble", {0: torch.ones(2)})
        events: list[Any] = []

        @contextmanager
        def exclusive(busy_msg: str, *, phase_msg: str) -> Any:
            events.append(("lock", "bake" in busy_msg, "generation" in phase_msg))
            try:
                yield
            finally:
                events.append("unlock")

        def evict(namespace: str, name: str, *, feature: str | None = None) -> None:
            events.append(("evict", namespace, name))
            registry.evict_artifact(namespace, name, feature=feature)

        def register(name: str, profile: Any) -> None:
            events.append(("register", name))
            registry.put_profile(name, profile)

        extraction = make_extraction(
            monkeypatch, handle=session, exclusive=exclusive,
            evict=evict, register_profile=register,
        ).extraction
        with patch("drowse.fit.bake.merge_into_manifold", return_value=merged_folder):
            extraction.bake("noble", "0.3 default/honest", force=True, strict=False)

        assert events == [
            ("lock", True, True), ("evict", "local", "noble"),
            ("register", "noble"), "unlock",
        ]
        assert "local/noble" not in registry.manifolds
        assert torch.allclose(registry.profiles["noble"][5].float(), expected[5].float(), atol=1e-5)


def test_session_bake_delegates_to_the_extraction(tiny_session: Any) -> None:
    from drowse.errors import DrowseError

    with pytest.raises(DrowseError):
        tiny_session.bake("noble", "0.3 local/missing")
