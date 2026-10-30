"""``DrowseSession.bake``: the model-scoped merge, its fingerprint guard, and registration."""

# pyright: reportUnusedVariable=false

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.session.session import DrowseSession
from tests._fakes import make_mock_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_session():
    # Shared native-API wiring lives in tests/_fakes.py.
    session = make_mock_session()

    monitor = MagicMock()
    monitor.probe_names = []
    monitor.profiles = {}
    session._monitor = monitor
    session._tokenizer = MagicMock()
    session._layers = []
    session.events = MagicMock()
    session.events.subscribe = lambda cb: (lambda: None)
    # Bind the real method so its contract (including the loaded-weight
    # fingerprint it forwards into the merge) is under test rather than a
    # mock's return value.
    session.bake = MethodType(DrowseSession.bake, session)
    return session


@pytest.fixture
def session() -> Any:
    return _mock_session()


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
        from drowse.fit.capture import (
            fold_directions_to_subspace, folded_directions,
        )
        from drowse.store.manifold_tensors import load_manifold
        from drowse.store.manifold_authoring import BakedTensor, publish_baked_manifold
        from drowse.base.paths import tensor_filename

        dirs = {0: torch.tensor([1.0, 0.0, 0.0, 0.0]), 5: torch.tensor([0.0, 2.0, 0.0, 0.0])}
        from tests._whitener import isotropic_whitener
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

        with patch(
            "drowse.fit.bake.merge_into_manifold",
            return_value=merged_folder,
        ) as m:
            name, profile = session.bake("noble", "0.3 default/honest + 0.4 default/warm")
        assert name == "noble"
        assert profile.layers == [0, 5]
        # Confirm we forwarded model + expression + force=True, and that the
        # loaded-weight fingerprint guard rides along.
        kwargs = m.call_args.kwargs
        args = m.call_args.args
        assert args[0] == "noble"
        assert args[1] == "0.3 default/honest + 0.4 default/warm"
        assert args[2] == "test/model"
        assert kwargs["force"] is True
        assert kwargs["expected_model_fingerprint"] == loaded_model_fingerprint(
            session._model, session.model_id,
        )
        # Auto-register on the session with the folded merged direction.
        steer_mock = cast(MagicMock, session.steer)
        steer_mock.assert_called_once()
        call_name, call_profile = steer_mock.call_args.args
        assert call_name == "noble"
        assert call_profile.layers == [0, 5]
        assert torch.allclose(call_profile[0].float(), expected[0].float(), atol=1e-5)
        assert torch.allclose(call_profile[5].float(), expected[5].float(), atol=1e-5)

    @pytest.mark.parametrize("message", [
        "merge requires at least one component",
        "component default/missing not installed",
    ])
    def test_merge_errors_propagate_as_400(self, session: Any, message: str):
        from drowse.fit.bake import MergeError
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
        from drowse.fit.bake import MergeError
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
        from drowse.fit.bake import MergeError
        empty_folder = tmp_path / "local" / "x"
        empty_folder.mkdir(parents=True)

        with patch(
            "drowse.fit.bake.merge_into_manifold",
            return_value=empty_folder,
        ):
            with pytest.raises(MergeError, match="produced no tensor"):
                session.bake("x", "0.3 default/a")
