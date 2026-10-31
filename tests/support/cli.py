"""A model-free stand-in for ``DrowseSession`` in CLI runner tests.

Tests patch ``DrowseSession.from_pretrained`` (the CLI's one construction
seam, reached through ``drowse.cli.context.open_session``) so the real
narration, YAML sampling defaults, and close-on-exit all run.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from drowse.generation.decode import GenerationConfig


class FakeSession:
    """The attributes ``open_session`` touches, plus recorded kwargs."""

    def __init__(self, model_id: str = "fake/model", **kwargs: Any) -> None:
        self.model_id = model_id
        self.load_kwargs = kwargs
        self.model_info = {
            "model_type": "fake", "num_layers": 2, "hidden_dim": 8,
            "vram_used_gb": 0.0,
        }
        self.probes: dict[str, Any] = {}
        self.config = GenerationConfig()
        self.closed = False

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


def install_fake_session(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[..., Any] = FakeSession,
) -> list[Any]:
    """Make ``DrowseSession.from_pretrained`` build ``factory(model_id, **kw)``;
    returns the list of sessions created, in order."""
    from drowse.session.session import DrowseSession

    created: list[Any] = []

    def from_pretrained(model_id: str, **kwargs: Any) -> Any:
        session = factory(model_id, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(DrowseSession, "from_pretrained", staticmethod(from_pretrained))
    return created


def poison_session(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make any model load raise; returns the list of attempted model ids."""
    from drowse.session.session import DrowseSession

    attempts: list[str] = []

    def from_pretrained(model_id: str, **_kwargs: Any) -> Any:
        attempts.append(model_id)
        raise RuntimeError("model load attempted")

    monkeypatch.setattr(DrowseSession, "from_pretrained", staticmethod(from_pretrained))
    return attempts
