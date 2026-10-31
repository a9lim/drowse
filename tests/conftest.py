"""Suite-wide pytest fixtures and hooks.

Fixtures (helpers they build on live in ``tests/support``):

- ``_isolated_drowse_home`` (session, autouse): every worker runs under a
  scratch ``DROWSE_HOME``; nothing reaches the user's real home.
- ``isolated_home``: a fresh per-test ``DROWSE_HOME`` with selector caches
  reset, for tests that write artifacts.
- ``tiny_checkpoint`` (session): a 4-layer Llama-3-format checkpoint with a
  byte-level tokenizer, written once per worker.
- ``make_tiny_session``: a :class:`~tests.support.sessions.TinySessionFactory`;
  every session it builds is closed at teardown.
- ``tiny_session``: one real, warmed CPU ``DrowseSession`` (``probes=[]``).
- ``app_client``: ``app_client(session, **create_app_kwargs)`` returns a
  started FastAPI ``TestClient`` over the real app.
- ``gpu_session`` (session): the real GPU model (``DROWSE_TEST_MODEL``),
  skipped without CUDA/MPS; tests using it are marked ``gpu`` automatically.

Markers: ``perf`` tests measure wall-clock throughput and are skipped inside
xdist workers (contention makes them meaningless); run them serially with
``pytest -m perf -p no:xdist``.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import torch

from tests.support.sessions import TinyCheckpoint, TinySessionFactory, drowse_home


@pytest.fixture(scope="session", autouse=True)
def _isolated_drowse_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point every test (and every module/session fixture) at a scratch home.

    Tests that need their own home use ``isolated_home`` (or
    ``monkeypatch.setenv("DROWSE_HOME", ...)``); this only guarantees nothing
    ever falls through to the user's real home.
    """
    with drowse_home(tmp_path_factory.mktemp("drowse-home")):
        yield


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fresh ``DROWSE_HOME`` for one test, with selector caches reset."""
    from drowse.store import selectors

    home = tmp_path / "drowse-home"
    home.mkdir()
    monkeypatch.setenv("DROWSE_HOME", str(home))
    selectors.invalidate()
    yield home
    selectors.invalidate()


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return TinyCheckpoint.write(tmp_path_factory.mktemp("tiny-checkpoint"))


@pytest.fixture(scope="session")
def _tiny_neutral_seed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tiny-neutral-seed")


@pytest.fixture
def make_tiny_session(
    tiny_checkpoint: TinyCheckpoint, _tiny_neutral_seed: Path,
) -> Iterator[TinySessionFactory]:
    factory = TinySessionFactory(tiny_checkpoint, _tiny_neutral_seed)
    yield factory
    factory.close()


@pytest.fixture
def tiny_session(make_tiny_session: TinySessionFactory) -> Any:
    return make_tiny_session()


@pytest.fixture
def app_client() -> Iterator[Callable[..., Any]]:
    from fastapi.testclient import TestClient

    from drowse.server.app import create_app

    with ExitStack() as stack:
        def _client(session: Any, **create_app_kwargs: Any) -> TestClient:
            app = create_app(session, **create_app_kwargs)
            return stack.enter_context(TestClient(app))

        yield _client


@pytest.fixture(scope="session")
def gpu_session() -> Iterator[Any]:
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("No GPU backend available (neither CUDA nor MPS)")
    from drowse.session.session import DrowseSession
    from tests.support.gpu import gpu_model_id, load_or_skip_inaccessible

    model_id = gpu_model_id()
    session = load_or_skip_inaccessible(
        lambda: DrowseSession.from_pretrained(model_id, device="auto", probes=[]),
        model_id,
    )
    yield session
    session.close()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    in_xdist_worker = hasattr(config, "workerinput")
    serial_only = pytest.mark.skip(
        reason="perf test: run serially with `pytest -m perf -p no:xdist`",
    )
    for item in items:
        if "gpu_session" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.gpu)
        if in_xdist_worker and item.get_closest_marker("perf") is not None:
            item.add_marker(serial_only)
