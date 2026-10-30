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

``FakeLogitsModel``/``CharTokenizer`` below are imported by test modules as
``tests.conftest`` helpers and stay here until those imports move to
``tests.support``.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest
import torch

from tests.support.sessions import TinyCheckpoint, TinySessionFactory


@pytest.fixture(scope="session", autouse=True)
def _isolated_drowse_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point every test (and every module/session fixture) at a scratch home.

    Tests that need their own home use ``isolated_home`` (or
    ``monkeypatch.setenv("DROWSE_HOME", ...)``); this only guarantees nothing
    ever falls through to the user's real home.
    """
    previous = os.environ.get("DROWSE_HOME")
    os.environ["DROWSE_HOME"] = str(tmp_path_factory.mktemp("drowse-home"))
    yield
    if previous is None:
        os.environ.pop("DROWSE_HOME", None)
    else:
        os.environ["DROWSE_HOME"] = previous


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
    from tests._gpu_model import gpu_model_id, load_or_skip_inaccessible

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


__all__ = [
    "FakeLogitsModel",
    "make_logits_model",
    "CharTokenizer",
]


class FakeLogitsModel:
    """Minimal HF-causal-LM-like stub returning ``.logits`` from a callable.

    The forward (``__call__``) accepts the kwargs the engine passes —
    ``input_ids`` and ``use_cache`` (both ignored beyond ``input_ids``,
    which is handed to ``logits_fn``) — and wraps ``logits_fn``'s output in
    a ``SimpleNamespace`` so consumers can read ``out.logits``.  ``logits_fn``
    is the single point of variation: a constant log-uniform distribution, a
    seeded-random one, a scripted argmax, etc.

    Args:
        logits_fn: maps an ``input_ids`` tensor ``(1, T)`` to a logits tensor;
            return ``(1, T, V)`` for per-position consumers (trajectory /
            joint-logprob reads) or ``(1, 1, V)`` for single-step decode.
        with_parameters: when ``True`` (the default) expose ``parameters()``
            yielding one throwaway CPU param so
            ``next(model.parameters()).device`` works.  Set ``False`` to model
            a stub whose consumer never inspects the device.
        with_past_key_values: when ``True`` set ``out.past_key_values`` to a
            sentinel object (the decode loop only checks it for ``None``); when
            ``False`` leave it absent.
        config / generation_config: optional ``SimpleNamespace`` attributes some
            consumers read (e.g. ``config.vocab_size``,
            ``generation_config.eos_token_id``).
    """

    def __init__(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        with_parameters: bool = True,
        with_past_key_values: bool = False,
        config: SimpleNamespace | None = None,
        generation_config: SimpleNamespace | None = None,
    ) -> None:
        self._logits_fn = logits_fn
        self._with_past_key_values = with_past_key_values
        # One throwaway parameter so ``next(model.parameters()).device`` works;
        # the value is irrelevant, only its (CPU) device matters.
        self._param = torch.zeros(1, requires_grad=False) if with_parameters else None
        if config is not None:
            self.config = config
        if generation_config is not None:
            self.generation_config = generation_config

    def parameters(self) -> Any:
        if self._param is None:
            raise AttributeError("FakeLogitsModel built with with_parameters=False")
        yield self._param

    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        use_cache: bool | None = None,
        **_kwargs: Any,
    ) -> Any:
        del use_cache  # stub is stateless; cache flag is ignored
        assert input_ids is not None, "FakeLogitsModel needs input_ids"
        logits = self._logits_fn(input_ids)
        out = SimpleNamespace(logits=logits)
        if self._with_past_key_values:
            out.past_key_values = object()
        return out


def make_logits_model(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    **kwargs: Any,
) -> FakeLogitsModel:
    """Thin factory for :class:`FakeLogitsModel` (reads well at call sites)."""
    return FakeLogitsModel(logits_fn, **kwargs)


class CharTokenizer:
    """Deterministic char-code tokenizer — encode text to its code points.

    The lightest tokenizer-ish helper the suite reaches for: a string becomes
    a ``(1, T)`` long tensor of ``ord(c) % mod`` ids (offset by 1 so id 0 stays
    free for a BOS fallback), and ``decode`` is best-effort.  Callable form
    matches the HF ``tokenizer(text, return_tensors="pt")`` surface; ``encode``
    is the bare-list/tensor sibling.

    Args:
        mod: modulus bounding the id range (so a small ``vocab`` model stays in
            range).  Defaults to a value comfortably above ASCII.
    """

    bos_token_id = 0

    def __init__(self, mod: int = 256) -> None:
        self._mod = mod

    def _ids(self, text: str) -> list[int]:
        return [(ord(c) % self._mod) + 1 for c in text] or [self.bos_token_id]

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens  # always returns a pt tensor
        return {"input_ids": torch.tensor([self._ids(text)], dtype=torch.long)}

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        ids = self._ids(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids
