"""``session.extract`` of an already-installed concept fits it in place.

Without a namespace, ``extract("formal.casual")`` used to author a second
``local/formal.casual`` next to the bundled ``default/formal.casual``, after
which every bare ``formal.casual`` reference raised ``AmbiguousSelectorError``
(the AGENTS.md quick-start example broke the same way).
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from drowse.base.paths import manifold_dir
from drowse.session.session import DrowseSession, GenState
from drowse.store import selectors
from drowse.store.manifolds import create_discover_manifold_folder
from drowse.store.selectors import AmbiguousSelectorError


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "home"))
    selectors.invalidate()
    yield
    selectors.invalidate()


def _install(namespace: str) -> Path:
    return create_discover_manifold_folder(
        namespace, "formal.casual", "Bipolar axis: formal (+) vs casual (-).",
        fit_mode="pca",
        node_corpora={"formal": ["Good day."], "casual": ["hey"]},
        hyperparams={"max_dim": 1, "var_threshold": 0.7},
        node_kinds={"formal": "abstract", "casual": "abstract"},
    )


def _session(fitted: list[Path], generated: list[list[str]]) -> Any:
    s: Any = DrowseSession.__new__(DrowseSession)
    s._gen_phase = GenState.IDLE
    s._gen_lock = threading.Lock()
    s._steering_composer = SimpleNamespace(_stack=[])

    def generate_responses(concepts: list[str], *_a: Any, **_k: Any) -> Any:
        generated.append(list(concepts))
        return {c: [f"{c} reply"] for c in concepts}

    def fit_concept(name: str, folder: Any, **_kwargs: Any) -> Any:
        fitted.append(Path(folder))
        return name, None

    s.generate_responses = generate_responses
    s._fit_concept_manifold = fit_concept
    return s


def test_extract_fits_the_installed_bundled_concept_in_place() -> None:
    _install("default")
    fitted: list[Path] = []
    generated: list[list[str]] = []

    name, _ = _session(fitted, generated).extract("formal.casual")

    assert name == "formal.casual"
    assert fitted == [manifold_dir("default", "formal.casual")]
    assert generated == []
    assert not manifold_dir("local", "formal.casual").exists()
    # The bare name still resolves to exactly one manifold.
    assert len(selectors.resolve(selectors.parse("formal.casual"))) == 1


def test_force_extract_of_a_bundled_concept_needs_a_namespace() -> None:
    _install("default")
    fitted: list[Path] = []
    generated: list[list[str]] = []
    session = _session(fitted, generated)

    with pytest.raises(ValueError, match="namespace='local'"):
        session.extract("formal.casual", force=True)
    assert generated == [] and fitted == []

    session.extract("formal.casual", force=True, namespace="local")
    assert generated == [["formal", "casual"]]
    assert fitted == [manifold_dir("local", "formal.casual")]


def test_extract_of_a_new_concept_authors_under_local() -> None:
    fitted: list[Path] = []
    generated: list[list[str]] = []

    _session(fitted, generated).extract("formal.casual")

    assert generated == [["formal", "casual"]]
    assert fitted == [manifold_dir("local", "formal.casual")]


def test_extract_refuses_an_already_ambiguous_bare_name() -> None:
    _install("default")
    _install("local")
    with pytest.raises(AmbiguousSelectorError):
        _session([], []).extract("formal.casual")
