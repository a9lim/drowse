"""Where ``extract`` authors and fits a concept, and when it reuses a folder.

Without a namespace, ``extract("formal.casual")`` used to author a second
``local/formal.casual`` next to the bundled ``default/formal.casual``, after
which every bare ``formal.casual`` reference raised ``AmbiguousSelectorError``.
An installed concept is now fitted where it lives; an installed folder with an
upstream (bundled or hub-installed) is never regenerated in place, since
``pack refresh`` would restore its corpus.  The author-or-reuse decision is
made under the folder's manifest lock.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from drowse.base.paths import manifold_dir
from drowse.events import ManifoldExtracted
from drowse.store import selectors
from drowse.store.manifold_authoring import create_discover_manifold_folder
from drowse.store.selectors import AmbiguousSelectorError
from tests.support.extraction import make_extraction


@pytest.fixture(autouse=True)
def _home(isolated_home: Path) -> Iterator[None]:
    yield


def _install(namespace: str, *, source: str | None = None) -> Path:
    folder = create_discover_manifold_folder(
        namespace, "formal.casual", "Bipolar axis: formal (+) vs casual (-).",
        fit_mode="pca",
        node_corpora={"formal": ["Good day."], "casual": ["hey"]},
        hyperparams={"max_dim": 1, "var_threshold": 0.7},
        node_kinds={"formal": "abstract", "casual": "abstract"},
    )
    if source is not None:
        manifest = folder / "manifold.json"
        payload = json.loads(manifest.read_text())
        payload["source"] = source
        manifest.write_text(json.dumps(payload))
    selectors.invalidate()
    return folder


_HUB = "hf://alice/formal.casual@" + "0" * 40


def test_extract_fits_the_installed_bundled_concept_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install("default")
    rig = make_extraction(monkeypatch)

    name, _ = rig.extraction.extract("formal.casual")

    assert name == "formal.casual"
    assert rig.fitted == [manifold_dir("default", "formal.casual")]
    assert rig.generated == []
    assert not manifold_dir("local", "formal.casual").exists()
    # The bare name still resolves to exactly one manifold.
    assert len(selectors.resolve(selectors.parse("formal.casual"))) == 1


def test_extract_fits_a_hub_installed_concept_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install("alice", source=_HUB)
    rig = make_extraction(monkeypatch)

    rig.extraction.extract("formal.casual")

    assert rig.fitted == [manifold_dir("alice", "formal.casual")]
    assert rig.generated == []


@pytest.mark.parametrize(
    ("namespace", "source"), [("default", None), ("alice", _HUB), ("alice", "bundled")],
)
def test_force_extract_of_an_upstream_installed_concept_needs_a_namespace(
    namespace: str, source: str | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pack refresh`` restores the corpus of a bundled or hub-installed pack."""
    installed = _install(namespace, source=source)
    before = (installed / "manifold.json").read_bytes()
    rig = make_extraction(monkeypatch)

    with pytest.raises(ValueError, match="namespace='local'"):
        rig.extraction.extract("formal.casual", force=True)
    assert rig.generated == [] and rig.fitted == []
    assert (installed / "manifold.json").read_bytes() == before

    rig.extraction.extract("formal.casual", force=True, namespace="local")
    assert rig.generated == [["formal", "casual"]]
    assert rig.fitted == [manifold_dir("local", "formal.casual")]


def test_force_extract_of_a_local_concept_regenerates_it_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install("local")
    rig = make_extraction(monkeypatch)

    rig.extraction.extract("formal.casual", force=True)

    assert rig.generated == [["formal", "casual"]]
    assert rig.fitted == [manifold_dir("local", "formal.casual")]


def test_extract_of_a_new_concept_authors_under_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = make_extraction(monkeypatch)

    name, profile = rig.extraction.extract("formal.casual")

    assert rig.generated == [["formal", "casual"]]
    folder = manifold_dir("local", "formal.casual")
    assert rig.fitted == [folder]
    # One adoption, one event: the duplicate profile refresh is gone.
    assert rig.adopted == [folder]
    assert profile.metadata["name"] == name == "formal.casual"


def test_extract_emits_one_event_with_the_variant_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = make_extraction(monkeypatch)
    seen: list[ManifoldExtracted] = []
    rig.events.subscribe(
        lambda event: seen.append(event) if isinstance(event, ManifoldExtracted) else None,
    )

    name, _ = rig.extraction.extract("formal.casual", role="pirate")

    assert name == "formal.casual:role-pirate"
    assert [event.name for event in seen] == ["formal.casual:role-pirate"]
    assert rig.fit_kwargs[0]["emit_event"] is False


def test_extract_refuses_an_already_ambiguous_bare_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install("default")
    _install("local")
    with pytest.raises(AmbiguousSelectorError):
        make_extraction(monkeypatch).extraction.extract("formal.casual")


def test_role_mismatch_on_an_installed_concept_names_the_namespace_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One error with the remedy that works, not a force=True dead end."""
    _install("default")
    rig = make_extraction(monkeypatch)

    with pytest.raises(ValueError, match="namespace='local'") as excinfo:
        rig.extraction.extract("formal.casual", role="pirate")
    assert "force=True" not in str(excinfo.value)
    assert rig.generated == [] and rig.fitted == []

    rig.extraction.extract("formal.casual", role="pirate", namespace="local")
    assert rig.fitted == [manifold_dir("local", "formal.casual")]


def test_role_mismatch_on_a_local_concept_names_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install("local")
    rig = make_extraction(monkeypatch)

    with pytest.raises(ValueError, match="force=True"):
        rig.extraction.extract("formal.casual", role="pirate")

    rig.extraction.extract("formal.casual", role="pirate", force=True)
    assert rig.fitted == [manifold_dir("local", "formal.casual")]


def test_extract_from_corpora_resolves_the_namespace_like_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install("default")
    rig = make_extraction(monkeypatch)

    rig.extraction.extract_from_corpora("formal.casual", ["Indeed."], ["yo"])
    assert rig.fitted == [manifold_dir("default", "formal.casual")]
    assert not manifold_dir("local", "formal.casual").exists()

    with pytest.raises(ValueError, match="namespace='local'"):
        rig.extraction.extract_from_corpora(
            "formal.casual", ["Indeed."], ["yo"], force=True,
        )


def test_extract_authors_when_the_folder_vanishes_before_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``rm`` that wins the lock race leaves a folder to author, not an assert.

    The author-or-reuse decision used to be taken before the manifest lock, so
    a folder removed in between reached authoring with no corpus.
    """
    import drowse.store.manifold_folder as folder_mod

    installed = _install("local")
    real_lock = folder_mod._locked_manifest
    entries: list[Path] = []

    @contextmanager
    def _racing_lock(folder: Path) -> Iterator[None]:
        entries.append(folder)
        if len(entries) == 1:
            shutil.rmtree(installed)
        with real_lock(folder):
            yield

    monkeypatch.setattr(folder_mod, "_locked_manifest", _racing_lock)
    rig = make_extraction(monkeypatch)

    rig.extraction.extract("formal.casual", namespace="local")

    assert rig.generated == [["formal", "casual"]]
    assert rig.fitted == [installed]
    assert (installed / "manifold.json").is_file()


def test_extract_reuses_a_folder_authored_during_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation runs outside the lock; the second pass re-decides under it."""
    rig = make_extraction(monkeypatch, on_generate=lambda: _install("local"))

    rig.extraction.extract("formal.casual", namespace="local")

    folder = manifold_dir("local", "formal.casual")
    assert rig.generated == [["formal", "casual"]]
    assert rig.fitted == [folder]
    # The concurrent writer's corpus is kept, not replaced.
    assert json.loads((folder / "nodes" / "00_formal.json").read_text()) == ["Good day."]
