"""Hugging Face Hub distribution: coords, push, and search.

Install-from-Hub (pull) runs through the shared pipeline and is covered,
alongside the local and ``.drowse`` sources, in ``tests/test_install.py``.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from drowse.base import hf
from drowse.base.paths import manifold_dir, tensor_filename
from drowse.store import hub
from tests.test_install import MODEL, author_manifold, write_fitted_manifold


def test_split_revision_plain():
    assert hf.split_revision("user/happy") == ("user/happy", None)


def test_split_revision_with_tag():
    assert hf.split_revision("user/happy@v1.2.0") == ("user/happy", "v1.2.0")


def test_split_revision_with_sha():
    assert hf.split_revision("user/happy@abcdef0") == ("user/happy", "abcdef0")


def test_split_revision_empty_rev_errors():
    with pytest.raises(hf.HFError, match="empty revision"):
        hf.split_revision("user/happy@")


def test_resolve_target_coord_explicit():
    assert hf.resolve_target_coord("happy", "bob/happy") == "bob/happy"


def test_resolve_target_coord_bad_as():
    with pytest.raises(hf.HFError, match="--as"):
        hf.resolve_target_coord("happy", "bob")


def test_resolve_target_coord_uses_whoami(monkeypatch: pytest.MonkeyPatch):
    import huggingface_hub
    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    assert hf.resolve_target_coord("happy", None) == "alice/happy"


# --------------------------------------------------------------------- push --


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "home"))
    return tmp_path


class _Uploads:
    """A fake ``HfApi`` that snapshots each uploaded folder before cleanup."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []

    def create_repo(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def upload_folder(self, **kwargs: Any) -> Any:
        source = Path(kwargs["folder_path"])
        snapshot = self.root / f"upload-{len(self.calls)}"
        snapshot.mkdir()
        for path in source.iterdir():
            (snapshot / path.name).write_bytes(path.read_bytes())
        self.calls.append({**kwargs, "snapshot": snapshot})
        return SimpleNamespace(oid="f" * 40)


def _archive_tensors(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as packed:
        return sorted(
            Path(name).name for name in packed.namelist()
            if name.endswith(".safetensors")
        )


def test_push_uploads_one_archive_summary_and_card(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold()
    api = _Uploads(home)
    monkeypatch.setattr(hub, "_hf_api", lambda: api)

    url, sha = hub.push_manifold(folder, "alice/mood")

    assert (url, sha) == ("https://huggingface.co/alice/mood", "f" * 40)
    assert api.created == [{
        "repo_id": "alice/mood", "repo_type": "model", "private": False,
        "exist_ok": True,
    }]
    [call] = api.calls
    snapshot = call["snapshot"]
    assert sorted(p.name for p in snapshot.iterdir()) == [
        ".gitattributes", "README.md", "manifold.json", "mood.drowse",
    ]
    assert "*.drowse" in call["delete_patterns"]
    summary = json.loads((snapshot / "manifold.json").read_text())
    assert summary["name"] == "mood" and "artifact_id" not in summary
    assert _archive_tensors(snapshot / "mood.drowse") == [tensor_filename(MODEL)]
    card = (snapshot / "README.md").read_text()
    assert "  - drowse-manifold" in card
    assert f'  - "{MODEL}"' in card
    assert '  archive: "mood.drowse"' in card
    assert '  node_count: 2' in card
    assert "drowse pack install alice/mood" in card


def test_push_card_frontmatter_round_trips_through_the_hub_parser(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub import ModelCard

    folder = author_manifold()
    api = _Uploads(home)
    monkeypatch.setattr(hub, "_hf_api", lambda: api)
    hub.push_manifold(folder, "alice/mood")
    card = ModelCard((api.calls[0]["snapshot"] / "README.md").read_text())
    block = card.data.to_dict()["drowse"]
    assert block == {
        "archive": "mood.drowse",
        "name": "mood",
        "description": "a mood axis",
        "tags": [],
        "node_count": 2,
        "domain_label": "discover-pca",
        "fit_mode": "pca",
        "tensor_models": [Path(tensor_filename(MODEL)).stem],
    }


def test_push_filters_fitted_pairs_by_model_and_variant(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold()
    write_fitted_manifold(folder, "other/model")
    api = _Uploads(home)
    monkeypatch.setattr(hub, "_hf_api", lambda: api)

    hub.push_manifold(folder, "alice/mood", model_scope="other/model")
    assert _archive_tensors(api.calls[-1]["snapshot"] / "mood.drowse") == [
        tensor_filename("other/model"),
    ]
    hub.push_manifold(folder, "alice/mood", variant="sae")
    snapshot = api.calls[-1]["snapshot"]
    assert _archive_tensors(snapshot / "mood.drowse") == []
    assert json.loads((snapshot / "manifold.json").read_text())["files"] == {}


def test_push_dry_run_never_touches_the_hub(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold()
    monkeypatch.setattr(
        hub, "_hf_api", lambda: pytest.fail("dry run reached the Hub"),
    )
    assert hub.push_manifold(folder, "alice/mood", dry_run=True) == (
        "https://huggingface.co/alice/mood", None,
    )


def test_push_rejects_a_stale_fit(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from drowse.store.archive import DrowseArchiveError

    folder = author_manifold()
    node = next((folder / "nodes").iterdir())
    node.write_text(json.dumps(["edited corpus"]))
    monkeypatch.setattr(hub, "_hf_api", lambda: pytest.fail("pushed a stale fit"))
    with pytest.raises(DrowseArchiveError, match="node hash"):
        hub.push_manifold(folder, "alice/mood")


def test_pushed_repo_installs_through_the_pull_path(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Push and pull speak one convention: what push uploads, pull installs."""
    from drowse.store.install import install_manifold
    from drowse.store.manifold_folder import ManifoldFolder

    folder = author_manifold()
    api = _Uploads(home)
    monkeypatch.setattr(hub, "_hf_api", lambda: api)
    hub.push_manifold(folder, "alice/mood")
    snapshot = api.calls[0]["snapshot"]
    sha = "a" * 40
    monkeypatch.setattr(hub, "_hf_api", lambda: SimpleNamespace(
        model_info=lambda coord, revision=None: SimpleNamespace(
            sha=sha, tags=["drowse-manifold"],
            siblings=[SimpleNamespace(rfilename=p.name) for p in snapshot.iterdir()],
        ),
    ))
    monkeypatch.setattr(
        hub, "_hf_hub_download",
        lambda coord, filename, revision: str(snapshot / filename),
    )
    installed = install_manifold("alice/mood", "team/mood")
    assert installed == manifold_dir("team", "mood")
    assert ManifoldFolder.load(installed).source == f"hf://alice/mood@{sha}"


# ------------------------------------------------------------------- search --


def _model(
    repo: str, *, archives: tuple[str, ...] = ("x.drowse",),
    card: dict[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        id=repo,
        siblings=[SimpleNamespace(rfilename=name) for name in (*archives, "README.md")],
        card_data=card,
    )


def test_search_reads_rows_from_card_data_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    block = {
        "name": "mood", "description": "d", "tags": ["affect"], "node_count": 3,
        "domain_label": "box(1d)", "fit_mode": "authored",
        "tensor_models": ["_zabc"],
    }

    def list_models(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return [
            _model("alice/mood", card={"drowse": block}),
            _model("bob/no-card"),
            _model("carol/loose", archives=()),
            _model("dave/two", archives=("a.drowse", "b.drowse")),
            _model("bare"),
        ]

    monkeypatch.setattr(hub, "_hf_api", lambda: SimpleNamespace(list_models=list_models))
    rows = hub.search_manifolds("mo")

    assert calls == [{
        "filter": "drowse-manifold", "limit": 20, "full": True,
        "cardData": True, "search": "mo",
    }]
    assert rows == [
        {**block, "namespace": "alice"},
        {
            "name": "no-card", "namespace": "bob", "description": "", "tags": [],
            "node_count": 0, "domain_label": "?", "fit_mode": "?",
            "tensor_models": [],
        },
        {
            "name": "bare", "namespace": "", "description": "", "tags": [],
            "node_count": 0, "domain_label": "?", "fit_mode": "?",
            "tensor_models": [],
        },
    ]


def test_search_without_query_lists_by_tag_and_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def list_models(**kwargs: Any) -> list[Any]:
        seen.update(kwargs)
        return [_model(f"u/m{i}") for i in range(30)]

    monkeypatch.setattr(hub, "_hf_api", lambda: SimpleNamespace(list_models=list_models))
    rows = hub.search_manifolds(None)
    assert "search" not in seen
    assert len(rows) == 20
