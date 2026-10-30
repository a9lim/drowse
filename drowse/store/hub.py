"""Hugging Face Hub distribution for manifolds: push, search, and pull.

One repository convention, shared with the drowse.ai browser client.  A
manifold repo is a Hub *model* repo tagged ``drowse-manifold`` carrying, at the
root:

* ``<name>.drowse`` — the complete transport archive (:mod:`drowse.store.archive`),
  the only file an install reads;
* ``manifold.json`` — a summary copy of the archived manifest (browsers that
  render a row without downloading the archive read it);
* ``README.md`` — a model card whose YAML frontmatter carries the discovery
  tags, ``base_model`` back-links, and a ``drowse:`` block with the search-row
  fields, so a search needs no per-row request.

Push exports the archive and uploads those three files in one commit; pull
resolves the requested revision to its immutable commit, downloads the single
root ``.drowse`` at that commit, and installs it through the one install
pipeline (:mod:`drowse.store.install`) stamped ``source = hf://<repo>@<sha>``.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

from drowse.base.hf import HFError, _hf_api, _hf_hub_download
from drowse.store.manifolds import ManifoldFolder

#: Row ceiling for a Hub search; the picker renders one card per row.
_HF_SEARCH_CAP = 20
_MANIFOLD_TAG = "drowse-manifold"
_ROOT_ARCHIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.drowse$", re.IGNORECASE)

ProgressCallback = Callable[[str], None]


# ---------------------------------------------------------------------- pull --


def _root_archives(siblings: Any) -> list[str]:
    names = [getattr(sibling, "rfilename", None) for sibling in siblings or []]
    return [
        name for name in names
        if isinstance(name, str) and _ROOT_ARCHIVE.fullmatch(name)
    ]


def resolve_manifold_repo(
    coord: str, revision: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve ``coord@revision`` to ``(commit_sha, archive_filename)``.

    The repo must be tagged ``drowse-manifold`` and carry exactly one root
    ``.drowse``; any revision (branch, tag, commit) pins to its commit.
    """
    label = f"{coord}@{revision}" if revision else coord
    try:
        info = _hf_api().model_info(coord, revision=revision)
    except Exception as exc:
        raise HFError(f"{label}: not found ({exc})") from exc
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise HFError(f"{label}: Hugging Face returned no commit sha")
    if _MANIFOLD_TAG not in (getattr(info, "tags", None) or []):
        raise HFError(f"{label}: repository is not tagged {_MANIFOLD_TAG!r}")
    archives = _root_archives(getattr(info, "siblings", None))
    if len(archives) != 1:
        raise HFError(
            f"{coord}@{sha}: a drowse manifold repo must carry exactly one root "
            f".drowse archive (found {len(archives)}); see `drowse pack push`"
        )
    return sha, archives[0]


def pull_manifold(
    coord: str,
    *,
    revision: Optional[str] = None,
    as_: Optional[str] = None,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Install ``coord`` (pinned to a commit) through the archive pipeline."""
    from drowse.base.staging import staging_area
    from drowse.store.archive import extract_drowse_archive
    from drowse.store.install import install_closure

    label = f"{coord}@{revision}" if revision else coord
    if on_progress is not None:
        on_progress(f"Resolving {label} on Hugging Face...")
    sha, filename = resolve_manifold_repo(coord, revision)
    if on_progress is not None:
        on_progress(f"Downloading {coord}@{sha[:12]}/{filename}...")
    try:
        archive = Path(_hf_hub_download(coord, filename, revision=sha))
    except Exception as exc:
        raise HFError(f"{coord}@{sha}: download failed ({exc})") from exc
    with staging_area() as area:
        if on_progress is not None:
            on_progress(f"Validating {filename}...")
        closure = extract_drowse_archive(archive, area)
        return install_closure(
            closure, as_=as_, force=force, source=f"hf://{coord}@{sha}",
            on_progress=on_progress,
        )


# ---------------------------------------------------------------------- push --


def _manifold_sidecar_stem_to_hf_coord(stem: str) -> Optional[str]:
    """Convert a fitted-tensor stem back to its base-model HF coord."""
    from drowse.base.paths import parse_tensor_filename, unsafe_model_id

    parsed = parse_tensor_filename(f"{stem}.safetensors")
    if parsed is None:
        return None
    safe_model, _variant = parsed
    return unsafe_model_id(safe_model)


def _domain_label(mf: ManifoldFolder) -> str:
    """Row label: the domain when one is declared, else the discover mode."""
    from drowse.store.manifolds import domain_label

    if mf.domain:
        return domain_label(mf.domain)
    if mf.fit_mode in {"pca", "spectral", "auto"}:
        return f"discover-{mf.fit_mode}"
    return "?"


def _search_row(mf: ManifoldFolder, tensor_stems: list[str]) -> dict[str, Any]:
    """The search-row fields, as carried in the card's ``drowse:`` block."""
    return {
        "name": mf.name,
        "description": mf.description,
        "tags": list(mf.tags),
        "node_count": len(mf.node_labels),
        "domain_label": _domain_label(mf),
        "fit_mode": mf.fit_mode,
        "tensor_models": sorted(tensor_stems),
    }


def _render_manifold_card(
    mf: ManifoldFolder, tensor_stems: list[str], coord: str, archive: str,
) -> str:
    """A model card: discovery frontmatter + a human-readable body.

    Frontmatter scalars are written as JSON, which is valid YAML flow syntax,
    so arbitrary descriptions round-trip through the Hub's card parser.
    """
    base_models = sorted({
        c for stem in tensor_stems
        if (c := _manifold_sidecar_stem_to_hf_coord(stem)) is not None
    })
    row = _search_row(mf, tensor_stems)
    fm = [
        "---",
        "library_name: drowse",
        "tags:",
        *(f"  - {tag}" for tag in (_MANIFOLD_TAG, "activation-steering", "steering-manifold")),
    ]
    if base_models:
        fm.append("base_model:")
        fm += [f"  - {json.dumps(model)}" for model in base_models]
        fm.append("base_model_relation: adapter")
    fm.append("drowse:")
    fm.append(f"  archive: {json.dumps(archive)}")
    fm += [f"  {key}: {json.dumps(value)}" for key, value in row.items()]
    fm.append("---")

    body: list[str] = [
        f"# {mf.name}",
        "",
        mf.description,
        "",
        f"**Domain:** `{row['domain_label']}`  |  **fit_mode:** `{mf.fit_mode}`  |  "
        f"**nodes:** {row['node_count']}",
        "",
        "## Install",
        "",
        "```bash",
        f"drowse pack install {coord}",
        "```",
        "",
        "## Nodes",
        "",
        ", ".join(f"`{label}`" for label in mf.node_labels),
        "",
    ]
    if tensor_stems:
        from drowse.base.paths import parse_tensor_filename

        body += [
            "## Fitted tensors",
            "",
            "| base model | variant |",
            "| --- | --- |",
        ]
        for stem in sorted(tensor_stems):
            base = _manifold_sidecar_stem_to_hf_coord(stem) or stem
            parsed = parse_tensor_filename(f"{stem}.safetensors")
            variant = "raw" if (parsed is None or parsed[1] is None) else parsed[1]
            body.append(f"| `{base}` | `{variant}` |")
        body.append("")

    body += ["---", "", "Generated by `drowse pack push`.", ""]
    return "\n".join(fm) + "\n\n" + "\n".join(body)


def _manifold_variant_matches(key: str, variant: str) -> bool:
    """``key`` is ``raw`` / ``sae-<rel>`` / ``from-<src>``; ``variant`` a filter."""
    if variant == "all":
        return True
    if variant == "raw":
        return key == "raw"
    if variant == "sae":
        return key.startswith("sae-")
    if variant == "from":
        return key.startswith("from-")
    return False


def push_manifold(
    folder: Path,
    coord: str,
    *,
    private: bool = False,
    model_scope: Optional[str] = None,
    variant: str = "raw",
    dry_run: bool = False,
) -> tuple[str, Optional[str]]:
    """Publish an installed manifold folder to ``coord`` as one Hub commit.

    Exports ``<name>.drowse`` (the corpus always; fitted pairs filtered by
    ``model_scope`` and ``variant`` — ``raw`` / ``sae`` / ``from`` / ``all``),
    then uploads it with the summary ``manifold.json`` and the model card,
    deleting any stale root archive or pre-archive loose layout in the same
    commit.  Returns ``(repo_url, commit_sha)``; ``sha`` is ``None`` on dry-run.
    """
    from drowse.base.paths import parse_tensor_filename, safe_model_id
    from drowse.store.archive import export_drowse_archive

    folder = Path(folder)
    scope_safe = safe_model_id(model_scope) if model_scope is not None else None

    def include(tensor_name: str) -> bool:
        parsed = parse_tensor_filename(tensor_name)
        if parsed is None:
            return False
        model, slug = parsed
        if scope_safe is not None and model != scope_safe:
            return False
        return _manifold_variant_matches("raw" if slug is None else slug, variant)

    staging = Path(tempfile.mkdtemp(prefix="drowse-manifold-push-"))
    try:
        archive = export_drowse_archive(
            folder.parent.name, folder.name,
            output=staging / f"{folder.name}.drowse",
            include_tensor=include,
        )
        # The summary manifest and the card's tensor list describe exactly
        # what was archived, read back from the archive's own entries.
        with zipfile.ZipFile(archive) as packed:
            pack = json.loads(packed.read("pack.json"))
            primary = pack["primary"]
            (staging / "manifold.json").write_bytes(
                packed.read(f"{primary}/manifold.json"),
            )
        stems = sorted(
            Path(entry).stem for entry in pack["files"]
            if entry.startswith(f"{primary}/") and entry.endswith(".safetensors")
            and "/" not in entry[len(primary) + 1:]
        )
        mf = ManifoldFolder.load(folder, verify_manifest=False)
        (staging / "README.md").write_text(
            _render_manifold_card(mf, stems, coord, archive.name), encoding="utf-8",
        )
        (staging / ".gitattributes").write_text(
            "*.drowse filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8",
        )

        repo_url = f"https://huggingface.co/{coord}"
        if dry_run:
            return (repo_url, None)

        api = _hf_api()
        api.create_repo(
            repo_id=coord, repo_type="model", private=private, exist_ok=True,
        )
        info = api.upload_folder(
            repo_id=coord,
            repo_type="model",
            folder_path=str(staging),
            commit_message=f"drowse pack push: {mf.name}",
            delete_patterns=["*.drowse", "*.safetensors", "*.json", "nodes/*"],
        )
        sha = getattr(info, "oid", None) or getattr(info, "commit_sha", None)
        return (repo_url, sha)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# -------------------------------------------------------------------- search --


def _card_block(card_data: Any) -> dict[str, Any]:
    if card_data is None:
        return {}
    data = card_data.to_dict() if hasattr(card_data, "to_dict") else card_data
    block = data.get("drowse") if isinstance(data, dict) else None
    return block if isinstance(block, dict) else {}


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def search_manifolds(query: Optional[str]) -> list[dict[str, Any]]:
    """Search the Hub for installable drowse manifold repos.

    One ``list_models`` call with card data: each row comes from the card's
    ``drowse:`` block, and a repo that does not carry exactly one root
    ``.drowse`` (so it could not be installed) is skipped.  At most
    ``_HF_SEARCH_CAP`` rows.
    """
    kwargs: dict[str, Any] = {
        "filter": _MANIFOLD_TAG, "limit": _HF_SEARCH_CAP,
        "full": True, "cardData": True,
    }
    if query:
        kwargs["search"] = query
    rows: list[dict[str, Any]] = []
    for model in list(_hf_api().list_models(**kwargs))[:_HF_SEARCH_CAP]:
        if len(_root_archives(getattr(model, "siblings", None))) != 1:
            continue
        namespace, _, repo = str(model.id).rpartition("/")
        block = _card_block(getattr(model, "card_data", None))
        node_count = block.get("node_count")
        rows.append({
            "name": str(block.get("name") or repo),
            "namespace": namespace,
            "description": str(block.get("description") or ""),
            "tags": _str_list(block.get("tags")),
            "node_count": node_count if isinstance(node_count, int) else 0,
            "domain_label": str(block.get("domain_label") or "?"),
            "fit_mode": str(block.get("fit_mode") or "?"),
            "tensor_models": _str_list(block.get("tensor_models")),
        })
    return rows
