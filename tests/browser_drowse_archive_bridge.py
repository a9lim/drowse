from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_fixture(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _summary(folder_path: Path) -> dict[str, Any]:
    from drowse.io.manifold_folder import ManifoldFolder

    folder = ManifoldFolder.load(folder_path)
    return {
        "primary": f"manifolds/{folder_path.parent.name}/{folder.name}",
        "template": None,
        "name": folder.name,
        "fit_mode": folder.fit_mode,
        "node_labels": folder.node_labels,
        "node_groups": dict(folder.node_groups()),
        "hyperparams": folder.hyperparams,
        "source": folder.source,
        "tags": folder.tags,
        "template_ref": folder.template_ref,
    }


def _export(
    fixture: dict[str, Any], output: Path,
) -> dict[str, Any]:
    from drowse.io.manifold_authoring import create_discover_manifold_folder
    from drowse.io.drowse_archive import export_drowse_archive

    nodes = fixture["nodes"]
    folder = create_discover_manifold_folder(
        fixture["namespace"],
        fixture["name"],
        fixture["description"],
        fit_mode=fixture["fitMode"],
        node_corpora={node["label"]: node["corpus"] for node in nodes},
        hyperparams=fixture["hyperparams"],
        node_roles={node["label"]: node["role"] for node in nodes},
        node_kinds={node["label"]: node["kind"] for node in nodes},
    )
    export_drowse_archive(fixture["namespace"], fixture["name"], output=output)
    return _summary(folder)


def _install(archive: Path) -> dict[str, Any]:
    from drowse.io.drowse_archive import install_drowse_archive

    return _summary(install_drowse_archive(archive))


def main() -> None:
    if len(sys.argv) != 5 or sys.argv[1] not in {"export", "install"}:
        raise SystemExit(
            "usage: browser_drowse_archive_bridge.py export|install FIXTURE HOME ARCHIVE"
        )
    action, fixture_path, home_path, archive_path = sys.argv[1:]
    os.environ["DROWSE_HOME"] = home_path
    fixture = _load_fixture(fixture_path)
    if action == "export":
        result = _export(fixture, Path(archive_path))
    else:
        result = _install(Path(archive_path))
    sys.stdout.write(f"{json.dumps(result, sort_keys=True)}\n")


if __name__ == "__main__":
    main()
