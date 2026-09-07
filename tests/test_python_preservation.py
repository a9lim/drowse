from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from drowse import LoomTree, Profile
from drowse.core.capture import folded_directions
from drowse.io.hf_manifolds import install_manifold
from drowse.io.manifold_tensors import load_manifold
from drowse.io.manifolds import ManifoldFolder
from drowse.io.manifold_folder import ManifoldFormatError, validate_manifold_sidecar_payload
from tests._python_contract import python_contract


FIXTURES = Path(__file__).parent / "fixtures" / "saklas_v5_3"


def test_python_library_does_not_import_its_frontends() -> None:
    subprocess.run([
        sys.executable, "-c",
        "import drowse, sys; "
        "[getattr(drowse, name) for name in drowse.__all__]; "
        "assert not any(name.startswith(('drowse.server', 'drowse.web', 'fastapi')) "
        "for name in sys.modules)",
    ], check=True)


def test_original_python_api_cli_and_routes_remain_available() -> None:
    expected = json.loads((FIXTURES / "contract.json").read_text())
    current = python_contract()
    assert set(expected["exports"]) <= set(current["exports"])
    assert set(expected["properties"]) <= set(current["properties"])
    assert set(expected["routes"]) <= set(current["routes"])
    for name, parameters in expected["callables"].items():
        actual = current["callables"][name]
        actual_parameters = {parameter["name"]: parameter for parameter in actual}
        for parameter in parameters:
            if not parameter["required"]:
                assert actual_parameters[parameter["name"]]["default"] == parameter["default"], name
        signature = inspect.Signature([
            inspect.Parameter(
                parameter["name"], getattr(inspect.Parameter, parameter["kind"]),
                default=inspect.Parameter.empty if parameter["required"] else None,
            )
            for parameter in actual
        ])
        for all_arguments in (False, True):
            for prefer_keywords in (False, True):
                args: list[None] = []
                kwargs: dict[str, None] = {}
                for parameter in parameters:
                    if not all_arguments and not parameter["required"]:
                        continue
                    kind = parameter["kind"]
                    if kind == "POSITIONAL_ONLY" or (kind == "POSITIONAL_OR_KEYWORD" and not prefer_keywords):
                        args.append(None)
                    elif kind in {"POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"}:
                        kwargs[parameter["name"]] = None
                try:
                    signature.bind(*args, **kwargs)
                except TypeError as error:
                    pytest.fail(f"Original call to {name} no longer binds: {error}")
    for command, actions in expected["cli"].items():
        for action in actions:
            actual_action = next(
                candidate for candidate in current["cli"][command]
                if candidate["dest"] == action["dest"] and set(action["options"]) <= set(candidate["options"])
            )
            assert actual_action["required"] == action["required"], command
            assert actual_action["nargs"] == action["nargs"], command
            if action["choices"] is not None:
                assert set(action["choices"]) <= set(actual_action["choices"]), command


def test_original_profile_is_readable(tmp_path: Path) -> None:
    for suffix in (".safetensors", ".json"):
        shutil.copy2(FIXTURES / f"legacy_profile{suffix}", tmp_path)
    profile = Profile.load(tmp_path / "legacy_profile.safetensors")
    assert torch.equal(profile[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_original_conversation_is_readable() -> None:
    tree = LoomTree.load(FIXTURES / "legacy_tree.json")
    assert tree.messages_for() == [
        {"role": "user", "content": "keep this Saklas conversation"},
    ]


def test_original_baked_manifold_preserves_its_direction(tmp_path: Path) -> None:
    folder = tmp_path / "legacy_baked"
    shutil.copytree(FIXTURES / "legacy_baked", folder)
    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    manifest = ManifoldFolder.load(folder)
    assert manifest.fit_mode == "baked"
    tensor = next(folder.glob("*.safetensors"))
    manifold = load_manifold(tensor)
    assert torch.allclose(folded_directions(manifold)[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    for name, data in before.items():
        assert (folder / name).read_bytes() == data


def test_original_baked_manifold_can_be_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "home"))
    installed = install_manifold(str(FIXTURES / "legacy_baked"), "local/imported")
    folder = ManifoldFolder.load(installed)
    assert folder.name == "imported"
    assert folder.tensor_models()
    manifold = load_manifold(next(installed.glob("*.safetensors")))
    assert torch.allclose(folded_directions(manifold)[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.parametrize("damage", ["missing_field", "wrong_version", "conflicting_brand", "current_brand"])
def test_original_baked_adapter_does_not_accept_damaged_schema(damage: str) -> None:
    sidecar = next(path for path in (FIXTURES / "legacy_baked").glob("*.json") if path.name != "manifold.json")
    payload = json.loads(sidecar.read_text())
    if damage == "missing_field":
        del payload["node_count"]
    elif damage == "wrong_version":
        payload["format_version"] -= 1
    elif damage == "conflicting_brand":
        payload["drowse_version"] = "conflict"
    else:
        payload["drowse_version"] = payload.pop("saklas_version")
    with pytest.raises(ManifoldFormatError):
        validate_manifold_sidecar_payload(payload)
