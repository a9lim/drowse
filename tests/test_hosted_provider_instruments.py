import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _module():
    path = Path(__file__).parents[1] / "browser-runtime" / "import-provider-instruments.py"
    spec = spec_from_file_location("hosted_provider_instruments", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jlens_layer_selection_matches_browser_budget():
    module = _module()
    assert module.select_jlens_layers(list(range(25)), 1152) == [0, 3, 7, 10, 14, 17, 21, 24]
    assert module.select_jlens_layers(list(range(33)), 2560) == [0, 5, 9, 14, 18, 23, 27, 32]


def test_jlens_layer_selection_keeps_one_large_matrix():
    module = _module()
    assert module.select_jlens_layers(list(range(10)), 16384) == [9]


@pytest.mark.parametrize("counts", [
    {}, {"n_prompts": None}, {"n_prompts": 0}, {"n_prompts": -1},
    {"n_prompts": True}, {"n_prompts": 460.5}, {"n_prompts": "460"},
    {"prompts_fitted": False}, {"n_prompts": 460, "prompts_fitted": 278},
    {"n_prompts": 460, "prompts_fitted": 460.0},
])
def test_jlens_import_rejects_invented_or_inconsistent_prompt_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, counts: dict[str, object]):
    module = _module()
    checkpoint = {"J": {0: torch.eye(2)}, **counts}
    monkeypatch.setattr(module.torch, "load", lambda *_args, **_kwargs: checkpoint)
    args = SimpleNamespace(jlens_checkpoint=tmp_path / "source.pt", output=tmp_path / "output")
    with pytest.raises(SystemExit, match="consistent positive prompt count"):
        module.create_jlens_pack(args, {"hiddenSize": 2, "layerMap": [0, 1]})
    assert not args.output.exists()


@pytest.mark.parametrize("counts", [
    {"n_prompts": 460}, {"prompts_fitted": 278},
    {"n_prompts": 546, "prompts_fitted": 546},
])
def test_jlens_import_preserves_explicit_prompt_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, counts: dict[str, int]):
    module = _module()
    checkpoint = {"J": {0: torch.eye(2)}, **counts}

    def load_checkpoint(*_args: object, **kwargs: object):
        assert kwargs["weights_only"] is True
        return checkpoint

    monkeypatch.setattr(module.torch, "load", load_checkpoint)
    monkeypatch.setattr(module, "create_browser_vocabulary", lambda *_args: None)
    source = tmp_path / "source.pt"
    source.write_bytes(b"provider checkpoint fixture")
    args = SimpleNamespace(
        jlens_checkpoint=source, output=tmp_path / "output", runtime_fingerprint="a" * 64,
    )
    model = {"hiddenSize": 2, "layerMap": [0, 1], "sourceRevision": "b" * 40}
    module.create_jlens_pack(args, model)
    manifest = json.loads((args.output / "jlens-pack/packs/jlens/manifest.json").read_text())
    assert manifest["n_prompts"] == next(iter(counts.values()))
    assert manifest["usable_prompt_count"] == manifest["n_prompts"]
