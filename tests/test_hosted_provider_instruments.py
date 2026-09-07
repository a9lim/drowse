from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


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
