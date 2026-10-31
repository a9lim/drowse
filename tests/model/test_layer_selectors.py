"""The one layer-selector parser and its two resolvers (``drowse.model.layers``)."""
from __future__ import annotations

import pytest

from drowse.model.layers import (
    parse_layer_selector,
    resolve_layer_selector,
    resolve_model_layers,
)


@pytest.mark.parametrize(("text", "expected"), [
    (None, None),
    ("  ", None),
    ("3,7, 11", [3, 7, 11]),
    ("Workspace", "workspace"),
    ("all", "all"),
    ("sample", "sample"),
])
def test_parse_layer_selector(text: str | None, expected: object) -> None:
    assert parse_layer_selector(text) == expected


@pytest.mark.parametrize("text", ["3,x", ",,"])
def test_parse_layer_selector_rejects_malformed_text(text: str) -> None:
    with pytest.raises(ValueError):
        parse_layer_selector(text)


def test_resolve_layer_selector_draws_from_the_available_pool() -> None:
    fitted = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    assert resolve_layer_selector(None, fitted, n_layers=41) == fitted
    assert resolve_layer_selector("workspace", fitted, n_layers=41) == [16, 20, 24, 28, 32, 36]
    assert len(resolve_layer_selector("sample", fitted, n_layers=41)) == 9
    assert resolve_layer_selector([8, 4], fitted, n_layers=41) == [8, 4]
    with pytest.raises(ValueError, match="unknown layer mode"):
        resolve_layer_selector("sample", fitted, n_layers=41, allow_sample=False)


@pytest.mark.parametrize(("selector", "expected"), [
    (None, list(range(10))),
    ("all", list(range(10))),
    ("workspace", [4, 5, 6, 7, 8]),
    ("WORKSPACE", [4, 5, 6, 7, 8]),
    ("7, 3,3", [3, 7]),
    ([9, 2, 2], [2, 9]),
])
def test_resolve_model_layers_is_the_sorted_unique_fit_roster(
    selector: object, expected: list[int],
) -> None:
    assert resolve_model_layers(selector, 10) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(("selector", "message"), [
    ("", "at least one layer"),
    ([], "at least one layer"),
    ("sample", "unknown layer mode"),
    ("3,x", "malformed layers"),
    ([10], r"\[0, 10\)"),
    ([-1, 2], r"\[0, 10\)"),
])
def test_resolve_model_layers_rejects(selector: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_model_layers(selector, 10)  # type: ignore[arg-type]


def test_fit_and_preflight_share_the_resolver() -> None:
    """The fit and its weight-free preflight must resolve the same roster."""
    from drowse.fit import pipeline, preflight

    assert pipeline.resolve_model_layers is resolve_model_layers
    assert preflight.resolve_model_layers is resolve_model_layers
