import json
from pathlib import Path

import pytest
import torch

from drowse.core.capture import compute_dls_axes


FIXTURE = Path(__file__).parents[1] / "browser-runtime" / "fixtures" / "dls-axes-v1.json"


def test_browser_dls_fixture_matches_python_source_of_truth() -> None:
    fixture = json.loads(FIXTURE.read_text())
    centroids = {
        layer["layer"]: torch.tensor(layer["nodeCoordinates"], dtype=torch.float32).reshape(
            layer["nodeCount"], layer["components"],
        )
        for layer in fixture["layers"]
    }
    bases = {
        layer["layer"]: torch.eye(layer["components"], dtype=torch.float32)
        for layer in fixture["layers"]
    }
    means = {
        layer["layer"]: torch.zeros(layer["components"], dtype=torch.float32)
        for layer in fixture["layers"]
    }
    actual = compute_dls_axes(centroids, bases, means)
    assert {str(layer): sorted(axes) for layer, axes in actual.items()} == fixture["expected"]

    positive = {
        layer: values.abs() + 1.0
        for layer, values in centroids.items()
    }
    with pytest.warns(UserWarning, match="no layers pass"):
        fallback = compute_dls_axes(positive, bases, means)
    assert {str(layer): sorted(axes) for layer, axes in fallback.items()} == fixture["allFailExpected"]
