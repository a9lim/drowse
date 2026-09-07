import json
from pathlib import Path

import pytest
import torch

from drowse.core.mahalanobis import LayerWhitener
from drowse.core.manifold import fit_affine_subspace, subspace_share
from drowse.core.topology import derive_pca_coords, neutral_layout_coord


FIXTURE = Path(__file__).parents[1] / "browser-runtime" / "fixtures" / "affine-fisher-fit-v1.json"


def test_browser_affine_fisher_fixture_matches_python_source_of_truth() -> None:
    fixture = json.loads(FIXTURE.read_text())
    columns = fixture["columns"]
    neutral = torch.tensor(fixture["neutral"], dtype=torch.float32).reshape(-1, columns)
    centroids = torch.tensor(fixture["centroids"], dtype=torch.float32).reshape(-1, columns)
    neutral_mean = neutral.mean(dim=0)
    whitener = LayerWhitener.from_neutral_activations(
        {0: neutral},
        {0: neutral_mean},
        ridge_scale=fixture["ridgeScale"],
    )
    centered = centroids - centroids.mean(dim=0, keepdim=True)
    gram = whitener.subspace_gram(0, centered)
    inverse_rows = whitener.apply_inv(0, centered)
    subspace, mu_coordinates, explained = fit_affine_subspace(
        centroids,
        neutral_mean=neutral_mean,
        n_components=fixture["maxComponents"],
        whitener=whitener,
        layer=0,
        whitened_gram=gram,
        whitened_rows=inverse_rows,
        orient_to=fixture["orientTo"],
    )
    expected = fixture["expected"]
    assert neutral_mean.tolist() == pytest.approx(expected["neutralMean"], abs=1e-6)
    assert centroids.mean(dim=0).tolist() == pytest.approx(expected["centroidMean"], abs=1e-6)
    assert subspace.mean.tolist() == pytest.approx(expected["mean"], abs=1e-6)
    assert subspace.basis.flatten().tolist() == pytest.approx(expected["basis"], abs=1e-6)
    assert subspace.node_coords is not None
    assert subspace.node_coords.flatten().tolist() == pytest.approx(
        expected["nodeCoordinates"], abs=1e-6,
    )
    assert mu_coordinates.flatten().tolist() == pytest.approx(expected["muCoordinates"], abs=1e-6)
    assert gram.flatten().tolist() == pytest.approx(expected["whitenedGram"], abs=1e-5)
    neutral_cross_gram = inverse_rows @ (neutral_mean - centroids.mean(dim=0))
    assert neutral_cross_gram.tolist() == pytest.approx(expected["neutralCrossGram"], abs=1e-6)
    layout, _diagnostics = derive_pca_coords(gram, max_dim=2, var_threshold=0.7)
    neutral_layout = neutral_layout_coord(layout, neutral_cross_gram)
    # fp32 eigensolver roundoff varies between Linux and macOS LAPACK builds.
    assert layout.flatten().tolist() == pytest.approx(expected["layoutCoordinates"], abs=5e-6)
    assert neutral_layout.tolist() == pytest.approx(expected["neutralLayoutCoordinate"], abs=5e-6)
    assert (layout - neutral_layout).flatten().tolist() == pytest.approx(
        expected["anchoredLayoutCoordinates"], abs=5e-6,
    )
    assert explained == pytest.approx(expected["explainedVariance"], abs=1e-6)
    assert subspace.affine_map is None
    assert subspace.select_axes([0, 1]).affine_map is None
    reordered = subspace.select_axes([1, 0])
    assert reordered.affine_map is not None
    assert reordered.affine_map.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    with pytest.raises(ValueError, match="unique indices"):
        subspace.select_axes([0, 0])
    assert subspace_share(
        mu_coordinates, subspace.basis, whitener=whitener, layer=0,
    ) == pytest.approx(expected["mahalanobisShare"], abs=1e-6)
    selected = subspace.select_axes([0])
    selected_mu = mu_coordinates[:, [0]]
    selected_expected = expected["axisZeroFinal"]
    assert selected.mean.tolist() == pytest.approx(selected_expected["mean"], abs=1e-6)
    assert selected.basis.flatten().tolist() == pytest.approx(selected_expected["basis"], abs=1e-6)
    assert selected.node_coords is not None
    assert selected.node_coords.flatten().tolist() == pytest.approx(
        selected_expected["nodeCoordinates"], abs=1e-6,
    )
    assert selected.affine_map is not None
    assert selected.affine_map.tolist() == [[1.0], [0.0]]
    selected.validate_structure(
        feature_space="raw",
        expected_node_count=int(centroids.shape[0]),
    )
    embedded = torch.tensor([[0.25, -0.75]], dtype=torch.float32)
    expected_world = 0.25 * selected.basis[0] + selected.mean
    assert selected.eval_at(embedded).squeeze(0).tolist() == pytest.approx(
        expected_world.tolist(), abs=1e-6,
    )
    assert selected_mu.flatten().tolist() == pytest.approx(
        selected_expected["muCoordinates"], abs=1e-6,
    )
    assert subspace_share(
        selected_mu, selected.basis, whitener=whitener, layer=0,
    ) == pytest.approx(selected_expected["mahalanobisShare"], abs=1e-6)
