"""Cross-runtime fixtures. Zero means unresolved, never a claimed flat topology."""
import csv
import math
from pathlib import Path

import pytest
import torch

from drowse.core.topology import (
    _count_persistent_loops, _detect_periodic_axes, _laplacian_eigen,
    _periodic_neighborhoods_preserved, _rips_h1_persistence,
)


with (Path(__file__).parents[1] / "browser-runtime/fixtures/topology-adversarial-v1.csv").open() as fixture:
    CASES = list(csv.DictReader(fixture))


def sample(case: dict[str, str]):
    name = case["shape"]
    n, m = int(case["around"]), int(case["across"])
    width, noise = float(case["width"]), float(case["noise"])
    rows = []
    truth = []
    for i in range(n):
        u = 2 * math.pi * i / n
        for j in range(m):
            v = 2 * math.pi * j / m
            z = width * (2 * j / (m - 1) - 1) if m > 1 else 0
            if name in ("circle", "ellipse"):
                row = [width * math.cos(u), math.sin(u)]
            elif name == "product":
                row = [math.cos(u), math.sin(u), width * math.cos(v), width * math.sin(v)]
            elif name == "donut":
                row = [(1 + width * math.cos(v)) * math.cos(u), (1 + width * math.cos(v)) * math.sin(u), width * math.sin(v)]
            elif name == "mobius":
                row = [(1 + z * math.cos(u / 2)) * math.cos(u), (1 + z * math.cos(u / 2)) * math.sin(u), z * math.sin(u / 2)]
            elif name == "cylinder":
                row = [math.cos(u), math.sin(u), z]
            elif name == "sphere":
                z = 1 - 2 * (i + 0.5) / n
                angle = i * math.pi * (3 - math.sqrt(5))
                radius = math.sqrt(1 - z * z)
                row = [radius * math.cos(angle), radius * math.sin(angle), z]
            elif name == "arc":
                row = [math.cos(1.5 * math.pi * i / (n - 1)), math.sin(1.5 * math.pi * i / (n - 1))]
            elif name == "grid":
                row = [i / (n - 1), j / (m - 1)]
            else:
                assert name == "line"
                row = [i / (n - 1), 0]
            rows.append([x + noise * math.sin((i * m + j) * 12.9898 + axis * 78.233) for axis, x in enumerate(row)])
            truth.append([u, v] if name == "product" else [u])
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(truth)


def detect(points: torch.Tensor):
    centered = points - points.mean(0)
    gram = centered @ centered.T
    distances = torch.cdist(centered.double(), centered.double()).float()
    try:
        values, vectors, _, _ = _laplacian_eigen(gram)
    except ValueError as error:
        assert "connected components" in str(error)
        return None
    result = _detect_periodic_axes(distances, vectors, max_dim=6, eigenvalues=values)
    return None if result is None else result[0]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(c.values()))
@pytest.mark.parametrize("variant", ["original", "permuted", "scaled"])
def test_periodic_chart_specificity_and_coordinate_fidelity(case: dict[str, str], variant: str):
    points, truth = sample(case)
    if variant == "permuted":
        permutation = torch.randperm(len(points), generator=torch.Generator().manual_seed(917))
        points, truth = points[permutation], truth[permutation]
    if variant == "scaled":
        points = points * 0.025
    result = detect(points)
    assert (0 if result is None else result.shape[1]) == int(case["expected_axes"])
    if result is not None:
        matches = []
        for axis in result.T:
            coherence = torch.stack([
                max(abs(torch.exp(1j * (axis - target)).mean()), abs(torch.exp(1j * (axis + target)).mean()))
                for target in truth.T
            ])
            assert float(coherence.max()) > 0.95
            matches.append(int(coherence.argmax()))
        assert len(set(matches)) == result.shape[1]


@pytest.mark.parametrize("seed", range(50))
def test_random_sphere_is_not_mistaken_for_a_torus(seed: int):
    points = torch.randn(80, 3, generator=torch.Generator().manual_seed(seed))
    points /= points.norm(dim=1, keepdim=True)
    assert detect(points) is None


def test_duplicate_samples_do_not_validate_a_collapsed_chart():
    points, truth = sample(CASES[0])
    points[1] = points[0]
    distances = torch.cdist(points.double(), points.double()).float()
    assert not _periodic_neighborhoods_preserved(distances, truth)
    assert detect(points) is None


def test_far_outlier_does_not_invent_periodic_structure():
    points, _ = sample(CASES[0])
    assert detect(torch.cat((points, torch.tensor([[50.0, 0.0]])))) is None


@pytest.mark.parametrize("bad", [torch.ones(2, 3), torch.full((4, 4), math.nan), -torch.ones(4, 4), torch.eye(4)])
def test_invalid_distance_matrix_is_rejected(bad: torch.Tensor):
    with pytest.raises(ValueError):
        _count_persistent_loops(bad)


def test_budget_exhaustion_and_invalid_thresholds_are_explicit():
    distances = torch.ones(6, 6) - torch.eye(6)
    with pytest.raises(ValueError, match="budget exceeded"):
        _rips_h1_persistence(distances, 2, max_triangles=1)
    for fraction in [math.nan, math.inf, -1]:
        with pytest.raises(ValueError, match="fraction"):
            _count_persistent_loops(distances, persistence_frac=fraction)
    with pytest.raises(ValueError, match="128"):
        _count_persistent_loops(torch.zeros(129, 129))


def test_periodic_fit_must_improve_on_mean_only_predictions(monkeypatch: pytest.MonkeyPatch):
    from drowse.core import topology
    from tests.test_manifold_topology import _choose, _circle

    monkeypatch.setattr(topology, "_rbf_gcv_score", lambda *args, **kwargs: 1e20)
    choice = _choose(_circle(40))
    candidate = next(c for c in choice.candidates if c.name == "torus-T1")
    assert not candidate.viable
    assert "mean-only" in candidate.reason
    assert choice.winner_name == "flat-pca"
