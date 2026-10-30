"""Behavior-manifold naturalness eval — CPU only, synthetic + mock model."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.conftest import CharTokenizer, FakeLogitsModel
from drowse.geometry.domains import BoxAxis, BoxDomain
from drowse.fit.naturalness import (
    NaturalnessError,
    bhattacharyya_distance,
    evaluate,
    fit_behavior_manifold,
    node_behavior_centroid,
    preflight,
    response_distributions,
    to_hellinger,
    trajectory_naturalness,
)


def _curve_dists(k: int, v: int) -> torch.Tensor:
    """K probability distributions tracing a smooth circle in logit space."""
    dists = torch.zeros(k, v)
    for i in range(k):
        theta = 2.0 * math.pi * i / k
        logits = torch.zeros(v)
        logits[0] = 3.0 * math.cos(theta)
        logits[1] = 3.0 * math.sin(theta)
        dists[i] = torch.softmax(logits, dim=-1)
    return dists


def _loop_domain(k: int) -> tuple[BoxDomain, torch.Tensor, torch.Tensor]:
    """A 1-D periodic domain over K nodes; returns (domain, coords, params)."""
    domain = BoxDomain([BoxAxis("t", periodic=True, period=1.0)])
    coords = torch.tensor([[i / k] for i in range(k)])
    return domain, coords, domain.embed(coords)


# --------------------------------------------------------------- metrics ---

def test_to_hellinger_round_trip():
    p = torch.softmax(torch.randn(20), dim=-1)
    assert torch.allclose(to_hellinger(p) ** 2, p, atol=1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_bhattacharyya_identical_is_zero(seed: int):
    # Regression: fp32 roundoff made sum(sqrt(p*p)) exceed 1 on some draws,
    # yielding a negative distance.
    torch.manual_seed(seed)
    p = torch.softmax(torch.randn(32), dim=-1)
    d = bhattacharyya_distance(p, p).item()
    assert 0.0 <= d < 1e-5


def test_bhattacharyya_disjoint_is_large():
    v = 16
    p = torch.zeros(v)
    p[:4] = 0.25
    q = torch.zeros(v)
    q[4:8] = 0.25
    # Disjoint support -> BC = 0 -> distance saturates high.
    assert bhattacharyya_distance(p, q).item() > 10.0


def test_bhattacharyya_batched():
    p = torch.softmax(torch.randn(5, 12), dim=-1)
    q = torch.softmax(torch.randn(5, 12), dim=-1)
    d = bhattacharyya_distance(p, q)
    assert d.shape == (5,)
    assert torch.all(d >= 0)


# ----------------------------------------------------- behavior manifold ---

def test_fit_behavior_manifold_interpolates_nodes():
    dists = _curve_dists(8, 32)
    domain, coords, params = _loop_domain(8)
    behavior = fit_behavior_manifold(dists, params)
    # The fitted RBF passes through the Hellinger-mapped centroids.
    for i in range(8):
        got = behavior.eval_at(domain.embed(coords[i]))
        assert torch.allclose(got, to_hellinger(dists[i]), atol=1e-3)


def test_on_manifold_trajectory_scores_low():
    dists = _curve_dists(8, 32)
    domain, coords, params = _loop_domain(8)
    behavior = fit_behavior_manifold(dists, params)
    # A trajectory made of the node centroids sits exactly on the curve.
    on_manifold = trajectory_naturalness(dists, behavior, domain, coords)
    assert torch.all(on_manifold < 1e-2)


def test_off_manifold_trajectory_scores_higher():
    torch.manual_seed(0)
    dists = _curve_dists(8, 32)
    domain, coords, params = _loop_domain(8)
    behavior = fit_behavior_manifold(dists, params)
    on_manifold = trajectory_naturalness(dists, behavior, domain, coords).mean()
    # Random distributions are off the learned behavior manifold.
    random_traj = torch.softmax(torch.randn(8, 32), dim=-1)
    off_manifold = trajectory_naturalness(
        random_traj, behavior, domain, coords,
    ).mean()
    assert off_manifold > on_manifold
    assert off_manifold > 1e-2


# -------------------------------------------------------- mock-model path ---
#
# The behavior reads only care that the model returns a per-position
# ``.logits``; the shared ``FakeLogitsModel`` (conftest) supplies that wrapper,
# parametrized by a seeded-random ``logits_fn`` so the reads stay deterministic
# per token sequence.  ``CharTokenizer`` has no chat template, so rendering
# takes the raw ``system\nprompt\nresponse`` fallback (one token per char).


def _MockModel(vocab: int) -> FakeLogitsModel:
    def _logits(input_ids: torch.Tensor) -> torch.Tensor:
        seq = input_ids.shape[1]
        torch.manual_seed(int(input_ids.sum()))
        return torch.randn(1, seq, vocab)

    return FakeLogitsModel(_logits, with_parameters=False)


def _MockTok() -> CharTokenizer:
    return CharTokenizer(mod=13)


def test_response_distributions_cover_exactly_the_response_tokens():
    model = _MockModel(24)
    traj = response_distributions(
        model, _MockTok(), torch.device("cpu"), "a prompt", "steered",
    )
    # One distribution per response token (chars), predicted from the
    # position before it — prompt and system positions are not scored.
    assert traj.shape == (len("steered"), 24)
    assert torch.allclose(traj.sum(dim=-1), torch.ones(traj.shape[0]), atol=1e-4)


def test_response_distributions_are_conditioned_on_the_prompt():
    model = _MockModel(24)
    a = response_distributions(model, _MockTok(), torch.device("cpu"), "first", "same")
    b = response_distributions(model, _MockTok(), torch.device("cpu"), "second", "same")
    assert a.shape == b.shape
    assert not torch.allclose(a, b)


def test_node_behavior_centroid_is_distribution():
    centroid = node_behavior_centroid(
        _MockModel(24), _MockTok(), torch.device("cpu"),
        [("p1", "calm one"), ("p2", "calm two"), ("p3", "calm three")],
    )
    assert centroid.shape == (24,)
    assert centroid.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert torch.all(centroid >= 0)


def test_naturalness_end_to_end_mock():
    model = _MockModel(24)
    tok = _MockTok()
    groups = [
        [("p", "calm a"), ("q", "calm b")],
        [("p", "warm a"), ("q", "warm b")],
        [("p", "hot a"), ("q", "hot b")],
        [("p", "cold a"), ("q", "cold b")],
    ]
    centroids = torch.stack([
        node_behavior_centroid(model, tok, torch.device("cpu"), g) for g in groups
    ])
    domain, coords, params = _loop_domain(len(groups))
    behavior = fit_behavior_manifold(centroids, params)
    traj = response_distributions(
        model, tok, torch.device("cpu"), "p", "some generated text here",
    )
    scores = trajectory_naturalness(traj, behavior, domain, coords)
    assert scores.shape[0] == traj.shape[0]
    assert torch.all(scores >= 0)


# ------------------------------------------------------------- evaluate() ---


class _FakeSession:
    """The slice of ``DrowseSession`` :func:`evaluate` touches."""

    def __init__(self, fitted: Any = None) -> None:
        self.model = _MockModel(24)
        self.tokenizer = _MockTok()
        self.device = torch.device("cpu")
        self.fitted = fitted
        self.fit_calls: list[Path] = []
        self.generated: list[str | None] = []

    def fit(self, folder: Path) -> Any:
        self.fit_calls.append(Path(folder))
        return self.fitted

    def generate(self, prompt: str, *, steering: str | None, sampling: Any) -> Any:
        self.generated.append(steering)
        return SimpleNamespace(first=SimpleNamespace(text=f"reply under {steering}"))


def _discover_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nodes: int) -> Path:
    from drowse.store.manifold_authoring import create_discover_manifold_folder

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    return create_discover_manifold_folder(
        "local", "moods", "", fit_mode="auto",
        node_corpora={
            f"mood{i}": [f"response {i} one", f"response {i} two"]
            for i in range(nodes)
        },
    )


def test_evaluate_discover_manifold_uses_the_fitted_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: discover folders store no domain/coords, so the behavior
    chart must come from the per-model fitted manifold (this crashed with
    ``unknown manifold domain type None``)."""
    from drowse.geometry.domains import CustomDomain
    from drowse.geometry.manifold import Manifold

    folder = _discover_folder(tmp_path, monkeypatch, nodes=4)
    fitted = Manifold(
        name="moods", domain=CustomDomain(2),
        node_labels=[f"mood{i}" for i in range(4)],
        node_coords=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        layers={},
    )
    session = _FakeSession(fitted)
    report = evaluate(session, folder, "how are you?", "0.5 moods%mood1")  # type: ignore[arg-type]

    assert session.fit_calls == [folder]
    assert session.generated == ["0.5 moods%mood1"]
    (score,) = report.scores
    assert score.label == "manifold"
    assert len(score.per_step) == len("reply under 0.5 moods%mood1")
    assert all(d >= 0 for d in score.per_step)
    payload = report.to_dict()
    assert payload["results"][0]["mean_bhattacharyya"] == pytest.approx(score.mean)


def test_evaluate_rejects_a_two_node_manifold_before_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _discover_folder(tmp_path, monkeypatch, nodes=2)
    session = _FakeSession()
    with pytest.raises(NaturalnessError, match="at least 3"):
        evaluate(session, folder, "p", "0.5 mood0")  # type: ignore[arg-type]
    assert session.fit_calls == [] and session.generated == []


def test_evaluate_rejects_a_fit_whose_nodes_differ_from_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.geometry.domains import CustomDomain
    from drowse.geometry.manifold import Manifold

    folder = _discover_folder(tmp_path, monkeypatch, nodes=3)
    fitted = Manifold(
        name="moods", domain=CustomDomain(1), node_labels=["mood2", "mood1", "mood0"],
        node_coords=torch.tensor([[1.0], [0.0], [-1.0]]), layers={},
    )
    with pytest.raises(NaturalnessError, match="do not match"):
        evaluate(_FakeSession(fitted), folder, "p", "0.5 mood0")  # type: ignore[arg-type]


def test_evaluate_rejects_a_folder_that_is_not_a_manifold(tmp_path: Path) -> None:
    with pytest.raises(NaturalnessError, match="neither a manifold folder"):
        evaluate(_FakeSession(), tmp_path / "missing", "p", "0.5 x")  # type: ignore[arg-type]


def test_evaluate_compare_linear_requires_a_single_manifold_term_before_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _discover_folder(tmp_path, monkeypatch, nodes=4)
    session = _FakeSession()
    with pytest.raises(NaturalnessError, match="single manifold term"):
        evaluate(  # type: ignore[arg-type]
            session, folder, "p", "0.5 formal.casual", compare_linear=True,
        )
    assert session.fit_calls == []


def test_preflight_accepts_discover_manifolds_by_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _discover_folder(tmp_path, monkeypatch, nodes=3)
    resolved = preflight("local/moods", "0.5 moods%mood1", compare_linear=True)
    assert resolved.folder == folder and resolved.is_discover


def test_preflight_does_not_hash_fitted_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The behavior manifold reads only authoring geometry and corpus, so
    resolving the folder (before any model load) hashes no fitted payload."""
    import json

    from drowse.base import integrity
    from drowse.store.manifold_folder import ManifoldFolder
    from drowse.store.manifold_authoring import create_manifold_folder
    from drowse.store.manifold_schema import ManifoldSidecar

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "period": 1.0,
         "lo": 0.0, "hi": 1.0},
    ]}
    nodes = [
        {"label": label, "coords": [i / 2], "statements": [label]}
        for i, label in enumerate(("a", "b", "c"))
    ]
    folder, _ = create_manifold_folder("local", "behavior", "", domain, nodes)
    tensor = folder / "unrelated.safetensors"
    sidecar = folder / "unrelated.json"
    tensor.write_bytes(b"unrelated fitted payload")
    sidecar.write_text(json.dumps(ManifoldSidecar.build(
        name="behavior", method="manifold_pca",
        domain=domain, node_labels=["a", "b", "c"], feature_space="raw",
        fit_mode="authored",
    ).to_payload()))
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        tensor, sidecar,
    )
    integrity._FINGERPRINT_CACHE.clear()
    real_hash = integrity.hash_file
    hashed: list[Path] = []

    def track_hash(path: Path) -> str:
        hashed.append(Path(path))
        return real_hash(Path(path))

    monkeypatch.setattr(integrity, "hash_file", track_hash)
    assert preflight(folder, "0.5 behavior%a").folder == folder
    assert hashed == []
