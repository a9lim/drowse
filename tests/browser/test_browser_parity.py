"""The Python source of truth for the drowse.ai browser runtime's goldens.

Each test recomputes one golden under ``tests/fixtures/browser`` (and the
steering-expression accept/reject table beside it) with the Python engine and
compares.  drowse.ai replays the same files through its WebGPU runtime, so a
semantic change here means regenerating the golden in the same change.  The
golden files and their paths are part of that seam: never rename or move them.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import drowse.loom.tree as loom_module
from drowse.events import EventBus, LoomMutated
from drowse.fit.capture import compute_dls_axes
from drowse.generation.decode import GenerationConfig, _effective_topk, _sampler_candidates
from drowse.geometry.domains import BoxAxis, BoxDomain, CustomDomain, SphereDomain
from drowse.geometry.fold import project_profile
from drowse.geometry.inject import subspace_inject, synthesize_subspace
from drowse.geometry.rbf import RbfFitPlan, eval_rbf
from drowse.geometry.subspace import LayerSubspace, fit_affine_subspace, subspace_share
from drowse.geometry.topology import (
    SpectralDiagnostics,
    _count_persistent_loops,
    _detect_periodic_axes,
    _laplacian_eigen,
    _periodic_neighborhoods_preserved,
    _rips_h1_persistence,
    derive_pca_coords,
    derive_spectral_coords,
    neutral_layout_coord,
    select_topology,
)
from drowse.geometry.whitener import LayerWhitener
from drowse.lens.jlens import JacobianLens, aggregate_readout_from_probabilities, token_readout_stats_from_probabilities
from drowse.loom.recipe import CastMember, Recipe
from drowse.loom.tree import LoomTree
from drowse.readout.measurements import build_measurements
from drowse.readout.monitor import _depth_stats
from drowse.readout.types import DepthSummary, ProbeReading, ScalarReading
from drowse.sae.backend import LocalSaeBackend
from drowse.steering.grammar import SteeringExprError, parse_expr, referenced_selectors
from drowse.steering.hooks import _lower_affine_subspaces
from drowse.steering.triggers import TriggerContext
from tests.support.whitener import isotropic_whitener
from tests.support.sae import make_resident, sae_instrument

GOLDENS = Path(__file__).parent.parent / "fixtures" / "browser"


# ---------------------------------------------------------------------------
# Composed affine lowering
# ---------------------------------------------------------------------------


COMPOSED_LOWERING = json.loads(
    (
        GOLDENS
        / "affine-composed-lowering-v1.json"
    ).read_text()
)

@pytest.mark.parametrize("case", COMPOSED_LOWERING["cases"], ids=lambda case: case["name"])
def test_browser_composed_affine_lowering_matches_python(case: dict[str, object]) -> None:
    whitener, means = _whitener()
    pushes = []
    for term in cast(list[dict[str, Any]], case["terms"]):
        base = _profile(term["base"], folded=term["projection"] is not None)
        projection = term["projection"]
        if projection is not None:
            onto = _profile(projection[1:], folded=True)
            base = project_profile(base, onto, projection[0], whitener=whitener)
        basis = {}
        coordinates = {}
        for layer, direction in base.items():
            norm = float(direction.norm())
            if norm <= 1e-9:
                continue
            basis[layer] = (direction / norm).reshape(1, -1)
            coordinates[layer] = torch.tensor([norm], dtype=torch.float32)
        pushes.append((basis, coordinates, term["coefficient"]))

    synthesized = synthesize_subspace(pushes, [], means, whitener=whitener)
    lowered = _lower_affine_subspaces(synthesized)
    for expected in cast(list[dict[str, Any]], case["expectedLayers"]):
        layer = expected["layer"]
        actual = lowered[layer]
        world_target = actual.target @ actual.subspace.basis
        target = float(world_target.norm())
        basis = world_target / target
        delta = actual.eff_along * world_target
        assert basis.tolist() == pytest.approx(expected["basis"], abs=1e-6)
        assert actual.subspace.mean.tolist() == pytest.approx(expected["neutral"], abs=1e-7)
        assert target == pytest.approx(expected["target"], abs=1e-6)
        assert actual.eff_along == pytest.approx(expected["along"], abs=1e-6)
        assert delta.tolist() == pytest.approx(expected["delta"], abs=1e-5)

def _whitener() -> tuple[LayerWhitener, dict[int, torch.Tensor]]:
    centered = {}
    small_inverse = {}
    ridge = {}
    means = {}
    for row in COMPOSED_LOWERING["whiteners"]:
        layer = row["layer"]
        basis = torch.tensor(row["basis"], dtype=torch.float32)
        eigenvalues = torch.tensor(row["eigenvalues"], dtype=torch.float32)
        rank = len(eigenvalues)
        centered[layer] = torch.diag(torch.sqrt(rank * eigenvalues)) @ basis
        small_inverse[layer] = torch.diag(
            1 / (rank * (eigenvalues + row["ridge"]))
        )
        ridge[layer] = row["ridge"]
        means[layer] = torch.tensor(row["mean"], dtype=torch.float32)
    return LayerWhitener(centered, small_inverse, ridge, means), means

def _profile(selector: list[str], *, folded: bool) -> dict[int, torch.Tensor]:
    manifold_name, label = selector
    manifold = COMPOSED_LOWERING["manifolds"][manifold_name]
    label_index = manifold["labels"].index(label)
    result = {}
    for row in manifold["layers"]:
        basis = torch.tensor(row["basis"], dtype=torch.float32)
        magnitude = (
            row["mahalanobisShare"]
            if folded
            else row["nodeCoordinates"][label_index]
        )
        result[row["layer"]] = basis * magnitude
    return result


# ---------------------------------------------------------------------------
# Rank-one ablation lowering
# ---------------------------------------------------------------------------


ABLATION_LOWERING = json.loads(
    (
        GOLDENS
        / "affine-ablation-lowering-v1.json"
    ).read_text()
)

def test_browser_rank_one_ablation_matches_python() -> None:
    whitener, means = _whitener()
    directions = {
        row["layer"]: torch.tensor(row["basis"], dtype=torch.float32)
        * row["mahalanobisShare"]
        for row in COMPOSED_LOWERING["manifolds"]["demo"]["layers"]
    }
    synthesized = synthesize_subspace(
        [],
        [(directions, ABLATION_LOWERING["coefficient"])],
        means,
        whitener=whitener,
    )
    lowered = _lower_affine_subspaces(synthesized)

    for expected in ABLATION_LOWERING["layers"]:
        actual = lowered[expected["layer"]]
        assert actual.subspace.basis[0].tolist() == pytest.approx(expected["basis"])
        assert actual.subspace.mean.tolist() == pytest.approx(expected["neutral"])
        assert actual.target.tolist() == pytest.approx([0.0])
        collapse = actual.eff_along * actual.kappa
        assert collapse.tolist() == pytest.approx([ABLATION_LOWERING["coefficient"]])
        residual = torch.tensor(expected["input"], dtype=torch.float32)
        coordinate = actual.subspace.basis @ (residual - actual.subspace.mean)
        output = residual + actual.eff_along * (
            actual.target - actual.kappa * coordinate
        ) @ actual.subspace.basis
        assert output.tolist() == pytest.approx(expected["expected"])


# ---------------------------------------------------------------------------
# Affine rank-one lowering
# ---------------------------------------------------------------------------


AFFINE_LOWERING = json.loads(
    (
        GOLDENS
        / "affine-rank-one-lowering-v1.json"
    ).read_text()
)

def test_browser_affine_lowering_fixture_matches_python() -> None:
    layers = AFFINE_LOWERING["layers"]
    basis = {
        row["layer"]: torch.tensor([row["basis"]], dtype=torch.float32)
        for row in layers
    }
    coordinates = {
        row["layer"]: torch.tensor(
            [row["nodeCoordinates"][row["selectedNode"]]], dtype=torch.float32
        )
        for row in layers
    }
    means = {
        row["layer"]: torch.tensor(row["neutral"], dtype=torch.float32)
        for row in layers
    }
    whitener = isotropic_whitener(means, len(layers[0]["basis"]))
    synthesized = synthesize_subspace(
        [(basis, coordinates, AFFINE_LOWERING["coefficient"])],
        [],
        means,
        whitener=whitener,
    )
    lowered = _lower_affine_subspaces(synthesized)

    for row in layers:
        layer = lowered[row["layer"]]
        assert layer.target.tolist() == pytest.approx([row["expectedTarget"]])
        assert layer.eff_along == pytest.approx(row["expectedAlong"])
        assert layer.kappa.tolist() == [0.0]


# ---------------------------------------------------------------------------
# Affine Fisher fit
# ---------------------------------------------------------------------------


AFFINE_FISHER = GOLDENS / "affine-fisher-fit-v1.json"

def test_browser_affine_fisher_fixture_matches_python_source_of_truth() -> None:
    fixture = json.loads(AFFINE_FISHER.read_text())
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
    fit = fit_affine_subspace(
        centroids,
        neutral_mean=neutral_mean,
        n_components=fixture["maxComponents"],
        whitener=whitener,
        layer=0,
        whitened_gram=gram,
        whitened_rows=inverse_rows,
        orient_to=fixture["orientTo"],
    )
    subspace, mu_coordinates = fit.subspace, fit.mu_coords
    explained = fit.explained_variance
    expected = fixture["expected"]
    assert neutral_mean.tolist() == pytest.approx(expected["neutralMean"], rel=1e-5, abs=1e-6)
    assert centroids.mean(dim=0).tolist() == pytest.approx(expected["centroidMean"], rel=1e-5, abs=1e-6)
    assert subspace.mean.tolist() == pytest.approx(expected["mean"], rel=1e-5, abs=1e-6)
    assert subspace.basis.flatten().tolist() == pytest.approx(expected["basis"], rel=1e-5, abs=1e-6)
    assert subspace.node_coords is not None
    # Eigensolve outputs carry fp32 BLAS-order noise (~2e-6 on macOS arm64).
    assert subspace.node_coords.flatten().tolist() == pytest.approx(
        expected["nodeCoordinates"], rel=1e-5, abs=1e-6,
    )
    assert mu_coordinates.flatten().tolist() == pytest.approx(expected["muCoordinates"], rel=1e-5, abs=1e-6)
    assert gram.flatten().tolist() == pytest.approx(expected["whitenedGram"], abs=1e-5)
    neutral_cross_gram = inverse_rows @ (neutral_mean - centroids.mean(dim=0))
    assert neutral_cross_gram.tolist() == pytest.approx(expected["neutralCrossGram"], rel=1e-5, abs=1e-6)
    layout, _diagnostics = derive_pca_coords(gram, max_dim=2, var_threshold=0.7)
    neutral_layout = neutral_layout_coord(layout, neutral_cross_gram)
    # fp32 eigensolver roundoff varies between Linux and macOS LAPACK builds.
    assert layout.flatten().tolist() == pytest.approx(expected["layoutCoordinates"], abs=5e-6)
    assert neutral_layout.tolist() == pytest.approx(expected["neutralLayoutCoordinate"], abs=5e-6)
    assert (layout - neutral_layout).flatten().tolist() == pytest.approx(
        expected["anchoredLayoutCoordinates"], abs=5e-6,
    )
    assert explained == pytest.approx(expected["explainedVariance"], rel=1e-5, abs=1e-6)
    assert subspace.affine_map is None
    assert subspace.select_axes([0, 1]).affine_map is None
    reordered = subspace.select_axes([1, 0])
    assert reordered.affine_map is not None
    assert reordered.affine_map.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    with pytest.raises(ValueError, match="unique indices"):
        subspace.select_axes([0, 0])
    assert subspace_share(
        mu_coordinates, subspace.basis, whitener=whitener, layer=0,
    ) == pytest.approx(expected["mahalanobisShare"], rel=1e-5, abs=1e-6)
    selected = subspace.select_axes([0])
    selected_mu = mu_coordinates[:, [0]]
    selected_expected = expected["axisZeroFinal"]
    assert selected.mean.tolist() == pytest.approx(selected_expected["mean"], rel=1e-5, abs=1e-6)
    assert selected.basis.flatten().tolist() == pytest.approx(selected_expected["basis"], rel=1e-5, abs=1e-6)
    assert selected.node_coords is not None
    assert selected.node_coords.flatten().tolist() == pytest.approx(
        selected_expected["nodeCoordinates"], rel=1e-5, abs=1e-6,
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
        expected_world.tolist(), rel=1e-5, abs=1e-6,
    )
    assert selected_mu.flatten().tolist() == pytest.approx(
        selected_expected["muCoordinates"], rel=1e-5, abs=1e-6,
    )
    assert subspace_share(
        selected_mu, selected.basis, whitener=whitener, layer=0,
    ) == pytest.approx(selected_expected["mahalanobisShare"], rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# Discriminative layer selection
# ---------------------------------------------------------------------------


DLS_AXES = GOLDENS / "dls-axes-v1.json"

def test_browser_dls_fixture_matches_python_source_of_truth() -> None:
    fixture = json.loads(DLS_AXES.read_text())
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


# ---------------------------------------------------------------------------
# Post-block rank-one hook
# ---------------------------------------------------------------------------


RANK_ONE_HOOK = GOLDENS / "post-block-rank-one-v2.json"

@pytest.mark.parametrize("case_index", range(len(json.loads(RANK_ONE_HOOK.read_text())["cases"])))
def test_browser_rank_one_hook_fixture_matches_python_fp32(case_index: int) -> None:
    fixture = json.loads(RANK_ONE_HOOK.read_text())
    case = fixture["cases"][case_index]
    residual = torch.tensor(case["residual"], dtype=torch.float32)
    basis = torch.tensor(case["basis"], dtype=torch.float32)
    neutral = torch.tensor(case["neutral"], dtype=torch.float32)
    from drowse.geometry.domains import CustomDomain
    from drowse.geometry.inject import subspace_inject
    from drowse.geometry.subspace import LayerSubspace

    coordinate = torch.dot(residual - neutral, basis)
    result, _ = subspace_inject(
        residual,
        LayerSubspace.affine(neutral, basis.reshape(1, -1)),
        CustomDomain(1),
        torch.tensor([case["target"]], dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        case["along"],
        0.0,
        kappa=torch.tensor([case["collapse"]], dtype=torch.float32),
    )
    probe = torch.dot(
        result - torch.tensor(case["probeNeutral"], dtype=torch.float32),
        torch.tensor(case["probeBasis"], dtype=torch.float32),
    )

    assert coordinate.item() == pytest.approx(case["expectedCoordinate"], abs=1e-7)
    assert result.tolist() == pytest.approx(case["expectedResidual"], abs=1e-7)
    assert probe.item() == pytest.approx(case["expectedProbe"], abs=1e-7)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


SAMPLING = (
    GOLDENS
    / "sampling-parity-v1.json"
)

@pytest.mark.parametrize("case", json.loads(SAMPLING.read_text())["cases"])
def test_browser_sampling_fixture_matches_python_source_of_truth(
    case: dict[str, Any],
) -> None:
    logits = torch.tensor([case["logits"]], dtype=torch.float32)
    config = GenerationConfig(
        temperature=case["temperature"],
        top_p=case["topP"],
        top_k=case["topK"] or None,
    )
    token_ids, probabilities = _sampler_candidates(
        logits,
        config,
        _effective_topk(config, logits.shape[-1]),
    )
    support = probabilities > 0
    support_ids = token_ids[support].tolist()
    support_probabilities = probabilities[support].tolist()
    expected = case["expected"]
    assert support_ids == expected["supportTokenIds"]
    assert support_probabilities == pytest.approx(
        expected["supportProbabilities"], rel=1e-6, abs=1e-7,
    )

    entropy = -sum(probability * math.log(probability) for probability in support_probabilities)
    assert entropy == pytest.approx(expected["entropyNats"], rel=1e-6, abs=1e-7)
    assert math.exp(entropy) == pytest.approx(expected["perplexity"], rel=1e-6)
    for sample in expected["samples"]:
        cumulative = 0.0
        sampled_id = support_ids[-1]
        sampled_probability = support_probabilities[-1]
        for token_id, probability in zip(
            support_ids, support_probabilities, strict=True,
        ):
            cumulative += probability
            if sample["uniform"] < cumulative:
                sampled_id = token_id
                sampled_probability = probability
                break
        assert sampled_id == sample["tokenId"]
        assert math.log(sampled_probability) == pytest.approx(
            sample["logprob"], rel=1e-6, abs=1e-7,
        )


# ---------------------------------------------------------------------------
# Steering expression accept/reject table
# ---------------------------------------------------------------------------


STEERING_EXPRESSIONS = json.loads(
    (GOLDENS.parent / "browser_steering_expression.json").read_text()
)

@pytest.mark.parametrize("case", STEERING_EXPRESSIONS["accepted"])
def test_browser_steering_fixture_accepts_python_expressions(
    case: dict[str, object],
) -> None:
    expression = case["expression"]
    assert isinstance(expression, str)
    parse_expr(expression)
    assert [list(item) for item in referenced_selectors(expression)] == case["selectors"]

@pytest.mark.parametrize("expression", STEERING_EXPRESSIONS["rejected"])
def test_browser_steering_fixture_rejects_python_expressions(expression: str) -> None:
    with pytest.raises(SteeringExprError):
        parse_expr(expression)


# ---------------------------------------------------------------------------
# Loom transcript
# ---------------------------------------------------------------------------


LOOM_TRANSCRIPT = (
    GOLDENS
    / "loom-operation-transcript-v1.json"
)

def _event_dict(event: LoomMutated) -> dict[str, object]:
    return {
        "op": event.op,
        "rev": event.rev,
        "added": list(event.added),
        "removed": list(event.removed),
        "updated": list(event.updated),
        "active_node_id": event.active_node_id,
    }

def _canonical_tree(tree: LoomTree) -> dict[str, object]:
    return {
        "rev": tree.rev,
        "root_id": tree.root_id,
        "active_node_id": tree.active_node_id,
        "nodes": [
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "role": node.role,
                "text": node.text,
                "starred": node.starred,
                "notes": node.notes,
                "edit_count": node.edit_count,
            }
            for node in tree.nodes.values()
        ],
        "children_of": {
            node_id: list(children)
            for node_id, children in tree.children_of.items()
        },
        "cast": {
            label: member.to_dict()
            for label, member in tree.cast.items()
        },
    }

def test_shared_loom_operation_transcript_matches_python_source_of_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = json.loads(LOOM_TRANSCRIPT.read_text())
    generated_ids = iter(
        operation["createdId"]
        for operation in fixture["operations"]
        if "createdId" in operation
    )
    monkeypatch.setattr(loom_module, "_ulid", lambda: next(generated_ids))

    events = EventBus()
    mutations: list[LoomMutated] = []
    events.subscribe(
        lambda event: mutations.append(event)
        if isinstance(event, LoomMutated)
        else None
    )
    tree = LoomTree.from_dict(fixture["initial"], events=events)
    preserved_measurements = None

    for operation in fixture["operations"]:
        kind = operation["op"]
        if kind == "edit":
            tree.edit(operation["node"], operation["text"])
        elif kind == "branch":
            created = tree.branch(
                operation["node"],
                operation["text"],
                role=operation["role"],
            )
            assert created == operation["createdId"]
        elif kind == "star":
            tree.star(operation["node"], operation["value"])
        elif kind == "note":
            tree.annotate(operation["node"], operation["text"])
        elif kind == "navigate":
            tree.navigate(operation["node"])
        elif kind == "castPut":
            tree.set_cast_member(
                operation["label"],
                CastMember(
                    recipe=Recipe.from_parts(
                        steering=operation["steering"],
                        thinking=operation["thinking"],
                        seed=operation["seed"],
                    ),
                    notes=operation["notes"],
                ),
            )
        elif kind == "delete":
            assert tree.delete_subtree(operation["node"]) == 1
        elif kind == "reset":
            tokens = tree.get("assistant-1").tokens
            assert tokens is not None
            preserved_measurements = tokens[0][
                "measurements"
            ]
            tree.reset()
        else:  # pragma: no cover - fixture schema is intentionally closed
            raise AssertionError(f"unknown shared loom operation {kind!r}")

    assert [_event_dict(event) for event in mutations] == fixture["expected"][
        "events"
    ]
    assert preserved_measurements == fixture["expected"][
        "preservedMeasurementsBeforeReset"
    ]
    assert _canonical_tree(tree) == fixture["expected"]["final"]


# ---------------------------------------------------------------------------
# Measurement envelope
# ---------------------------------------------------------------------------


MEASUREMENTS = (
    GOLDENS
    / "measurement-envelope-v1.json"
)

def test_browser_measurement_fixture_matches_python_source_of_truth() -> None:
    fixture = json.loads(MEASUREMENTS.read_text())
    geometry = fixture["geometry"]
    rows = geometry["rows"]
    weights = geometry["shareWeights"]
    total_weight = sum(weights)

    def weighted(key: str) -> float:
        return sum(
            weight * row[key] for weight, row in zip(weights, rows)
        ) / total_weight

    coords = tuple(
        sum(
            weight * row["coords"][axis]
            for weight, row in zip(weights, rows)
        ) / total_weight
        for axis in range(len(rows[0]["coords"]))
    )
    distances = [
        sum(
            weight * row["distances"][candidate]
            for weight, row in zip(weights, rows)
        ) / total_weight
        for candidate in range(len(geometry["labels"]))
    ]
    logits = [
        -(distance**2) / (2 * bandwidth**2) + bias
        for distance, bandwidth, bias in zip(
            distances,
            geometry["assignBandwidth"],
            geometry["assignLogVolumeBias"],
        )
    ]
    normalizer = sum(math.exp(value - max(logits)) for value in logits)
    assignment = [
        math.exp(value - max(logits)) / normalizer for value in logits
    ]
    nearest = sorted(
        zip(geometry["labels"], distances), key=lambda item: item[1]
    )
    ranked_assignment = sorted(
        zip(geometry["labels"], assignment),
        key=lambda item: item[1],
        reverse=True,
    )
    model_layer_count = fixture["modelLayerCount"]
    depths = [layer / (model_layer_count - 1) for layer in fixture["layerMap"]]
    layers = fixture["layerMap"]
    geometry_depth_center, geometry_depth_spread = _depth_stats(
        {
            layer: tuple(row["coords"])
            for layer, row in zip(layers, rows)
        },
        dict(zip(layers, weights)),
        model_layer_count - 1,
    )
    geometry_reading = ProbeReading(
        fraction=weighted("fraction"),
        nearest=[
            (label, distance / geometry["labelScale"])
            for label, distance in nearest
        ],
        coords=coords,
        residual=weighted("residual"),
        fraction_per_layer={
            layer: row["fraction"] for layer, row in zip(layers, rows)
        },
        coords_per_layer={
            layer: tuple(row["coords"]) for layer, row in zip(layers, rows)
        },
        residual_per_layer={
            layer: row["residual"] for layer, row in zip(layers, rows)
        },
        assignment=ranked_assignment,
        membership=weighted("membership"),
        depth_com=geometry_depth_center,
        depth_spread=geometry_depth_spread,
        subspace_coords_per_layer={
            layer: tuple(row["subspaceCoords"])
            for layer, row in zip(layers, rows)
        },
    )

    lens = fixture["lens"]
    lens_probabilities = torch.tensor(
        lens["probabilities"], dtype=torch.float32,
    ).reshape(len(layers), 1)
    lens_token_stats = token_readout_stats_from_probabilities(
        lens_probabilities, depths, [0],
    )[0]
    lens_strength, lens_center, lens_spread, lens_per_layer = lens_token_stats
    lens_aggregate = aggregate_readout_from_probabilities(
        lens_probabilities, depths, top_k=1,
    )[0]
    assert lens_aggregate[0] == 0
    assert lens_aggregate[1:] == (
        lens_strength, lens_center, lens_spread,
    )
    lens_reading = ScalarReading(
        value=lens_strength,
        unit="mean_token_probability",
        per_layer=dict(zip(layers, lens_per_layer)),
        depth=DepthSummary(
            (lens_center,),
            (lens_spread,),
            "readout_probability_mass",
        ),
    )

    sae = fixture["sae"]
    sae_session = SimpleNamespace(
        layers=[object() for _ in range(model_layer_count)],
    )
    sae_inst = sae_instrument(sae_session)
    make_resident(
        sae_inst.runtime,
        layer=sae["layer"],
        width=sae["featureCount"],
        feature_meta={
            str(sae["featureId"]): {"label": sae["label"], "max_act": sae["maxAct"]},
        },
    )
    sae_inst.probes[sae["name"]] = {
        "feature_id": sae["featureId"],
        "layer": sae["layer"],
        "label": sae["label"],
        "max_act": sae["maxAct"],
    }
    sae_activations = torch.zeros(sae["featureCount"], dtype=torch.float32)
    sae_activations[sae["featureId"]] = sae["activation"]
    sae_reading = sae_inst.score_probes_from_activations(
        sae_activations,
    )[sae["name"]]
    assert sae_reading.value == sae["value"]
    per_layer_scores = {
        str(layer): {
            geometry["name"]: row["coords"][0],
            lens["name"]: probability,
            **(
                {sae["name"]: sae["value"]}
                if layer == sae["layer"]
                else {}
            ),
        }
        for layer, row, probability in zip(layers, rows, lens["probabilities"])
    }
    envelope = build_measurements(
        scope="token",
        geometry_readings={geometry["name"]: geometry_reading},
        lens_readings={lens["name"]: lens_reading},
        sae_readings={sae["name"]: sae_reading},
        per_layer_scores=per_layer_scores,
        lens_readout={
            layer: [(lens["token"], probability)]
            for layer, probability in zip(layers, lens["probabilities"])
        },
        lens_aggregate=[(
            lens["token"],
            lens_strength,
            lens_center,
            lens_spread,
        )],
        lens_token_ids={layer: [lens["tokenId"]] for layer in layers},
        lens_source=lens["source"],
        sae_features=[(
            sae["featureId"],
            sae["activation"],
            sae["label"],
            sae["maxAct"],
        )],
        sae_source=sae["source"],
        sae_layer=sae["layer"],
        steering=fixture["steering"],
    )

    assert envelope == fixture["expected"]


# ---------------------------------------------------------------------------
# Structured steering program
# ---------------------------------------------------------------------------


STRUCTURED_PROGRAM = (
    GOLDENS
    / "structured-program-parity-v1.json"
)

def _curve_subspace(curve: dict[str, Any]) -> LayerSubspace:
    return LayerSubspace(
        mean=torch.tensor(curve["neutral"], dtype=torch.float32),
        basis=torch.tensor(curve["basis"], dtype=torch.float32),
        node_params=torch.tensor(curve["nodeParameters"], dtype=torch.float32),
        rbf_weights=torch.tensor(curve["rbfWeights"], dtype=torch.float32),
        poly_coeffs=torch.tensor(curve["polynomial"], dtype=torch.float32),
        coord_offset=torch.tensor(curve["coordinateOffset"], dtype=torch.float32),
        coord_scale=torch.tensor(curve["coordinateScale"], dtype=torch.float32),
    )

def _curve_domain(curve: dict[str, Any]) -> SphereDomain | BoxDomain:
    if curve["domainKind"] == "sphere":
        return SphereDomain(len(curve["axes"]))
    return BoxDomain([
        BoxAxis(
            name=f"axis_{index}",
            periodic=axis["periodic"],
            period=axis["period"],
            lo=axis["lowerBound"],
            hi=axis["upperBound"],
        )
        for index, axis in enumerate(curve["axes"])
    ])

def _run_curve(
    residual: torch.Tensor,
    curve: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    return subspace_inject(
        residual,
        _curve_subspace(curve),
        _curve_domain(curve),
        torch.tensor(curve["target"], dtype=torch.float32),
        torch.tensor(curve["origin"], dtype=torch.float32),
        float(curve["along"]),
        float(curve["onto"]),
        gn_steps=4,
        origin=torch.tensor(curve["origin"], dtype=torch.float32),
    )

def test_multi_curve_periodic_fixture_matches_python_kernel() -> None:
    fixture = json.loads(STRUCTURED_PROGRAM.read_text())["multiCurve"]
    residual = torch.tensor(fixture["residual"], dtype=torch.float32)
    feet = []
    for curve in fixture["curves"]:
        residual, foot = _run_curve(residual, curve)
        feet.append(foot.tolist())

    assert residual.tolist() == pytest.approx(
        fixture["expected"]["residual"], abs=3e-6,
    )
    for actual, expected in zip(
        feet, fixture["expected"]["preSlideFeet"], strict=True,
    ):
        assert actual == pytest.approx(expected, abs=3e-6)

def test_sphere_curve_fixture_matches_python_kernel() -> None:
    fixture = json.loads(STRUCTURED_PROGRAM.read_text())["sphereCurve"]
    residual, foot = _run_curve(
        torch.tensor(fixture["residual"], dtype=torch.float32),
        fixture["curve"],
    )
    assert residual.tolist() == pytest.approx(
        fixture["expected"]["residual"], abs=3e-6,
    )
    assert foot.tolist() == pytest.approx(
        fixture["expected"]["preSlideFoot"], abs=3e-6,
    )

def test_sae_fixture_matches_local_drowse_backend() -> None:
    fixture = json.loads(STRUCTURED_PROGRAM.read_text())["sae"]
    weights = {
        "W_enc": torch.tensor(fixture["encoderDirection"], dtype=torch.float32).reshape(-1, 1),
        "W_dec": torch.tensor(fixture["decoderDirection"], dtype=torch.float32).reshape(1, -1),
        "b_enc": torch.tensor([fixture["encoderBias"]], dtype=torch.float32),
        "b_dec": torch.tensor(fixture["decoderBias"], dtype=torch.float32),
    }
    backend = LocalSaeBackend(
        release="fixture",
        revision="fixture",
        fingerprint="fixture",
        layers=frozenset({0}),
        model_fingerprint="fixture",
        _loader=lambda: weights,
    )
    activation = backend.encode_layer(
        0, torch.tensor([fixture["residual"]], dtype=torch.float32),
    )[0, 0]
    assert float(activation / fixture["maxAct"]) == pytest.approx(
        fixture["expectedNormalizedActivation"], abs=1e-7,
    )
    assert backend.feature_direction(0, 0).tolist() == pytest.approx(
        fixture["expectedDecoderDirection"], abs=1e-7,
    )

def test_jlens_fixture_matches_python_token_direction() -> None:
    fixture = json.loads(STRUCTURED_PROGRAM.read_text())["jlens"]
    lens = JacobianLens(
        {0: torch.tensor(fixture["jacobian"], dtype=torch.float32)},
        n_prompts=1,
        d_model=len(fixture["unembeddingRow"]),
    )
    unembedding = torch.tensor([fixture["unembeddingRow"]], dtype=torch.float32)
    assert lens.token_direction(0, unembedding)[0].tolist() == pytest.approx(
        fixture["expectedDirection"], abs=1e-7,
    )

@pytest.mark.parametrize("gate_index", [0, 1])
def test_gate_fixture_matches_python_trigger(gate_index: int) -> None:
    fixture = json.loads(STRUCTURED_PROGRAM.read_text())["gates"][gate_index]
    steering = parse_expr(fixture["expression"])
    trigger = next(iter(steering.entries().values()))[1]
    for case in fixture["contexts"]:
        context = TriggerContext(
            is_prefill=case["prefill"],
            thinking=case["thinking"],
            gen_step=case["generatedTokens"],
            probe_scores=case["scores"],
        )
        assert trigger.active(context) is case["active"]


# ---------------------------------------------------------------------------
# Topology RBF fit
# ---------------------------------------------------------------------------


TOPOLOGY_RBF = (
    GOLDENS
    / "topology-rbf-parity-v1.json"
)

def test_browser_topology_rbf_fixture_matches_python_source_of_truth() -> None:
    fixture = json.loads(TOPOLOGY_RBF.read_text())
    node_count = fixture["nodeCount"]
    output_dimensions = fixture["outputDimensions"]
    targets = torch.tensor(fixture["targets"], dtype=torch.float32).reshape(
        node_count, output_dimensions,
    )
    centered = targets - targets.mean(dim=0, keepdim=True)
    selection = fixture["selection"]
    coordinates, diagnostics = derive_spectral_coords(
        centered @ centered.transpose(0, 1),
        max_dim=selection["maxDimensions"],
        min_dim=selection["minDimensions"],
    )

    expected = fixture["expected"]
    assert coordinates.shape == (node_count, expected["intrinsicDimensions"])
    assert diagnostics.k_nn == expected["kNN"]
    assert diagnostics.bandwidth == pytest.approx(expected["bandwidth"], abs=1e-6)
    assert diagnostics.gap_magnitude == pytest.approx(
        expected["gapMagnitude"], abs=1e-6,
    )
    assert diagnostics.eigenvalues.tolist() == pytest.approx(
        expected["eigenvalues"], abs=1e-6,
    )
    for pair in expected["pairDistances"]:
        distance = torch.linalg.vector_norm(
            coordinates[pair["left"]] - coordinates[pair["right"]],
        )
        assert float(distance) == pytest.approx(pair["distance"], abs=1e-6)

    plan = RbfFitPlan.build(coordinates, smoothing=selection["smoothing"])
    fit = plan.fit(targets, smoothing=selection["smoothing"])
    evaluated = eval_rbf(plan.node_params, fit.weights, fit.poly, plan.node_params)
    assert fit.lam == pytest.approx(expected["lambda"], abs=1e-6)
    assert fit.edf == pytest.approx(
        expected["effectiveDegreesOfFreedom"], abs=1e-5,
    )
    assert evaluated.flatten().tolist() == pytest.approx(
        # Spectral coordinates and the RBF solve each contribute fp32 roundoff.
        expected["evaluatedTargets"], abs=5e-6,
    )

    whitener = LayerWhitener(
        {0: torch.zeros(1, output_dimensions, dtype=torch.float32)},
        {0: torch.ones(1, 1, dtype=torch.float32)},
        {0: 1.0},
        {0: torch.zeros(output_dimensions, dtype=torch.float32)},
    )
    gram = whitener.subspace_gram(0, centered)
    choice = select_topology(
        {0: centered},
        {0: gram},
        gram,
        whitener=whitener,
        max_dim=selection["maxDimensions"],
        smoothing=selection["smoothing"],
        persistence_frac=selection["persistenceFraction"],
    )
    assert choice.winner_name == expected["winnerName"]
    assert choice.fit_mode == expected["fitMode"]
    assert isinstance(choice.domain, CustomDomain)
    assert expected["domainKind"] == "custom"
    assert [
        {
            "name": candidate.name,
            "intrinsicDimensions": candidate.intrinsic_dim,
            "viable": candidate.viable,
        }
        for candidate in choice.candidates
    ] == expected["candidates"]
    assert isinstance(choice.diagnostics, SpectralDiagnostics)
    assert choice.diagnostics.k_nn == expected["kNN"]
    assert choice.diagnostics.bandwidth == pytest.approx(
        expected["bandwidth"], abs=1e-6,
    )


# ---------------------------------------------------------------------------
# Adversarial topology detection
# ---------------------------------------------------------------------------


with (GOLDENS / "topology-adversarial-v1.csv").open() as fixture:
    TOPOLOGY_ADVERSARIAL_CASES = list(csv.DictReader(fixture))

def _adversarial_sample(case: dict[str, str]):
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

def _adversarial_detect(points: torch.Tensor):
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

@pytest.mark.parametrize("case", TOPOLOGY_ADVERSARIAL_CASES, ids=lambda c: "-".join(c.values()))
@pytest.mark.parametrize("variant", ["original", "permuted", "scaled"])
def test_periodic_chart_specificity_and_coordinate_fidelity(case: dict[str, str], variant: str):
    points, truth = _adversarial_sample(case)
    if variant == "permuted":
        permutation = torch.randperm(len(points), generator=torch.Generator().manual_seed(917))
        points, truth = points[permutation], truth[permutation]
    if variant == "scaled":
        points = points * 0.025
    result = _adversarial_detect(points)
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
    assert _adversarial_detect(points) is None

def test_duplicate_samples_do_not_validate_a_collapsed_chart():
    points, truth = _adversarial_sample(TOPOLOGY_ADVERSARIAL_CASES[0])
    points[1] = points[0]
    distances = torch.cdist(points.double(), points.double()).float()
    assert not _periodic_neighborhoods_preserved(distances, truth)
    assert _adversarial_detect(points) is None

def test_far_outlier_does_not_invent_periodic_structure():
    points, _ = _adversarial_sample(TOPOLOGY_ADVERSARIAL_CASES[0])
    assert _adversarial_detect(torch.cat((points, torch.tensor([[50.0, 0.0]])))) is None

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
