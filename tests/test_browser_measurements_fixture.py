from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch

from drowse.core.instruments.sae import SaeInstrument
from drowse.core.instruments.types import DepthSummary, ScalarReading
from drowse.core.jlens import (
    aggregate_readout_from_probabilities,
    token_readout_stats_from_probabilities,
)
from drowse.core.measurements import build_measurements
from drowse.core.monitor import _depth_stats
from drowse.core.results import ProbeReading
from drowse.core.session import DrowseSession


FIXTURE = (
    Path(__file__).parents[1]
    / "browser-runtime"
    / "fixtures"
    / "measurement-envelope-v1.json"
)


def test_browser_measurement_fixture_matches_python_source_of_truth() -> None:
    fixture = json.loads(FIXTURE.read_text())
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
        _layers=[object() for _ in range(model_layer_count)],
        _require_sae=lambda: (None, sae["layer"], sae["featureCount"]),
        _sae_max_act=lambda _feature_id: sae["maxAct"],
        _readout_long_tensor=lambda ids, device: torch.tensor(
            ids, dtype=torch.long, device=device,
        ),
    )
    sae_instrument = SaeInstrument(cast(DrowseSession, sae_session))
    sae_instrument.probes[sae["name"]] = {
        "feature_id": sae["featureId"],
        "layer": sae["layer"],
        "label": sae["label"],
        "max_act": sae["maxAct"],
    }
    sae_activations = torch.zeros(sae["featureCount"], dtype=torch.float32)
    sae_activations[sae["featureId"]] = sae["activation"]
    sae_reading = sae_instrument.score_probes_from_activations(
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
