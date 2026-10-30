"""Token-probe payload helper tests."""

from __future__ import annotations

import math

import pytest

from drowse.generation.results import ProbeReading
from drowse.generation.token_payloads import TokenProbePayload


def test_from_geometry_shapes_scores_and_channels() -> None:
    reading = ProbeReading(
        fraction=0.5,
        nearest=[("happy", 0.1)],
        coords=(0.25,),
        coords_per_layer={3: (0.25,)},
    )

    payload = TokenProbePayload.from_geometry({"toy": reading}, per_layer=True)

    assert payload.scores == {"toy": pytest.approx(0.25)}
    # Monitor readings land in the geometry family slot; ``all_readings`` is the
    # cross-family union.
    assert payload.geometry_readings == {"toy": reading}
    assert payload.all_readings == {"toy": reading}
    assert payload.per_layer_scores == {"3": {"toy": pytest.approx(0.25)}}

    envelope = payload.to_token_payload(lens={"3": [("tok", 1.0)]})["measurements"]
    assert envelope["scope"] == "token"
    assert envelope["instruments"]["geometry"]["readings"]["toy"] == (
        reading.to_dict()
    )
    assert envelope["instruments"]["lens"]["readout"]["layers"] == [
        {
            "layer": 3,
            "tokens": [{"token": "tok", "id": -1, "logprob": pytest.approx(0.0)}],
        },
    ]


def test_token_payload_builds_measurement_envelope() -> None:
    reading = ProbeReading(
        fraction=0.5,
        nearest=[("happy", 0.1)],
        coords=(0.25,),
        coords_per_layer={3: (0.25,)},
    )
    payload = TokenProbePayload.from_geometry(
        {"toy": reading}, per_layer=True,
    ).to_token_payload(
        lens={3: [(" tok", 0.25)]},
        lens_aggregate=[(" tok", 0.2, 0.6, 0.1)],
        lens_token_ids={3: [42]},
        lens_source="local:default",
        sae=[(7, 3.5, "feature seven", 5.0)],
        sae_source="saelens:release",
        sae_layer=3,
        steering="0.3 calm",
    )

    envelope = payload["measurements"]
    assert envelope["version"] == 1
    assert envelope["scope"] == "token"
    assert envelope["provenance"] == "captured"
    assert envelope["scores"] == {"toy": pytest.approx(0.25)}

    instruments = envelope["instruments"]
    assert instruments["geometry"]["readings"]["toy"] == reading.to_dict()
    assert instruments["lens"] == {
        "readout": {
            "layers": [{
                "layer": 3,
                "tokens": [{
                    "token": " tok",
                    "id": 42,
                    "logprob": pytest.approx(math.log(0.25)),
                }],
            }],
            "aggregate": [{
                "token": " tok",
                "strength": 0.2,
                "com": 0.6,
                "spread": 0.1,
            }],
        },
        "binding": {"source": "local:default", "steering": "0.3 calm"},
    }
    assert instruments["sae"] == {
        "readout": {
            "features": [{
                "id": 7,
                "activation": 3.5,
                "label": "feature seven",
                "max_act": 5.0,
            }],
        },
        "binding": {
            "source": "saelens:release",
            "steering": "0.3 calm",
            "layer": 3,
        },
    }


def test_merge_readings_routes_to_named_family_slot() -> None:
    geo = ProbeReading(0.0, [], coords=(0.1,))
    lens = ProbeReading(0.0, [], coords=(0.2,))
    sae = ProbeReading(0.0, [], coords=(0.3,))

    payload = TokenProbePayload()
    payload.merge_readings({"toy": geo}, family="geometry")
    payload.merge_readings({"jlens/yes": lens}, family="lens")
    payload.merge_readings({"sae/7": sae}, family="sae")

    assert payload.geometry_readings == {"toy": geo}
    assert payload.lens_readings == {"jlens/yes": lens}
    assert payload.sae_readings == {"sae/7": sae}
    assert payload.all_readings == {"toy": geo, "jlens/yes": lens, "sae/7": sae}
    assert payload.scores == {
        "toy": pytest.approx(0.1),
        "jlens/yes": pytest.approx(0.2),
        "sae/7": pytest.approx(0.3),
    }
    with pytest.raises(ValueError, match="unknown reading family"):
        payload.merge_readings({"x": geo}, family="bogus")


def test_from_geometry_leaves_per_layer_rows_off_unless_asked() -> None:
    reading = ProbeReading(
        fraction=0.4, nearest=[("calm", 0.2)], coords=(0.75,),
        coords_per_layer={2: (0.5,)},
    )
    payload = TokenProbePayload.from_geometry({"toy": reading}, per_layer=False)

    assert payload.scores == {"toy": pytest.approx(0.75)}
    assert payload.geometry_readings == {"toy": reading}
    assert payload.per_layer_scores is None
