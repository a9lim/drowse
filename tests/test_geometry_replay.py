"""CPU tests for ``GeometryInstrument.token_readout`` (the geometry token replay).

The instrument reads the Monitor roster at the forward a real
:class:`~drowse.session.replay.TokenReplayer` rebuilds over a toy model.  The
Monitor is stubbed (its scoring math has its own suite): these tests pin the
geometry read — the roster's layer union drives the capture, the read forces
the cold foot solve, the readings land in the replay envelope, and an empty
roster is a caller error even when the detach wins the race into the
exclusive section.  The shared rebuild (anchor, render, prefix, steering) is
pinned in ``test_token_replay``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drowse.readout.types import ProbeReading
from drowse.loom.errors import UnknownNodeError
from drowse.loom.recipe import Recipe
from drowse.readout.geometry import GeometryInstrument
from tests.support.replay import ReplayRig


class _StubMonitor:
    """Records what the replay hands the Monitor; returns a fixed reading."""

    def __init__(self, probe_names: list[str], layers: set[int]) -> None:
        self.probe_names = probe_names
        self._layers = set(layers)
        self.scored: list[dict[int, torch.Tensor]] = []
        self.warm_calls: list[bool] = []

    def probe_layers(self) -> set[int]:
        return set(self._layers)

    def enable_curved_warm(self, flag: bool) -> None:
        self.warm_calls.append(bool(flag))

    def score_single_token(
        self, hidden: dict[int, torch.Tensor],
    ) -> dict[str, ProbeReading]:
        self.scored.append(hidden)
        return {
            name: ProbeReading(
                fraction=0.4,
                nearest=[("formal", 1.2)],
                coords=(0.7,),
                residual=0.0,
            )
            for name in self.probe_names
        }


def _geometry(rig: ReplayRig) -> tuple[GeometryInstrument, Any]:
    session = SimpleNamespace(_monitor=_StubMonitor(["formal.casual"], {0, 1}))
    return GeometryInstrument(session, replay=rig.replay), session  # type: ignore[arg-type]


def _geometry_block(out: dict[str, Any]) -> dict[str, Any]:
    measurements = out["measurements"]
    assert measurements["scope"] == "replay"
    assert measurements["provenance"] == "replayed"
    return measurements["instruments"]["geometry"]


def test_geometry_token_readout_reads_the_roster_at_the_producing_position() -> None:
    rig = ReplayRig()
    geometry, session = _geometry(rig)
    node_id = rig.turn(rig.tokenizer.encode("abcdefg"), Recipe(system_prompt=None))

    block = _geometry_block(geometry.token_readout(node_id, 3))

    # Readout position: the forward that PRODUCED the clicked token —
    # prompt + raw[:3], pooled at the final position, never including the
    # clicked token itself; capture narrowed to the roster's layer union.
    assert rig.captures == [(rig.prompt_len + 3, rig.prompt_len + 2, (0, 1))]
    assert block["binding"] == {"source": None, "steering": None}
    assert block["readings"]["formal.casual"]["coords"] == [0.7]
    # The Monitor scored exactly the captured layer union, and the replay
    # forced the cold foot solve (a stale warm foot from a prior generation
    # must not seed a one-shot read).
    assert set(session._monitor.scored[0]) == {0, 1}
    assert session._monitor.warm_calls == [False]
    assert rig.exclusive_messages[0][0] == (
        "session.geometry.token_readout called while another model use is "
        "in flight"
    )


def test_geometry_token_readout_binds_the_recipe_steering() -> None:
    rig = ReplayRig()
    geometry, _session = _geometry(rig)
    node_id = rig.turn(
        rig.tokenizer.encode("abcd"), Recipe(steering="0.3 formal.casual"),
    )

    block = _geometry_block(geometry.token_readout(node_id, 2))
    assert block["binding"]["steering"] == "0.3 formal.casual"
    assert rig.steering_scopes == ["0.3 formal.casual"]

    # The unsteered counterfactual: no scope opened, steering reported null.
    block = _geometry_block(
        geometry.token_readout(node_id, 2, apply_steering=False),
    )
    assert block["binding"]["steering"] is None
    assert rig.steering_scopes == ["0.3 formal.casual"]


def test_geometry_token_readout_raw_mode_render() -> None:
    rig = ReplayRig()
    geometry, _session = _geometry(rig)
    node_id = rig.turn(rig.tokenizer.encode("abcd"))
    geometry.token_readout(node_id, 1, raw=True)
    call = rig.prepare_calls[0]
    assert call["raw"] is True
    assert call["input"] == ""
    assert call["parent_node_id"] == rig.tree.get(node_id).parent_id


def test_geometry_token_readout_errors() -> None:
    rig = ReplayRig()
    geometry, session = _geometry(rig)
    node_id = rig.turn(rig.tokenizer.encode("abc"))

    with pytest.raises(UnknownNodeError):
        geometry.token_readout("no-such-node", 0)
    # An empty roster is a clear caller error, not an empty reading, and it
    # is checked before any replay work.
    session._monitor = _StubMonitor([], set())
    with pytest.raises(ValueError, match="no geometry probes attached"):
        geometry.token_readout(node_id, 0)
    assert rig.captures == []


def test_geometry_token_readout_detach_race_keeps_caller_error() -> None:
    """sol's round-3 P3: a detach winning the gap between the entry roster
    check and the exclusive section must still surface the caller-facing
    "no geometry probes attached" error — not a capture-layer
    implementation leak.  The roster is re-read inside the exclusive
    section; the rig's exclusive guard performs the detach to pin the
    interleaving deterministically."""
    rig = ReplayRig()
    geometry, session = _geometry(rig)
    node_id = rig.turn(rig.tokenizer.encode("abc"))

    def _detach() -> None:
        # A detach mutates the session's one Monitor in place.
        session._monitor.probe_names = []
        session._monitor._layers = set()

    rig.on_exclusive = _detach
    with pytest.raises(ValueError, match="no geometry probes attached"):
        geometry.token_readout(node_id, 0)
    assert rig.captures == []
