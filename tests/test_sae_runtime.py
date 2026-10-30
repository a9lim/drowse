"""The live SAE family: runtime residency + metadata, instrument reads, and the
session wiring (CPU).

Unit tests drive a real :class:`~drowse.sae.instrument.SaeInstrument` and
:class:`~drowse.sae.runtime.SaeRuntime` over a narrow stub (``_instrument``);
integration tests load a mock dictionary into a real tiny session through the
public ``session.sae.load`` path (``_load_mock``).
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse import SamplingConfig
from drowse.readout.types import ReadRequest
from drowse.sae.backend import ResolvedSae, sae_device_str, select_runtime_layer
from drowse.steering.composer import SteeringComposer
from drowse.steering.grammar import format_expr, parse_expr
from tests.support.sae import MockSaeBackend, make_resident, stub_sae_instrument
from tests.support.sessions import TinySessionFactory


class _Capture:
    def __init__(self, row: torch.Tensor, layer: int = 1) -> None:
        self.layer = layer
        self.row = row

    def per_layer_buckets(self) -> dict[int, list[torch.Tensor]]:
        return {self.layer: [self.row]}

    def latest_per_layer(self) -> dict[int, torch.Tensor]:
        return {self.layer: self.row}


def _instrument() -> Any:
    """A real SAE instrument over a stub: a 4-wide identity dictionary
    resident at layer 1, capture row ``[0.2, 3.0, 5.0, 1.0]``, and feature 2
    already carrying Neuronpedia metadata."""
    stub = SimpleNamespace(
        model_id="org/model",
        model_info={"hidden_dim": 4},
        layers=[object(), object(), object(), object()],
        device=torch.device("cpu"),
        dtype=torch.float32,
        _capture=_Capture(torch.tensor([0.2, 3.0, 5.0, 1.0])),
    )
    inst = stub_sae_instrument(stub)
    make_resident(
        inst.runtime,
        MockSaeBackend(layers=frozenset({1}), d_model=4, d_feature=4),
        layer=1,
        feature_meta={"2": {"label": "feature two", "max_act": 10.0, "checked": True}},
    )
    inst.active_for_generation = True
    return inst


def _attach(inst: Any, name: str, fid: int, *, label: Any = None, max_act: Any = None) -> None:
    inst.probes[name] = {
        "feature_id": fid, "layer": 1, "label": label, "max_act": max_act,
    }


def _lock_held_elsewhere(lock: Any) -> bool:
    """Whether ``lock`` is held, probed from another thread (an RLock held by
    the caller's own thread would otherwise look free)."""
    out: list[bool] = []

    def probe() -> None:
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        out.append(not acquired)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return out[0]


def _load_mock(
    session: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    layer: int = 1,
    release: str = "mock-release",
    neuronpedia_id: str | None = None,
) -> MockSaeBackend:
    """Load an identity mock dictionary through ``session.sae.load``, with
    only provider resolution patched (binding + metadata go to disk)."""
    import drowse.sae.runtime as runtime_module

    hidden = int(session.model_info["hidden_dim"])
    backend = MockSaeBackend(
        layers=frozenset({layer}), d_model=hidden, release=release,
        revision="c" * 40,
        neuronpedia_ids_by_layer=(
            {str(layer): neuronpedia_id} if neuronpedia_id else {}
        ),
    )
    monkeypatch.setattr(
        runtime_module, "resolve_sae_backend",
        lambda *_a, **_k: ResolvedSae(backend, layer, hidden),
    )
    session.sae.load(release)
    return backend


# ---------------------------------------------------------------- backend


def test_sae_device_str_strips_mps_index() -> None:
    # safetensors rejects the indexed MPS form a live model reports
    # ("device mps:0 is invalid"); CUDA indices must survive.
    assert sae_device_str(torch.device("mps", 0)) == "mps"
    assert sae_device_str("mps:0") == "mps"
    assert sae_device_str("mps") == "mps"
    assert sae_device_str("cuda:1") == "cuda:1"
    assert sae_device_str("cpu") == "cpu"


def test_select_runtime_layer_prefers_workspace_near_65_percent() -> None:
    assert select_runtime_layer({1, 8, 14, 18}, 24) == 14
    assert select_runtime_layer({1, 8, 14, 18}, 24, requested=8) == 8
    with pytest.raises(ValueError, match="does not cover"):
        select_runtime_layer({1, 8}, 24, requested=9)


# ---------------------------------------------------------------- runtime


def test_failed_provider_binding_does_not_half_adopt_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.sae.runtime as runtime_module
    from drowse.sae import sources as sae_io

    inst = _instrument()
    previous = inst.runtime.resident
    replacement = MockSaeBackend(
        layers=frozenset({1}), d_model=4, release="replacement",
        revision="commit", repo_id="org/sae",
    )
    monkeypatch.setattr(
        runtime_module, "resolve_sae_backend",
        lambda *_a, **_k: ResolvedSae(replacement, 1, 4),
    )
    monkeypatch.setattr(sae_io, "load_sae_feature_meta", lambda *_a, **_k: {})
    monkeypatch.setattr(
        sae_io, "save_sae_metadata",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    _attach(inst, "sae/2", 2)

    with pytest.raises(RuntimeError, match="disk full"):
        inst.load("replacement", layer=1)

    assert inst.runtime.resident is previous
    assert inst.info["release"] == "mock-release"
    assert previous.feature_meta == {
        "2": {"label": "feature two", "max_act": 10.0, "checked": True},
    }
    assert "sae/2" in inst.probes  # nothing keyed by the old release dropped
    assert inst.effects["evictions"] == 0


def test_sae_encoder_readout_is_detached_from_autograd() -> None:
    acts = _instrument().runtime.encode(torch.ones(4, requires_grad=True))
    assert not acts.requires_grad


def test_sae_feature_validation_and_decoder_row_registration() -> None:
    inst = _instrument()
    assert inst.validate_feature(2) == {
        "id": 2, "label": "feature two", "layer": 1, "max_act": 10.0,
    }
    name = inst.register_direction(2)
    assert name == "sae/2"
    assert torch.equal(
        inst.effects["directions"][name][1], torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )
    with pytest.raises(ValueError, match="out of range"):
        inst.validate_feature(9)
    with pytest.raises(ValueError, match="out of range"):
        inst.register_direction(9)


def test_offline_provider_restore_loads_the_binding_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup restore reloads a fetched provider SAE at its pinned commit.

    Offline (every Hub metadata call fails) the recorded full commit still
    pins the load, so a cached SAE restores instead of failing to resolve an
    immutable revision; online it can never drift to upstream HEAD.
    """
    import sys
    import types

    import drowse.base.hf as hf
    from drowse.sae.backend import load_sae_backend
    from drowse.sae.sources import save_sae_metadata

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    commit = "b" * 40
    seen: list[str | None] = []
    fake_sae_lens = types.ModuleType("sae_lens")

    class FakeSAE:
        def __init__(self) -> None:
            self.cfg = types.SimpleNamespace(
                d_in=4, d_sae=8, model_name="org/model", hook_layer=1,
                metadata={"hook_name": "blocks.1.hook_resid_post"},
            )
            self.W_dec = torch.zeros(8, 4)

        @classmethod
        def from_pretrained(
            cls, release: Any, sae_id: Any, device: Any = None,
            revision: str | None = None,
        ) -> Any:
            del release, sae_id, device
            seen.append(revision)
            return cls(), {"hook_layer": 1}, None

        def to(self, **_kwargs: Any) -> "FakeSAE":
            return self

    fake_sae_lens.SAE = FakeSAE  # type: ignore[attr-defined]
    fake_sae_lens.get_pretrained_saes_directory = lambda: {  # type: ignore[attr-defined]
        "prov": {
            "saes_map": {"layer_1": 1}, "model": "org/model",
            "repo_id": "org/saes",
        },
    }
    monkeypatch.setitem(sys.modules, "sae_lens", fake_sae_lens)

    def offline() -> Any:
        raise OSError("offline: no Hub metadata")

    monkeypatch.setattr(hf, "_hf_api", offline)

    backend = load_sae_backend(
        "prov", revision=commit, model_id="org/model", device="cpu",
    )
    assert backend.revision == commit and backend.fingerprint is not None
    assert backend.feature_count(1) == 8
    assert seen == [commit]

    save_sae_metadata("org/model", "prov", {
        "layer": 1, "width": 8, "revision": commit, "fingerprint": "fp",
        "sae_id": "layer_1", "repo_id": "org/saes", "neuronpedia_id": None,
    })
    inst = _instrument()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        inst.runtime, "load",
        lambda release, **kwargs: calls.append({"release": release, **kwargs}),
    )
    inst.restore_active()
    assert calls == [{
        "release": "prov", "layer": 1, "revision": commit,
        "on_swap": inst._on_swap,
    }]


# ----------------------------------------------------------- live + probes


def test_live_sae_readout_and_probe_share_one_encoder_result() -> None:
    inst = _instrument()
    _attach(inst, "sae/2", 2, label="feature two", max_act=10.0)
    assert inst.enable_live() == {"layer": 1}
    readout = inst.live_readout_step(top_k=3)
    assert readout == [
        (2, 5.0, "feature two", 10.0),
        (1, 3.0, None, None),
        (3, 1.0, None, None),
    ]
    # The probe channel is normalized strength — activation / maxActApprox —
    # while the readout row keeps the raw activation beside the unit.
    reading = inst.last_step_readings["sae/2"]
    assert reading.value == 0.5
    assert reading.unit == "activation_over_max"
    assert reading.per_layer == {1: 0.5}


def _spy_raw_by_fid(inst: Any, *, drop_only: bool = False) -> dict[int, float]:
    seen_raw: dict[int, float] = {}
    original = inst.probe_values

    def spy_probe_values(
        activations: torch.Tensor,
        *,
        only: set[str] | None = None,
        raw_by_fid: dict[int, float] | None = None,
    ) -> list[tuple[str, int, float, float]]:
        seen_raw.update(raw_by_fid or {})
        if drop_only:
            only = None
        return original(activations, only=only, raw_by_fid=raw_by_fid)

    inst.probe_values = spy_probe_values
    return seen_raw


def test_live_sae_readout_seeds_topk_raw_values_for_pinned_probes() -> None:
    inst = _instrument()
    _attach(inst, "sae/2", 2, label="feature two", max_act=10.0)
    inst.enable_live()
    seen_raw = _spy_raw_by_fid(inst)

    inst.live_readout_step(top_k=3)

    assert seen_raw[2] == pytest.approx(5.0)


def test_sae_probe_without_metadata_reads_raw_activation() -> None:
    inst = _instrument()
    _attach(inst, "sae/1", 1)
    inst.enable_live()
    inst.live_readout_step(top_k=1)
    reading = inst.last_step_readings["sae/1"]
    assert reading.value == 3.0
    assert reading.unit == "raw_activation"


def test_sae_gate_scalar_stashes_activations_for_live_step() -> None:
    inst = _instrument()
    _attach(inst, "sae/1", 1)
    scalars = inst.current_run.gate_scalars(6, None, None)
    assert scalars["sae/1"] == pytest.approx(3.0)
    assert scalars["sae/1[0]"] == pytest.approx(3.0)
    # The fake geometry constants are gone (5.x): an SAE probe emits only
    # its real strength channel; unsupported channels are a preflight error,
    # never a silently-constant comparison.
    assert "sae/1:fraction" not in scalars
    assert "sae/1:membership" not in scalars
    assert inst.step_stash is not None
    assert inst.step_stash["step"] == 6


def test_sae_gate_scalar_and_live_step_share_one_encoder_result() -> None:
    inst = _instrument()
    _attach(inst, "sae/2", 2, label="feature two", max_act=10.0)
    inst.enable_live()
    calls = 0
    original = inst.runtime.encode

    def counting_encode(hidden: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(hidden)

    inst.runtime.encode = counting_encode

    scalars = inst.current_run.gate_scalars(2, None, {"sae/2"})
    readout = inst.live_readout_step(top_k=3, step_id=2)

    assert calls == 1
    assert scalars["sae/2"] == pytest.approx(0.5)
    assert readout is not None
    assert inst.last_step_readings is not None
    assert inst.last_step_readings["sae/2"].value == pytest.approx(0.5)


def test_sae_gate_raw_values_seed_live_probe_reads_outside_topk() -> None:
    inst = _instrument()
    _attach(inst, "sae/0", 0)
    inst.enable_live()  # Feature 2 wins the one-wide generation readout.
    seen_raw = _spy_raw_by_fid(inst, drop_only=True)

    scalars = inst.current_run.gate_scalars(9, None, {"sae/0"})
    readout = inst.live_readout_step(top_k=1, step_id=9)

    assert scalars["sae/0"] == pytest.approx(0.2)
    assert readout is not None
    assert seen_raw[0] == pytest.approx(0.2)
    assert inst.last_step_readings is not None
    assert inst.last_step_readings["sae/0"].value == pytest.approx(0.2)


def test_sae_probe_values_reuse_feature_selector_tensor() -> None:
    inst = _instrument()
    _attach(inst, "sae/1", 1)
    _attach(inst, "sae/2", 2, label="feature two", max_act=10.0)
    acts = inst.runtime.encode(inst._session._capture.latest_per_layer()[1])

    first = inst.score_probes_from_activations(acts)
    first_ids = {key: id(value) for key, value in inst._selectors.items()}
    second = inst.score_probes_from_activations(acts)

    assert first["sae/1"].value == second["sae/1"].value
    assert first["sae/2"].value == second["sae/2"].value
    assert first_ids
    assert {key: id(value) for key, value in inst._selectors.items()} == first_ids


def test_sae_gate_scalar_is_normalized_when_metadata_known() -> None:
    inst = _instrument()
    _attach(inst, "sae/2", 2, label="feature two", max_act=10.0)
    scalars = inst.current_run.gate_scalars(-1, None, None)
    assert scalars["sae/2"] == pytest.approx(0.5)


def test_sae_gate_scalar_scores_only_referenced_probe() -> None:
    inst = _instrument()
    _attach(inst, "sae/1", 1)
    _attach(inst, "sae/2", 2, label="feature two", max_act=10.0)

    scalars = inst.current_run.gate_scalars(-1, None, {"sae/1"})

    assert scalars["sae/1"] == pytest.approx(3.0)
    assert scalars["sae/1[0]"] == pytest.approx(3.0)
    assert "sae/2" not in scalars


def test_sae_gate_keys_none_vs_empty_contract() -> None:
    """``gate_keys=None`` scores the full roster; an explicit ``set()``
    scores nothing.  The SAE member of the three-family contract pin."""
    inst = _instrument()
    _attach(inst, "sae/2", 2, label="feature two", max_act=10.0)
    full = inst.gate_scalars(None, step_id=0)
    assert "sae/2" in full
    assert inst.gate_scalars(set(), step_id=0) == {}


def test_composer_detects_attached_sae_gate() -> None:
    stub = SimpleNamespace(
        instruments={
            "geometry": SimpleNamespace(names=[], validate_gate=lambda _r: None),
            "lens": SimpleNamespace(names=[], validate_gate=lambda _r: None),
            "sae": SimpleNamespace(
                names=["sae/2"], validate_gate=lambda _r: None,
            ),
        },
        _monitor=SimpleNamespace(probe_names=()),
    )
    composer = SteeringComposer(stub)  # type: ignore[arg-type]
    steering = parse_expr("0.3 sae/2@when:sae/2>3")
    composer._stack.append(dict(steering.alphas))  # type: ignore[arg-type]
    assert composer.gated_sae_probe_keys() == {"sae/2"}


def test_unsupported_sae_gate_channel_raises_at_preflight() -> None:
    """A gate on a channel the SAE family can never produce is a preflight
    error (the 5.x replacement for the silently-constant fake channels)."""
    from drowse.errors import UnsupportedProbeChannelError

    stub = SimpleNamespace(_monitor=SimpleNamespace(probe_names=()))
    instrument = stub_sae_instrument(stub)
    _attach(instrument, "sae/2", 2)
    stub.instruments = {
        "geometry": SimpleNamespace(names=[], validate_gate=lambda _r: None),
        "lens": SimpleNamespace(names=[], validate_gate=lambda _r: None),
        "sae": instrument,
    }
    composer = SteeringComposer(stub)  # type: ignore[arg-type]
    steering = parse_expr("0.3 sae/2@when:sae/2:membership>0.5")
    composer._stack.append(dict(steering.alphas))  # type: ignore[arg-type]
    with pytest.raises(UnsupportedProbeChannelError) as exc_info:
        composer.gated_sae_probe_keys()
    status, text = exc_info.value.user_message()
    assert status == 400
    assert "sae/2:membership" in text
    # A supported-channel gate on the same probe still composes.
    composer._stack.clear()
    ok = parse_expr("0.3 sae/2@when:sae/2>3")
    composer._stack.append(dict(ok.alphas))  # type: ignore[arg-type]
    assert composer.gated_sae_probe_keys() == {"sae/2"}


@pytest.mark.parametrize(
    "expr",
    [
        "0.3 sae/9143",
        "!sae/9143",
        "0.3 sae/9143@when:sae/9143>3",
    ],
)
def test_sae_integer_atom_grammar_roundtrips(expr: str) -> None:
    parsed = parse_expr(expr)
    assert format_expr(parsed) == expr


# --------------------------------------------------------- metadata on disk


def test_sae_runtime_metadata_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.sae.sources import load_sae_metadata, save_sae_metadata

    path = save_sae_metadata("org/model", "release/name", {
        "layer": 14, "width": 16_384, "revision": "main",
        "fingerprint": "abc", "sae_id": "layer_14", "repo_id": "org/sae",
        "neuronpedia_id": None,
    })
    assert path.exists()
    assert load_sae_metadata("org/model", "release/name") == {
        "format_version": 3,
        "model_id": "org/model",
        "release": "release/name",
        "layer": 14,
        "width": 16_384,
        "revision": "main",
        "fingerprint": "abc",
        "sae_id": "layer_14",
        "repo_id": "org/sae",
        "neuronpedia_id": None,
    }


def test_sae_feature_meta_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.sae.sources import (
        load_sae_feature_meta,
        save_sae_feature_meta,
    )

    assert load_sae_feature_meta("org/model", "rel") == {}
    save_sae_feature_meta("org/model", "rel", cast("dict[str, dict[str, Any]]", {
        "7": {"label": "days of the week", "max_act": 121.11, "checked": True},
        "9": {"label": None, "max_act": None},
    }))
    meta = load_sae_feature_meta("org/model", "rel")
    assert meta["7"] == {"label": "days of the week", "max_act": 121.11}
    assert meta["9"] == {"label": None, "max_act": None}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(format_version=1),
        lambda payload: payload.update(model_id="other/model"),
        lambda payload: payload.update(release="other-release"),
        lambda payload: payload.update(extra=True),
        lambda payload: payload["features"]["7"].update(checked=True),
        lambda payload: payload["features"].update({"8": {"label": None}}),
    ],
)
def test_sae_feature_meta_rejects_non_current_shapes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: Any,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    import json

    from drowse.sae.sources import (
        SAE_FEATURE_META_FORMAT_VERSION,
        load_sae_feature_meta,
        sae_features_path,
    )

    path = sae_features_path("org/model", "rel")
    path.parent.mkdir(parents=True)
    payload = {
        "format_version": SAE_FEATURE_META_FORMAT_VERSION,
        "model_id": "org/model",
        "release": "rel",
        "source": None,
        "features": {"7": {"label": "weekdays", "max_act": 3.0}},
    }
    mutation(payload)
    path.write_text(json.dumps(payload))
    assert load_sae_feature_meta("org/model", "rel") == {}


@pytest.mark.parametrize("max_act", [float("nan"), float("inf")])
def test_sae_feature_meta_rejects_nonfinite_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_act: float,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.sae.sources import save_sae_feature_meta

    with pytest.raises(ValueError, match="invalid SAE feature metadata"):
        save_sae_feature_meta("org/model", "release", {
            "1": {"label": "feature", "max_act": max_act},
        })


@pytest.mark.parametrize("feature_id", ["", "01", "-1", "feature"])
def test_sae_feature_meta_rejects_noncanonical_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feature_id: str,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.sae.sources import save_sae_feature_meta

    with pytest.raises(ValueError, match="feature id"):
        save_sae_feature_meta("org/model", "release", {
            feature_id: {"label": "feature", "max_act": 1.0},
        })


def test_stream_aggregate_keeps_lens_and_sae_probe_readings() -> None:
    # ``probe_reading_aggregate`` filters result readings to attached probes;
    # lens/SAE probes live on their own session registries (readout channels,
    # not the Monitor), so the filter must union all three rosters — it used
    # to drop their end-of-gen aggregates from every streaming done frame.
    from drowse.generation.results import ProbeReading
    from drowse.server.compat import probe_reading_aggregate

    readings = {
        "confident.uncertain": ProbeReading(0.1, [], coords=(0.3,)),
        "jlens/fake": ProbeReading(0.0, [], coords=(0.02,)),
        "sae/548": ProbeReading(0.0, [], coords=(0.84,)),
        "sae/999": ProbeReading(0.0, [], coords=(0.5,)),  # detached — dropped
    }
    session = SimpleNamespace(
        monitor=SimpleNamespace(probe_names=("confident.uncertain",)),
        lens=SimpleNamespace(names=["jlens/fake"]),
        sae=SimpleNamespace(names=["sae/548"]),
    )
    result = SimpleNamespace(probe_readings=readings)
    out = probe_reading_aggregate(session, result)  # type: ignore[arg-type]
    assert set(out) == {"confident.uncertain", "jlens/fake", "sae/548"}
    assert out["sae/548"]["coords"] == [0.84]


# ------------------------------------------------------ metadata backfill


def test_fetch_meta_batch_caches_and_updates_probes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.sae import neuronpedia
    from drowse.sae.sources import load_sae_feature_meta

    inst = _instrument()
    _attach(inst, "sae/1", 1)
    fetched: list[int] = []

    def fake_fetch(_np_id: Any, idx: int, **_kw: Any) -> dict[str, object] | None:
        fetched.append(idx)
        if idx == 3:
            return None  # network miss — retryable, not cached
        return {"label": f"feature {idx}", "max_act": 4.0, "checked": True}

    monkeypatch.setattr(neuronpedia, "fetch_feature", fake_fetch)
    out = inst.fetch_meta([1, 1, 2, 3, 99])
    # id 2 was already cached (max_act present) — no refetch; 99 is out of
    # range; 3 failed and stays absent so a later call retries it.
    assert sorted(fetched) == [1, 3]
    assert out["1"] == {"label": "feature 1", "max_act": 4.0}
    assert out["2"] == {"label": "feature two", "max_act": 10.0}
    assert "3" not in out and "99" not in out
    # The attached probe's spec + digest reflect the new unit.
    assert inst.probes["sae/1"]["max_act"] == 4.0
    assert inst.effects["forgotten"] == ["sae/1"]
    # Persisted — a fresh load sees the merged cache.
    on_disk = load_sae_feature_meta("org/model", "mock-release", source=inst.info)
    assert on_disk["1"]["max_act"] == 4.0


# ------------------------------------------------------- bound-run state


def test_sae_registry_lock_snapshots_idle_reads(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """round-6 P2: the idle-passthrough spec source is a per-call
    coherent snapshot under the SAE registry lock — handing out the live
    dict let one idle read RuntimeError mid-iteration under the
    un-locked detach, and the metadata backfill could rewrite a unit
    mid-read.  The session's ``remove_probe`` SAE branch routes through
    the atomic ``try_detach`` and blocks while the lock is held."""
    session = make_tiny_session(warm=False)
    _load_mock(session, monkeypatch)
    inst = session.sae
    _attach(inst, "sae/1", 1)

    # Idle _measurement_specs is a per-call copy, not the live dict.
    snapshot = inst._measurement_specs()
    assert snapshot is not inst.probes
    assert inst.try_detach("sae/1") is True
    assert "sae/1" in snapshot  # the copy is immune to the detach
    assert inst.try_detach("sae/1") is False

    # And the removal path serializes on the registry lock.
    _attach(inst, "sae/1", 1)
    entered = threading.Event()

    def _detach() -> None:
        entered.set()
        session.remove_probe("sae/1")

    with inst.state_lock:
        detacher = threading.Thread(target=_detach)
        detacher.start()
        assert entered.wait(timeout=5.0)
        detacher.join(timeout=0.2)
        assert detacher.is_alive()  # blocked on the registry lock
        assert "sae/1" in inst.probes
    detacher.join(timeout=5.0)
    assert not detacher.is_alive()
    assert "sae/1" not in inst.probes


def test_bound_run_freezes_sae_unit_against_metadata_backfill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The InstrumentBinding snapshot: a Neuronpedia backfill landing
    mid-generation (it mutates attached specs + the meta cache WITHOUT the
    generation lock) must not change a running generation's strength unit.
    Between generations (idle run) the refresh applies immediately, as
    before."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.sae import neuronpedia

    inst = _instrument()
    # Attached before any metadata exists: the unit is raw at bind time.
    _attach(inst, "sae/1", 1)
    acts = inst.runtime.encode(inst._session._capture.latest_per_layer()[1])
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    inst.bind(inst.plan(prep), prep)
    before = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert before["sae/1"] == pytest.approx(3.0)  # raw

    # The backfill lands mid-generation.
    monkeypatch.setattr(
        neuronpedia, "fetch_feature",
        lambda *_a, **_k: {"label": "one", "max_act": 4.0, "checked": True},
    )
    inst.fetch_meta([1])
    assert inst.probes["sae/1"]["max_act"] == 4.0
    during = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert during["sae/1"] == pytest.approx(3.0)  # unit frozen at bind

    inst.close_run()  # generation boundary
    after = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert after["sae/1"] == pytest.approx(3.0 / 4.0)  # refresh now applies


def test_sae_bind_resolves_unit_from_meta_cache() -> None:
    """Bind-time resolution: a spec whose ``max_act`` is unset but whose
    unit exists in the metadata cache freezes the RESOLVED unit — the
    live-cache fallback never runs under a bound run, so a mid-generation
    cache mutation cannot flip the unit either."""
    inst = _instrument()
    # Feature 2's unit (10.0) is already in the fixture's meta cache; the
    # spec itself carries no max_act (validate-time fetch never ran).
    _attach(inst, "sae/2", 2)
    acts = inst.runtime.encode(inst._session._capture.latest_per_layer()[1])
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    inst.bind(inst.plan(prep), prep)
    assert inst.current_run.binding.specs["sae/2"]["max_act"] == 10.0
    values = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert values["sae/2"] == pytest.approx(5.0 / 10.0)

    # A mid-generation cache mutation is invisible to the bound run.
    inst.runtime.resident.feature_meta["2"]["max_act"] = 2.0
    values = {n: v for n, _f, _r, v in inst.probe_values(acts)}
    assert values["sae/2"] == pytest.approx(5.0 / 10.0)


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("step_id", [-1, 7])
def test_observe_never_memoizes(bound: bool, step_id: int) -> None:
    """Per-forward reuse is the step-keyed encode stash, not a reading memo:
    repeated observations of one step rescore, bound or idle (an idle run
    persists indefinitely, and a repeated step with different hidden states
    must read fresh)."""
    inst = _instrument()
    _attach(inst, "sae/1", 1)
    if bound:
        prep = inst.prepare(ReadRequest(final_aggregate=True))
        inst.bind(inst.plan(prep), prep)
    run = inst.current_run
    assert run.bound is bound
    first = run.observe(step_id, {1: torch.tensor([0.2, 3.0, 5.0, 1.0])})
    second = run.observe(step_id, {1: torch.tensor([0.2, 8.0, 5.0, 1.0])})
    assert first["sae/1"].value == 3.0
    assert second["sae/1"].value == 8.0
    inst.close_run()


def test_detach_during_bound_generation_keeps_aggregate_roster() -> None:
    """Mutations apply next generation: a probe detached mid-generation
    (e.g. the synchronous DELETE route, which takes no generation lock)
    stays in the bound generation's aggregate roster; the next bind sees
    the removal."""
    inst = _instrument()
    _attach(inst, "sae/1", 1)
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    inst.bind(inst.plan(prep), prep)
    pooled = inst._session._capture.latest_per_layer()

    del inst.probes["sae/1"]  # the un-locked detach lands mid-generation

    readings = inst.score_aggregate([1], pooled=pooled)
    assert "sae/1" in readings  # frozen roster still measures it

    inst.close_run()
    readings = inst.score_aggregate([1], pooled=pooled)
    assert readings == {}  # the removal applies at the next boundary


def test_score_probes_entries_are_disjoint() -> None:
    """Two entries, no placeholder argument: ``score_probes`` takes capture
    slices and encodes them, ``score_probes_from_activations`` takes this
    forward's encode.  Both land the same reading."""
    inst = _instrument()
    _attach(inst, "sae/2", 2, max_act=10.0)
    hidden = inst._session._capture.latest_per_layer()
    acts = inst.runtime.encode(hidden[1])

    from_slices = inst.score_probes(hidden)
    from_acts = inst.score_probes_from_activations(acts)
    assert from_slices["sae/2"].value == pytest.approx(
        from_acts["sae/2"].value,
    )
    # A layer the resident hook does not cover reads nothing.
    assert inst.score_probes({7: hidden[1]}) == {}


def test_bound_run_reads_its_bind_time_live_snapshot() -> None:
    """The lens's live-state discipline, mirrored: a bound run reads the
    live-discovery config its generation started with, so a toggle from
    another thread cannot flip a running generation mid-stream.  Idle runs
    pass the current config through."""
    inst = _instrument()
    inst.enable_live()
    live_at_bind = dict(inst.live or {})
    assert live_at_bind

    prep = inst.prepare(ReadRequest(final_aggregate=True, live=True))
    inst.bind(inst.plan(prep), prep)
    assert inst._measurement_live() == live_at_bind

    # A concurrent library caller flips the toggle mid-generation.
    inst.disable_live()
    assert inst.live is None
    assert inst._measurement_live() == live_at_bind
    assert inst.live_readout_step(top_k=2, step_id=0) is not None

    inst.close_run()  # generation boundary: the idle run passes through
    assert inst._measurement_live() is None
    assert inst.live_readout_step(top_k=2, step_id=0) is None


# ------------------------------------------------- session integration


def test_load_swap_and_unload_drop_everything_keyed_by_the_release(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_tiny_session(warm=False)
    _load_mock(session, monkeypatch)
    assert session.sae.info["release"] == "mock-release"
    assert session.sae.info["layer"] == 1
    assert session.sae.register_direction(2) == "sae/2"
    assert "sae/2" in session.profiles
    name = session.add_probe("sae/1")
    digest = session.probe_hashes()[name]

    _load_mock(session, monkeypatch, release="other-release")
    assert "sae/2" not in session.profiles
    assert session.sae.names == []
    # The re-attached probe hashes against the new release, not a cached digest.
    session.add_probe("sae/1")
    assert session.probe_hashes()[name] != digest

    session.sae.unload()
    assert session.sae.info is None
    assert session.sae.names == []
    with pytest.raises(RuntimeError, match="no SAE loaded"):
        session.sae.register_direction(2)


def test_steering_an_sae_atom_never_fetches_metadata(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1: the first ``sae/<id>`` steer resolves inside composition while the
    exclusive-GPU lock is held, and used to fetch the feature's Neuronpedia
    metadata there (a 10 s timeout).  A direction needs only the decoder row:
    steering an uncached feature makes no request, and the probe path that
    does fetch runs without the lock."""
    from drowse.sae import neuronpedia

    session, _script = make_tiny_session.scripted("Hi.", warm=False)
    _load_mock(session, monkeypatch, neuronpedia_id="tiny/1-res")
    fetches: list[tuple[int, bool]] = []

    def recording_fetch(np_id: Any, idx: int, **_kw: Any) -> dict[str, Any]:
        assert np_id == "tiny/1-res"
        fetches.append((idx, _lock_held_elsewhere(session._gen_lock)))
        return {"label": "f", "max_act": 2.0, "checked": True}

    monkeypatch.setattr(neuronpedia, "fetch_feature", recording_fetch)
    result = session.generate(
        "Say hi.", steering="0.3 sae/5", sampling=SamplingConfig(max_tokens=2),
    )
    assert "sae/5" in result.first.applied_steering
    assert "sae/5" in session.profiles
    assert fetches == []

    session.add_probe("sae/6")
    assert fetches == [(6, False)]


def test_validation_fetch_refreshes_an_attached_probe(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata arriving through a single-feature validation reaches attached
    probes the way the batch backfill does: the spec takes the label and the
    strength unit, and the probe's cached digest (which folds the unit) is
    recomputed instead of going stale."""
    from drowse.sae import neuronpedia

    session = make_tiny_session(warm=False)
    _load_mock(session, monkeypatch, neuronpedia_id="tiny/1-res")
    answers: list[dict[str, Any] | None] = [None]  # attach-time fetch fails
    monkeypatch.setattr(
        neuronpedia, "fetch_feature", lambda *_a, **_k: answers.pop(0),
    )
    session.add_probe("sae/3")
    assert session.sae.specs()["sae/3"]["max_act"] is None
    digest = session.probe_hashes()["sae/3"]

    answers.append({"label": "three", "max_act": 4.0, "checked": True})
    assert session.sae.validate_feature(3)["max_act"] == 4.0
    spec = session.sae.specs()["sae/3"]
    assert (spec["label"], spec["max_act"]) == ("three", 4.0)
    assert session.probe_hashes()["sae/3"] != digest


def test_token_readout_replays_the_producing_forward(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.loom.errors import InvalidNodeOperationError

    session, _script = make_tiny_session.scripted("Hi there.", warm=False)
    _load_mock(session, monkeypatch)
    session.generate("Say hi.", sampling=SamplingConfig(max_tokens=3))
    node = session.tree.get(session.tree.active_node_id)

    envelope = session.sae.token_readout(node.id, 1, top_k=4)["measurements"]
    sae = envelope["instruments"]["sae"]
    assert sae["binding"]["layer"] == 1
    assert sae["binding"]["source"] == "saelens:mock-release"
    features = sae["readout"]["features"]
    assert len(features) == 4
    activations = [row["activation"] for row in features]
    assert activations == sorted(activations, reverse=True)

    # Same anchor, same forward: the replay is deterministic.
    again = session.sae.token_readout(node.id, 1, top_k=4)["measurements"]
    assert again["instruments"]["sae"]["readout"] == sae["readout"]

    with pytest.raises(InvalidNodeOperationError, match="out of range"):
        session.sae.token_readout(node.id, 99)
    with pytest.raises(ValueError, match="top_k"):
        session.sae.token_readout(node.id, 0, top_k=0)


def test_close_releases_the_resident_dictionary(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = make_tiny_session(warm=False)
    _load_mock(session, monkeypatch)
    session.add_probe("sae/1")
    session.close()
    assert session.sae.runtime.resident is None
    assert session.sae.names == []
