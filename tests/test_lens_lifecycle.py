"""The lens fit lifecycle: the pure planner and the executor.

Planner tests drive :func:`~drowse.lens.lifecycle.plan_lens_fit` with
hand-built sidecars.  Executor tests fit the toy lens model through a real
:class:`~drowse.lens.instrument.LensInstrument`
(:class:`tests.support.lens.LensHarness`) against an isolated store: resume,
top-up, no-op, checkpoint recovery and resident-lens handling.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.lens.jlens import JacobianLens, JacobianLensError
from drowse.lens.lifecycle import (
    CORPUS_HASH_KIND,
    LensFitCorpus,
    LensFitIdentity,
    LensFitPlan,
    LensFitter,
    ResidentLens,
    fit_covers_layers,
    plan_lens_fit,
    raw_corpus_sha256,
    resolve_fit_sources,
)
from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.lens.store import (
    LensProvenance,
    lens_estimator_policy,
    lens_identity,
    lens_manifest_path,
    load_lens,
    load_lens_checkpoint,
    save_lens_checkpoint,
)
from tests.support.jlens_toys import CharTokenizer
from tests.support.lens import LENS_PROMPTS as _PROMPTS
from tests.support.lens import TOY_MODEL_ID as _MODEL_ID
from tests.support.lens import LensHarness

pytestmark = pytest.mark.usefixtures("isolated_home")


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

_POLICY = lens_estimator_policy("J")


def _identity(
    rows: list[list[int]], *, sources: tuple[int, ...] = (0, 1),
) -> LensFitIdentity:
    corpus = LensFitCorpus(
        prompts=tuple(f"doc {i}" for i in range(len(rows))),
        token_rows=tuple(tuple(row) for row in rows),
        raw_sha256="raw",
        raw_count=len(rows),
    )
    return LensFitIdentity(
        kind="J", corpus=corpus, sources=sources, seq_len=128,
        model_fingerprint="fp", policy=_POLICY,
    )


def _sidecar(
    identity: LensFitIdentity,
    *,
    n: int,
    layers: list[int],
    checkpoint: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    corpus = identity.corpus
    sidecar: dict[str, Any] = {
        "n_prompts": n,
        "corpus_hash_kind": CORPUS_HASH_KIND,
        "seq_len": identity.seq_len,
        "model_fingerprint": identity.model_fingerprint,
        "estimator_policy": dict(identity.policy),
        "sites": {"out": {"layers": layers}},
    }
    if checkpoint:
        sidecar["corpus_sha256"] = corpus.sha256
        sidecar["consumed_prefix_sha256"] = corpus.prefix_sha256(n)
    else:
        sidecar["corpus_sha256"] = corpus.prefix_sha256(n)
    sidecar.update(overrides)
    return sidecar


_ROWS = [[1] * 20, [2] * 20, [3] * 20, [4] * 20]


def test_plan_force_is_always_fresh() -> None:
    identity = _identity(_ROWS)
    durable = _sidecar(identity, n=4, layers=[0, 1])
    plan = plan_lens_fit(
        identity, durable=durable, checkpoint=None, resident=None, force=True,
    )
    assert plan == LensFitPlan("fresh")


def test_plan_nothing_reusable_is_fresh() -> None:
    identity = _identity(_ROWS)
    other = _identity([[9] * 20] * 4)
    assert plan_lens_fit(
        identity, durable=None, checkpoint=None, resident=None,
    ) == LensFitPlan("fresh")
    # A durable fit over a different corpus, or another estimator, is ignored.
    assert plan_lens_fit(
        identity, durable=_sidecar(other, n=4, layers=[0, 1]),
        checkpoint=None, resident=None,
    ) == LensFitPlan("fresh")
    foreign = _sidecar(identity, n=4, layers=[0, 1], estimator_policy={"x": 1})
    assert plan_lens_fit(
        identity, durable=foreign, checkpoint=None, resident=None,
    ) == LensFitPlan("fresh")


def test_plan_noop_prefers_the_matching_resident() -> None:
    identity = _identity(_ROWS)
    durable = _sidecar(identity, n=4, layers=[0, 1])
    resident = ResidentLens(
        n_prompts=4, layers=frozenset({0, 1}), identity=lens_identity(durable),
    )
    assert plan_lens_fit(
        identity, durable=durable, checkpoint=None, resident=resident,
    ) == LensFitPlan("noop", durable="resident")
    stale = ResidentLens(n_prompts=4, layers=frozenset({0, 1}), identity="old")
    assert plan_lens_fit(
        identity, durable=durable, checkpoint=None, resident=stale,
    ) == LensFitPlan("noop", durable="selective")


def test_plan_tops_up_missing_layers_and_their_checkpoint() -> None:
    identity = _identity(_ROWS, sources=(0, 1, 2))
    durable = _sidecar(identity, n=4, layers=[0, 1])
    plan = plan_lens_fit(identity, durable=durable, checkpoint=None, resident=None)
    assert plan == LensFitPlan("topup", durable="full", missing=(2,))
    partial = _sidecar(identity, n=2, layers=[2], checkpoint=True)
    plan = plan_lens_fit(identity, durable=durable, checkpoint=partial, resident=None)
    assert plan == LensFitPlan("topup", durable="full", missing=(2,), checkpoint=True)


def test_plan_resumes_from_a_durable_prefix_or_a_farther_checkpoint() -> None:
    identity = _identity(_ROWS)
    durable = _sidecar(identity, n=2, layers=[0, 1])
    assert plan_lens_fit(
        identity, durable=durable, checkpoint=None, resident=None,
    ) == LensFitPlan("resume", durable="full")
    farther = _sidecar(identity, n=3, layers=[0, 1], checkpoint=True)
    assert plan_lens_fit(
        identity, durable=durable, checkpoint=farther, resident=None,
    ) == LensFitPlan("resume", checkpoint=True)
    behind = _sidecar(identity, n=1, layers=[0, 1], checkpoint=True)
    assert plan_lens_fit(
        identity, durable=durable, checkpoint=behind, resident=None,
    ) == LensFitPlan("resume", durable="full")
    assert plan_lens_fit(
        identity, durable=None, checkpoint=farther, resident=None,
    ) == LensFitPlan("resume", checkpoint=True)


def test_plan_skips_a_durable_artifact_that_cannot_be_a_base() -> None:
    """A prefix durable fit missing requested layers is neither a base nor
    a no-op, so its payload is never loaded."""
    identity = _identity(_ROWS, sources=(0, 1))
    durable = _sidecar(identity, n=2, layers=[0])
    assert plan_lens_fit(
        identity, durable=durable, checkpoint=None, resident=None,
    ) == LensFitPlan("fresh")


def test_plan_refuses_to_extend_a_subset_of_durable_layers() -> None:
    identity = _identity(_ROWS, sources=(0,))
    durable = _sidecar(identity, n=2, layers=[0, 1])
    with pytest.raises(JacobianLensError, match="full durable layer set"):
        plan_lens_fit(identity, durable=durable, checkpoint=None, resident=None)


def test_checkpoint_must_cover_exactly_the_requested_layers() -> None:
    identity = _identity(_ROWS)
    wrong_layers = _sidecar(identity, n=3, layers=[0], checkpoint=True)
    assert not identity.checkpoint_matches(wrong_layers, identity.sources)
    empty = _sidecar(identity, n=0, layers=[0, 1], checkpoint=True)
    assert not identity.checkpoint_matches(empty, identity.sources)


def test_corpus_identity_hashes_token_rows_as_lists() -> None:
    """The on-disk identity is sha256(repr(list-of-lists)); tuples would
    change every recorded digest."""
    identity = _identity(_ROWS)
    expected = hashlib.sha256(repr(_ROWS[:2]).encode("utf-8")).hexdigest()
    assert identity.corpus.prefix_sha256(2) == expected
    assert identity.corpus.sha256 == hashlib.sha256(
        repr(_ROWS).encode("utf-8"),
    ).hexdigest()
    assert raw_corpus_sha256(["a", "b"]) == hashlib.sha256(
        repr(["a", "b"]).encode("utf-8"),
    ).hexdigest()


def test_fit_sources_refuse_sample_and_drop_band() -> None:
    assert resolve_fit_sources(None, n_layers=5) == [0, 1, 2, 3]
    assert resolve_fit_sources([3, 1, 1], n_layers=5) == [1, 3]
    assert resolve_fit_sources("workspace", n_layers=11) == [4, 5, 6, 7, 8, 9]
    for mode in ("sample", "band"):
        with pytest.raises(ValueError, match="unknown layer mode"):
            resolve_fit_sources(mode, n_layers=5)


def test_fit_covers_layers_resolves_modes_against_the_recorded_depth() -> None:
    sidecar = {"sites": {"out": {"layers": [0, 1, 2]}}, "model_layer_count": 4}
    assert fit_covers_layers(sidecar, None)
    assert fit_covers_layers(sidecar, "all")
    assert fit_covers_layers(sidecar, [1, 2])
    assert not fit_covers_layers(sidecar, [3])
    assert not fit_covers_layers({**sidecar, "model_layer_count": None}, "all")


def test_fit_rejects_an_unfittable_kind() -> None:
    with pytest.raises(ValueError, match="lens kind"):
        LensHarness().lens.fit(_PROMPTS, kind="K")


def test_fit_uses_the_kind_directory_and_policy() -> None:
    harness = LensHarness()
    fitted = harness.lens.fit(_PROMPTS, kind="J")
    assert fitted.key.kind == "J"
    assert lens_manifest_path(_MODEL_ID, "j").exists()
    assert not lens_manifest_path(_MODEL_ID, "r").exists()


def test_fit_is_noop_proves_a_pinned_corpus_fit_before_streaming() -> None:
    """The pre-stream proof the fit route runs: a matching durable fit of
    the same pinned corpus, prompt count, layers and exact model source."""
    harness = LensHarness()
    # Stamped the way the loader stamps it (drowse.model.load).
    setattr(harness.model, "_drowse_source_fingerprint", "toy-source")
    spec = "hf:org/data/cfg@" + "a" * 40 + ";preprocess=1;doc_chars=4000"
    assert harness.lens.fit_is_noop(len(_PROMPTS), kind="J", corpus_spec=spec) is None
    harness.lens.fit(_PROMPTS, kind="J", corpus_spec=spec, source_layers=[0, 1])

    proven = harness.lens.fit_is_noop(
        len(_PROMPTS), kind="J", corpus_spec=spec, source_layers=[1],
    )
    assert proven is not None and proven["corpus_spec"] == spec
    for kwargs in (
        {"kind": "R"},
        {"corpus_spec": spec.replace("cfg", "other")},
        {"source_layers": [2]},
        {"seq_len": 64},
    ):
        call = {"kind": "J", "corpus_spec": spec, "source_layers": [1], **kwargs}
        assert harness.lens.fit_is_noop(len(_PROMPTS), **call) is None
    assert harness.lens.fit_is_noop(
        len(_PROMPTS) + 1, kind="J", corpus_spec=spec, source_layers=[1],
    ) is None
    setattr(harness.model, "_drowse_source_fingerprint", "other-source")
    assert harness.lens.fit_is_noop(
        len(_PROMPTS), kind="J", corpus_spec=spec, source_layers=[1],
    ) is None


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------

def _save_checkpoint(
    partial: JacobianLens, model_id: str, **kwargs: Any,
) -> Path:
    """Publish a self-contained current checkpoint from estimator sums."""
    sums = {
        layer: matrix * partial.n_prompts
        for layer, matrix in partial.jacobians.items()
    }
    consumed = kwargs.pop(
        "consumed_prefix_sha256",
        hashlib.sha256(json.dumps({
            "corpus_sha256": kwargs.get("corpus_sha256"),
            "n_prompts": partial.n_prompts,
        }, sort_keys=True).encode()).hexdigest(),
    )
    return save_lens_checkpoint(
        sums, partial.n_prompts, partial.d_model, model_id,
        LensProvenance(**kwargs), kind=partial.key.kind,
        consumed_prefix_sha256=consumed,
    )


def _first_shard(model_id: str, *, checkpoint: bool = False) -> Path:
    """The lowest layer's ``out`` shard named by the local J-lens pointer."""
    pointer = lens_manifest_path(model_id, "j", checkpoint=checkpoint)
    site = json.loads(pointer.read_text())["sites"]["out"]
    first = min(site["layers"])
    return pointer.parent / site["tensor_files"][str(first)]


class _CountingTokenizer(CharTokenizer):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        self.calls += 1
        return super().__call__(text, return_tensors=return_tensors)


def test_fit_persists_and_property_loads() -> None:
    s = LensHarness()
    fitted = s.lens.fit(_PROMPTS, corpus_spec="test")
    assert fitted.n_prompts == len(_PROMPTS)

    on_disk = load_lens(_MODEL_ID)
    assert on_disk is not None
    lens, sidecar = on_disk
    assert lens.n_prompts == len(_PROMPTS)
    assert sidecar["corpus_spec"] == "test"

    fresh = LensHarness()  # new stub: property must lazy-load from disk
    assert fresh.lens.resident() is not None
    assert fresh.lens.resident().n_prompts == len(_PROMPTS)


def test_terminal_checkpoint_is_promoted_without_second_tensor_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io
    from drowse.base import integrity
    real_save = lens_io._save_fp32_square_safetensors_atomic
    writes = 0
    hashes = 0

    def _counting_save(*args: Any, **kwargs: Any) -> Any:
        nonlocal writes
        writes += 1
        return real_save(*args, **kwargs)

    monkeypatch.setattr(
        lens_io, "_save_fp32_square_safetensors_atomic", _counting_save,
    )
    real_hash = integrity.hash_file

    def _counting_hash(path: Path) -> str:
        nonlocal hashes
        hashes += 1
        return real_hash(path)

    monkeypatch.setattr(integrity, "hash_file", _counting_hash)
    fitted = LensHarness().lens.fit(
        _PROMPTS[:2], force=True, checkpoint_every=2,
    )

    assert fitted.n_prompts == 2
    # One checkpoint shard per fitted layer; promotion only switches the
    # durable sidecar pointer and performs no second serialization pass.
    assert writes == len(fitted.source_layers)
    assert hashes == 0
    assert load_lens(_MODEL_ID) is not None
    assert load_lens_checkpoint(_MODEL_ID, "j") is None


def test_refit_rebuilds_live_lens_probes_and_evicts_directions() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS, source_layers=[0])
    s.lens.enable_live(layers=[0])
    live = s.lens.live
    assert live is not None
    old_stack = live["J_stack"]
    s.registry.put_profile("jlens/a", {0: torch.ones(4)})
    s.lens.probes["jlens/a"] = {
        "word": "a", "token_id": 1, "layers": [0],
    }

    fitted = s.lens.fit(_PROMPTS, source_layers=[1], force=True)

    assert fitted.source_layers == [1]
    live = s.lens.live
    assert live is not None
    assert live["layers"] == [1]
    assert live["uses_all_layers"] is True
    assert live["J_stack"] is not old_stack
    assert s.lens.probes["jlens/a"]["layers"] == [1]
    assert "jlens/a" not in s.profiles


def test_fit_already_done_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = LensHarness()
    first = s.lens.fit(_PROMPTS)
    messages: list[str] = []
    import drowse.lens.store as lens_io

    monkeypatch.setattr(
        lens_io, "load_lens",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("same-session no-op should reuse the resident lens")
        ),
    )
    again = s.lens.fit(_PROMPTS, on_progress=messages.append)
    assert any("nothing to do" in m for m in messages)
    for layer in first.source_layers:
        assert torch.equal(first.jacobians[layer], again.jacobians[layer])


def test_subset_noop_keeps_full_durable_lens_resident() -> None:
    s = LensHarness()
    full = s.lens.fit(_PROMPTS, source_layers=[0, 1], force=True)

    selected = s.lens.fit(_PROMPTS, source_layers=[1])

    assert selected.source_layers == [1]
    assert full.source_layers == [0, 1]
    assert s.lens.resident_lens.source_layers == [0, 1]
    assert s.lens.resident().source_layers == [0, 1]


def test_fresh_subset_noop_reads_only_requested_shard_and_preserves_disk_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    LensHarness().lens.fit(_PROMPTS, source_layers=[0, 1], force=True)
    payload_reads: list[str] = []
    real_safe_open = lens_io.safe_open

    class _TrackingSafeOpen:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._opened = real_safe_open(*args, **kwargs)
            self._tensors: Any = None

        def __enter__(self) -> "_TrackingSafeOpen":
            self._tensors = self._opened.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._opened.__exit__(*args)

        def keys(self) -> Any:
            return self._tensors.keys()

        def get_slice(self, key: str) -> Any:
            return self._tensors.get_slice(key)

        def get_tensor(self, key: str) -> Any:
            payload_reads.append(key)
            return self._tensors.get_tensor(key)

    monkeypatch.setattr(lens_io, "safe_open", _TrackingSafeOpen)
    session = LensHarness()
    selected = session.lens.fit(_PROMPTS, source_layers=[1])

    assert payload_reads == ["layer_1"]
    assert selected.source_layers == [1]
    monkeypatch.setattr(lens_io, "safe_open", real_safe_open)
    durable = load_lens(_MODEL_ID)
    assert durable is not None
    assert durable[0].source_layers == [0, 1]
    assert session.lens.resident().source_layers == [0, 1]


def test_fresh_partial_extension_rejects_before_payload_and_preserves_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    import drowse.lens.store as lens_io

    writer = LensHarness()
    writer.lens.fit(_PROMPTS[:2], source_layers=[0, 1], force=True)
    resident_ref = weakref.ref(writer.lens.resident_lens)
    del writer
    gc.collect()
    assert resident_ref() is None

    payload_reads: list[str] = []
    real_load = lens_io._load_lens_verified

    def record_load(*args: Any, **kwargs: Any) -> Any:
        payload_reads.append("load")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(lens_io, "_load_lens_verified", record_load)
    with pytest.raises(JacobianLensError, match="full durable layer set"):
        LensHarness().lens.fit(_PROMPTS, source_layers=[0])

    assert payload_reads == []
    monkeypatch.setattr(lens_io, "_load_lens_verified", real_load)
    durable = load_lens(_MODEL_ID)
    assert durable is not None
    assert durable[0].source_layers == [0, 1]
    assert durable[0].n_prompts == 2


def test_overlapping_partial_extension_requires_explicit_replacement() -> None:
    LensHarness(n_layers=4).lens.fit(
        _PROMPTS[:2], source_layers=[0, 1], force=True,
    )

    with pytest.raises(JacobianLensError, match="full durable layer set"):
        LensHarness(n_layers=4).lens.fit(
            _PROMPTS, source_layers=[1, 2],
        )

    durable = load_lens(_MODEL_ID)
    assert durable is not None
    assert durable[0].source_layers == [0, 1]
    assert durable[0].n_prompts == 2

    replaced = LensHarness(n_layers=4).lens.fit(
        _PROMPTS, source_layers=[1, 2], force=True,
    )
    assert replaced.source_layers == [1, 2]
    durable = load_lens(_MODEL_ID)
    assert durable is not None
    assert durable[0].source_layers == [1, 2]
    assert durable[0].n_prompts == len(_PROMPTS)


def test_exact_noop_reaps_checkpoint_left_after_final_publication() -> None:
    s = LensHarness()
    full = s.lens.fit(_PROMPTS, force=True)
    loaded = load_lens(_MODEL_ID)
    assert loaded is not None
    sidecar = loaded[1]
    _save_checkpoint(
        full, _MODEL_ID,
        corpus_spec=str(sidecar["corpus_spec"]),
        corpus_sha256=str(sidecar["corpus_sha256"]),
        corpus_hash_kind=str(sidecar["corpus_hash_kind"]),
        seq_len=int(sidecar["seq_len"]),
        dim_batch=int(sidecar["dim_batch"]),
        skip_first=int(sidecar["estimator_policy"]["skip_first_positions"]),
        model_fingerprint=str(sidecar["model_fingerprint"]),
    )
    assert load_lens_checkpoint(_MODEL_ID, "j") is not None

    selected = s.lens.fit(_PROMPTS)

    assert selected.source_layers == full.source_layers
    assert load_lens_checkpoint(_MODEL_ID, "j") is None


def test_resident_noop_does_not_reap_checkpoint_when_final_payload_corrupt() -> None:
    s = LensHarness()
    full = s.lens.fit(_PROMPTS, force=True)
    loaded = load_lens(_MODEL_ID)
    assert loaded is not None
    sidecar = loaded[1]
    _save_checkpoint(
        full, _MODEL_ID,
        corpus_spec=str(sidecar["corpus_spec"]),
        corpus_sha256=str(sidecar["corpus_sha256"]),
        corpus_hash_kind=str(sidecar["corpus_hash_kind"]),
        seq_len=int(sidecar["seq_len"]),
        dim_batch=int(sidecar["dim_batch"]),
        skip_first=int(sidecar["estimator_policy"]["skip_first_positions"]),
        model_fingerprint=str(sidecar["model_fingerprint"]),
    )
    final_tensor = _first_shard(_MODEL_ID)
    payload = bytearray(final_tensor.read_bytes())
    payload[-1] ^= 1
    final_tensor.write_bytes(payload)

    # The resident lens makes the fit a no-op, but it is not proof that the
    # current on-disk payload is sound enough to discard the recovery point.
    selected = s.lens.fit(_PROMPTS)

    assert selected.source_layers == full.source_layers
    assert load_lens_checkpoint(_MODEL_ID, "j") is not None


def test_overlapping_topup_keeps_preserved_extra_layer_resident() -> None:
    s = LensHarness(n_layers=4)
    s.lens.fit(_PROMPTS, source_layers=[0, 1], force=True)

    selected = s.lens.fit(_PROMPTS, source_layers=[1, 2])

    assert selected.source_layers == [1, 2]
    assert s.lens.resident_lens.source_layers == [0, 1, 2]
    assert s.lens.resident().source_layers == [0, 1, 2]


def test_resident_lens_is_not_reused_after_external_artifact_replacement() -> None:
    corpus_b = [f"replacement corpus prompt {i} with enough content" for i in range(4)]
    session_a = LensHarness()
    resident_a = session_a.lens.fit(_PROMPTS, force=True)
    disk_b = LensHarness().lens.fit(corpus_b, force=True)
    assert any(
        not torch.equal(resident_a.jacobians[layer], disk_b.jacobians[layer])
        for layer in resident_a.source_layers
    )

    refreshed = session_a.lens.fit(corpus_b)

    for layer in disk_b.source_layers:
        assert torch.allclose(
            refreshed.jacobians[layer], disk_b.jacobians[layer], atol=2e-3,
        )


def test_fit_serializes_complete_cross_session_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered: list[str] = []
    first_entered = threading.Event()
    release_first = threading.Event()

    def _transaction(self: LensFitter, prompts: object, *args: object, **kwargs: object) -> str:
        del prompts, args, kwargs
        entered.append(self._model_id)
        if len(entered) == 1:
            first_entered.set()
            assert release_first.wait(timeout=1.0)
        return "done"

    monkeypatch.setattr(LensFitter, "run", _transaction)
    first = LensHarness()
    second = LensHarness()
    results: list[str] = []
    thread_a = threading.Thread(target=lambda: results.append(first.lens.fit(_PROMPTS)))
    thread_b = threading.Thread(target=lambda: results.append(second.lens.fit(_PROMPTS)))
    thread_a.start()
    assert first_entered.wait(timeout=1.0)
    thread_b.start()
    assert len(entered) == 1
    release_first.set()
    thread_a.join(timeout=1.0)
    thread_b.join(timeout=1.0)
    assert len(entered) == 2
    assert results == ["done", "done"]


def test_fresh_fit_does_not_promote_stale_incompatible_checkpoint() -> None:
    prompts = _PROMPTS[:2]
    tokenizer = CharTokenizer()
    consumed = [
        [int(tok) for tok in tokenizer(
            prompt, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for prompt in prompts
    ]
    corpus_sha = hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest()
    stale = JacobianLens(
        {layer: torch.full((6, 6), 99.0) for layer in (0, 1)},
        n_prompts=2, d_model=6,
    )
    _save_checkpoint(
        stale, _MODEL_ID,
        corpus_spec="stale",
        corpus_sha256=corpus_sha,
        corpus_hash_kind="token_ids_v1",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_fingerprint="WRONG",
    )

    fitted = LensHarness().lens.fit(prompts, checkpoint_every=25)
    durable = load_lens(_MODEL_ID)

    assert durable is not None
    assert durable[1]["model_fingerprint"] != "WRONG"
    for layer in fitted.source_layers:
        assert not torch.equal(durable[0].jacobians[layer], stale.jacobians[layer])


def test_fit_noop_revalidates_token_ids_before_loading_tensor() -> None:
    tokenizer = _CountingTokenizer()
    s = LensHarness(tokenizer=tokenizer)
    s.lens.fit(_PROMPTS)
    tokenizer.calls = 0

    again = s.lens.fit(_PROMPTS, source_layers=[1])

    assert again.source_layers == [1]
    assert tokenizer.calls == len(_PROMPTS)


def test_fit_changed_tokenizer_invalidates_cache() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    original = load_lens(_MODEL_ID)
    assert original is not None

    class _ShiftedTokenizer(CharTokenizer):
        def __call__(
            self, text: str, return_tensors: str = "pt",
        ) -> dict[str, torch.Tensor]:
            result = super().__call__(text, return_tensors=return_tensors)
            result["input_ids"] = (result["input_ids"] + 1) % 13
            return result

    LensHarness(tokenizer=_ShiftedTokenizer()).lens.fit(_PROMPTS)
    changed = load_lens(_MODEL_ID)
    assert changed is not None
    assert changed[1]["corpus_sha256"] != original[1]["corpus_sha256"]


def test_fit_resumes_from_partial_and_matches_full_fit() -> None:
    s = LensHarness()
    full = s.lens.fit(_PROMPTS, force=True)

    # A normal smaller-corpus artifact carries the hash of that actual prefix.
    # Extending 2 -> N must recognize it without the old test-only trick of
    # stamping the prefix tensor with the future full-corpus hash.
    partial = LensHarness()
    partial.lens.fit(_PROMPTS[:2], force=True)

    resumed_session = LensHarness()
    messages: list[str] = []
    resumed = resumed_session.lens.fit(_PROMPTS, on_progress=messages.append)
    assert any("resuming from 2 prompts" in m for m in messages)
    assert any("prompt 4/4" in m for m in messages)
    assert resumed.n_prompts == len(_PROMPTS)
    # The resume base round-trips losslessly through the fp32 artifact. Allow
    # only ordinary accumulation-order noise against the from-scratch fit.
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=1e-6,
        ), f"layer {layer}: resumed fit diverges from the from-scratch fit"


def test_fit_tops_up_a_hub_corpus_by_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A larger read of the same pinned Hub corpus fits only the new rows."""
    import datasets

    import drowse.lens.fit as jlens_module
    from drowse.lens.corpus import HfLensCorpus, _collect_lens_corpus
    from drowse.lens.store import load_lens_sidecar

    rows = [{"text": ""}, *({"text": p} for p in _PROMPTS)]
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *_a, **_k: iter(rows),
    )
    corpus = HfLensCorpus("org/shuffled", "default", "a" * 40)
    first_docs, spec = _collect_lens_corpus(2, corpus)
    LensHarness().lens.fit(first_docs, corpus_spec=spec, force=True)

    real_fit = jlens_module.fit_jacobian_lens
    fitted_widths: list[int] = []

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        fitted_widths.append(len(args[2]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "fit_jacobian_lens", counted_fit)
    docs, grown_spec = _collect_lens_corpus(4, corpus)
    assert grown_spec == spec and docs == _PROMPTS
    messages: list[str] = []
    grown = LensHarness().lens.fit(
        docs, corpus_spec=spec, on_progress=messages.append,
    )
    assert any("resuming from 2 prompts" in m for m in messages)
    assert fitted_widths == [2]
    assert grown.n_prompts == len(_PROMPTS)
    sidecar = load_lens_sidecar(_MODEL_ID)
    assert sidecar is not None and sidecar["corpus_spec"] == spec


def test_fit_tops_up_a_corpus_mix_by_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Growing a fit on a weighted mix fits only the newly scheduled rows."""
    import datasets

    import drowse.lens.fit as jlens_module
    from drowse.lens.corpus import (
        HfLensCorpus,
        LensCorpusMix,
        _collect_lens_corpus,
    )
    from drowse.lens.store import load_lens_sidecar

    rows = {"en/web": [_PROMPTS[0], _PROMPTS[2]], "zh/web": [_PROMPTS[1], _PROMPTS[3]]}
    monkeypatch.setattr(
        datasets, "load_dataset",
        lambda repo, **_k: iter({"text": text} for text in rows[repo]),
    )
    mix = LensCorpusMix((
        HfLensCorpus("en/web", "default", "a" * 40),
        HfLensCorpus("zh/web", "default", "b" * 40),
    ), (1, 1))
    first_docs, spec = _collect_lens_corpus(2, mix)
    LensHarness().lens.fit(first_docs, corpus_spec=spec, force=True)

    real_fit = jlens_module.fit_jacobian_lens
    fitted_widths: list[int] = []

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        fitted_widths.append(len(args[2]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "fit_jacobian_lens", counted_fit)
    docs, grown_spec = _collect_lens_corpus(4, mix)
    assert grown_spec == spec and docs == _PROMPTS
    messages: list[str] = []
    LensHarness().lens.fit(docs, corpus_spec=spec, on_progress=messages.append)
    assert any("resuming from 2 prompts" in m for m in messages)
    assert fitted_widths == [2]
    sidecar = load_lens_sidecar(_MODEL_ID)
    assert sidecar is not None and sidecar["corpus_spec"] == spec
    assert sidecar["corpus_spec"].startswith("mix:1*hf:en/web/default@")


def test_fit_changed_prefix_restarts_from_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.fit as jlens_module

    LensHarness().lens.fit(_PROMPTS[:2], force=True)
    changed = ["a changed first prompt that is long enough", *_PROMPTS[1:]]
    real_fit = jlens_module.fit_jacobian_lens
    fitted_widths: list[int] = []

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        fitted_widths.append(len(args[2]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "fit_jacobian_lens", counted_fit)
    result = LensHarness().lens.fit(changed)
    assert result.n_prompts == len(changed)
    assert fitted_widths == [len(changed)]


def test_fit_loaded_weight_change_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.fit as jlens_module

    first = LensHarness()
    first.lens.fit(_PROMPTS)
    changed = LensHarness()
    changed_model: Any = changed.model
    with torch.no_grad():
        changed_model.model.layers[0].w1.data.reshape(-1)[1] += 0.125
    real_fit = jlens_module.fit_jacobian_lens
    fitted_widths: list[int] = []

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        fitted_widths.append(len(args[2]))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_module, "fit_jacobian_lens", counted_fit)
    changed.lens.fit(_PROMPTS)
    assert fitted_widths == [len(_PROMPTS)]


def test_fit_extends_real_prefix_checkpoint_without_full_artifact() -> None:
    full = LensHarness().lens.fit(_PROMPTS, force=True)
    head_session = LensHarness()
    head = head_session.lens.fit(_PROMPTS[:2], force=True)
    lens_manifest_path(_MODEL_ID, "j").unlink()
    consumed_prefix = [
        [int(tok) for tok in head_session.tokenizer(
            p, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for p in _PROMPTS[:2]
    ]
    prefix_sha = hashlib.sha256(
        repr(consumed_prefix).encode("utf-8")
    ).hexdigest()
    _save_checkpoint(
        head, _MODEL_ID,
        corpus_spec="test",
        # This is the real identity an interrupted two-prompt request owns,
        # not the future four-prompt corpus hash.
        corpus_sha256=prefix_sha,
        corpus_hash_kind="token_ids_v1",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            head_session.model, _MODEL_ID,
        ),
        consumed_prefix_sha256=prefix_sha,
    )

    messages: list[str] = []
    resumed = head_session.lens.fit(_PROMPTS, on_progress=messages.append)

    assert any("resuming from checkpoint at 2 prompts" in m for m in messages)
    assert resumed.n_prompts == len(_PROMPTS)
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )
    assert load_lens(_MODEL_ID) is not None
    assert not lens_manifest_path(_MODEL_ID, "j", checkpoint=True).exists()


def test_fit_checkpoint_survives_two_interruptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.fit as jlens_mod
    import drowse.lens.store as lens_io

    full = LensHarness().lens.fit(_PROMPTS, force=True)
    head_session = LensHarness()
    head = head_session.lens.fit(_PROMPTS[:2], force=True)
    consumed = [
        [int(tok) for tok in head_session.tokenizer(
            prompt, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for prompt in _PROMPTS
    ]
    corpus_sha = hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest()
    _save_checkpoint(
        head, _MODEL_ID,
        corpus_spec="test",
        corpus_sha256=corpus_sha,
        corpus_hash_kind="token_ids_v1",
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            head_session.model, _MODEL_ID,
        ),
    )

    real_fit = jlens_mod.fit_jacobian_lens

    def _interrupt_after_one(*args: Any, **kwargs: Any) -> Any:
        prompts = list(args[2])
        args = (*args[:2], prompts[:1], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:1]
        kwargs["checkpoint_every"] = 1
        real_fit(*args, **kwargs)
        raise RuntimeError("simulated second interruption")

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", _interrupt_after_one)
    with pytest.raises(RuntimeError, match="second interruption"):
        LensHarness().lens.fit(_PROMPTS)

    checkpoint = load_lens_checkpoint(_MODEL_ID, "j")
    assert checkpoint is not None
    assert checkpoint[0].n_prompts == 3

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", real_fit)
    monkeypatch.setattr(
        lens_io, "load_lens",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "farther self-contained checkpoint should win before "
                "durable payload materialization"
            )
        ),
    )
    messages: list[str] = []
    resumed = LensHarness().lens.fit(_PROMPTS, on_progress=messages.append)
    assert any("checkpoint at 3 prompts" in message for message in messages)
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )


def test_corrupt_farther_checkpoint_falls_back_to_durable_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.fit as jlens_mod

    session = LensHarness()
    durable = session.lens.fit(_PROMPTS[:2], force=True)
    consumed = [
        [int(tok) for tok in session.tokenizer(
            prompt, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for prompt in _PROMPTS
    ]
    corpus_sha = hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest()
    farther = JacobianLens(
        {layer: tensor.clone() for layer, tensor in durable.jacobians.items()},
        n_prompts=3, d_model=durable.d_model,
    )
    _save_checkpoint(
        farther, _MODEL_ID,
        corpus_spec="test", corpus_sha256=corpus_sha,
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            session.model, _MODEL_ID,
        ),
    )
    checkpoint_tensor = _first_shard(_MODEL_ID, checkpoint=True)
    payload = bytearray(checkpoint_tensor.read_bytes())
    payload[-1] ^= 1
    checkpoint_tensor.write_bytes(payload)

    real_fit = jlens_mod.fit_jacobian_lens
    initial_counts: list[int | None] = []

    def capture_initial(*args: Any, **kwargs: Any) -> Any:
        initial = kwargs.get("initial_lens")
        initial_counts.append(initial.n_prompts if initial is not None else None)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", capture_initial)
    result = session.lens.fit(_PROMPTS)

    assert initial_counts == [2]
    assert result.n_prompts == len(_PROMPTS)


def test_matching_checkpoint_evicts_incompatible_resident_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    import drowse.lens.store as lens_io

    session = LensHarness()
    session.lens.fit(_PROMPTS[:2], force=True)
    resident = session.lens.resident_lens
    assert resident is not None
    resident_ref = weakref.ref(resident)
    checkpoint = JacobianLens(
        {layer: tensor.clone() for layer, tensor in resident.jacobians.items()},
        n_prompts=2, d_model=resident.d_model,
    )
    del resident
    changed = ["a changed first prompt that is long enough", *_PROMPTS[1:]]
    consumed = [
        [int(tok) for tok in session.tokenizer(
            prompt, return_tensors="pt",
        )["input_ids"][0].tolist()]
        for prompt in changed
    ]
    _save_checkpoint(
        checkpoint, _MODEL_ID,
        corpus_spec="test",
        corpus_sha256=hashlib.sha256(repr(consumed).encode("utf-8")).hexdigest(),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16,
        model_fingerprint=loaded_model_fingerprint(
            session.model, _MODEL_ID,
        ),
    )
    real_load = lens_io.load_lens_checkpoint
    resident_gone_at_load: list[bool] = []

    def observe_load(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        resident_gone_at_load.append(resident_ref() is None)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(lens_io, "load_lens_checkpoint", observe_load)
    result = session.lens.fit(changed)

    assert resident_gone_at_load == [True]
    assert result.n_prompts == len(changed)


def test_resident_prefix_is_reloaded_after_resume_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.fit as jlens_mod

    session = LensHarness()
    session.lens.fit(_PROMPTS[:2], force=True)
    durable_pair = load_lens(_MODEL_ID)
    assert durable_pair is not None
    durable = durable_pair[0]
    expected = {
        layer: tensor.clone() for layer, tensor in durable.jacobians.items()
    }
    real_fit = jlens_mod.fit_jacobian_lens

    def fail_after_one(*args: Any, **kwargs: Any) -> Any:
        prompts = list(args[2])
        args = (*args[:2], prompts[:1], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:1]
        real_fit(*args, **kwargs)
        raise RuntimeError("injected resident resume failure")

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", fail_after_one)
    with pytest.raises(RuntimeError, match="resident resume"):
        session.lens.fit(_PROMPTS)

    assert session.lens.resident_lens is not None
    assert session.lens.resident_lens.n_prompts == 2
    for layer, tensor in expected.items():
        assert torch.allclose(session.lens.resident_lens.jacobians[layer], tensor)


def test_subset_resume_releases_unrequested_resident_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref

    import drowse.lens.fit as jlens_mod

    session = LensHarness()
    session.lens.fit(_PROMPTS[:2], force=True, source_layers=[0, 1])
    assert session.lens.resident_lens is not None
    session.lens.runtime.transport_stack(session.lens.resident_lens, [0, 1], torch.device("cpu"))
    assert session.lens.runtime._stacks
    old_live_stack = torch.zeros(64)
    live_stack_ref = weakref.ref(old_live_stack)
    session.lens.live = {
        "layers": [0], "top_k": 3, "J_stack": old_live_stack,
    }
    del old_live_stack
    resident_ref = weakref.ref(session.lens.resident_lens)
    real_fit = jlens_mod.fit_jacobian_lens
    released: list[bool] = []

    def observe_release(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        released.append(
            resident_ref() is None
            and live_stack_ref() is None
            and not session.lens.runtime._stacks
        )
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", observe_release)
    resumed = session.lens.fit(_PROMPTS, source_layers=[0], force=True)

    assert released == [True]
    assert resumed.source_layers == [0]


def test_finalization_failure_rebuilds_evicted_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    session = LensHarness()
    session.lens.fit(_PROMPTS[:2], force=True)
    durable_pair = load_lens(_MODEL_ID)
    assert durable_pair is not None
    expected = durable_pair[0]
    assert session.lens.resident_lens is not None
    session.lens.runtime.transport_stack(
        session.lens.resident_lens, session.lens.resident_lens.source_layers, torch.device("cpu"),
    )

    def fail_promotion(*_args: Any, **_kwargs: Any) -> bool:
        raise OSError("injected promotion failure")

    monkeypatch.setattr(lens_io, "promote_lens_checkpoint", fail_promotion)
    with pytest.raises(OSError, match="promotion failure"):
        session.lens.fit(_PROMPTS, checkpoint_every=1)

    assert session.lens.resident_lens is not None
    assert session.lens.resident_lens.n_prompts == expected.n_prompts
    assert len(session.lens.runtime._stacks) == 0
    for layer in expected.source_layers:
        assert torch.allclose(
            session.lens.resident_lens.jacobians[layer], expected.jacobians[layer],
        )


def test_fit_missing_layer_topup_resumes_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.fit as jlens_mod
    import drowse.lens.store as lens_io
    from drowse.base import integrity
    full = LensHarness().lens.fit(
        _PROMPTS, force=True, source_layers=[0, 1],
    )
    LensHarness().lens.fit(_PROMPTS, force=True, source_layers=[0])
    real_fit = jlens_mod.fit_jacobian_lens

    def _interrupt_topup(*args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("source_layers") == [1]
        prompts = list(args[2])
        args = (*args[:2], prompts[:2], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:2]
        kwargs["checkpoint_every"] = 1
        real_fit(*args, **kwargs)
        raise RuntimeError("simulated topup interruption")

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", _interrupt_topup)
    with pytest.raises(RuntimeError, match="topup interruption"):
        LensHarness().lens.fit(_PROMPTS, source_layers=[0, 1])

    checkpoint = load_lens_checkpoint(_MODEL_ID, "j")
    assert checkpoint is not None
    assert checkpoint[0].source_layers == [1]
    assert checkpoint[0].n_prompts == 2

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", real_fit)
    real_save = lens_io.save_lens
    reused: list[set[int]] = []
    hashes = 0
    real_hash = integrity.hash_file

    def _count_hash(path: Path) -> str:
        nonlocal hashes
        hashes += 1
        return real_hash(path)

    monkeypatch.setattr(integrity, "hash_file", _count_hash)

    def _capture_reuse(*args: Any, **kwargs: Any) -> Any:
        reused.append(set(kwargs.get("reuse_layers") or ()))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(lens_io, "save_lens", _capture_reuse)
    messages: list[str] = []
    resumed = LensHarness().lens.fit(
        _PROMPTS, source_layers=[0, 1], on_progress=messages.append,
    )
    assert any("missing-layer checkpoint at 2" in message for message in messages)
    assert reused == [{0}]
    assert hashes == 0
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )


def test_fit_drops_short_prompts() -> None:
    s = LensHarness()
    messages: list[str] = []
    fitted = s.lens.fit(["tiny", *_PROMPTS], on_progress=messages.append)
    assert fitted.n_prompts == len(_PROMPTS)
    assert any("dropped 1 too-short prompts" in m for m in messages)


