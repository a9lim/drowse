"""CPU round-trip tests for the per-model Jacobian-lens artifact."""

from __future__ import annotations

import hashlib

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LensKey
from drowse.lens.store import (
    LENS_FORMAT_VERSION,
    LensProvenance,
    lens_artifact_size,
    lens_fit_lock,
    lens_manifest_path,
    load_lens,
    load_lens_checkpoint,
    load_lens_sidecar,
    remove_lens,
    remove_lens_checkpoint,
    remove_subsumed_lens_checkpoint,
    promote_lens_checkpoint,
    save_lens,
    save_lens_checkpoint,
)
from drowse.base.paths import safe_model_id

_MODEL = "test-org/tiny-model"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


_D = 8


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))


def _lens(n_layers: int = 3, n_prompts: int = 7) -> JacobianLens:
    gen = torch.Generator().manual_seed(0)
    return JacobianLens(
        {l: torch.randn(_D, _D, generator=gen) for l in range(n_layers)},
        n_prompts=n_prompts,
        d_model=_D,
    )


def _save_lens(
    lens: JacobianLens, model_id: str, *,
    reuse_layers: Any = None, **provenance: Any,
) -> Path:
    """``save_lens`` with the provenance spelled as keywords."""
    return save_lens(
        lens, model_id, LensProvenance(**provenance), reuse_layers=reuse_layers,
    )


def _save(lens: JacobianLens) -> None:
    _save_lens(
        lens, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
    )


def _save_checkpoint(
    partial: JacobianLens, model_id: str, **kwargs: Any,
) -> Path:
    """Publish a self-contained current checkpoint from estimator sums."""
    sums = {
        layer: matrix * partial.n_prompts
        for layer, matrix in partial.jacobians.items()
    }
    consumed = kwargs.pop("consumed_prefix_sha256", _digest("consumed-prefix"))
    return save_lens_checkpoint(
        sums, partial.n_prompts, partial.d_model, model_id,
        LensProvenance(**kwargs), kind=partial.key.kind,
        consumed_prefix_sha256=consumed,
    )


def _paths(*, checkpoint: bool = False) -> tuple[Path, Path]:
    """The J-lens pointer's lowest-layer ``out`` shard, and the pointer."""
    pointer = lens_manifest_path(_MODEL, "j", checkpoint=checkpoint)
    if not pointer.exists():
        return pointer.parent / "out.safetensors", pointer
    site = json.loads(pointer.read_text())["sites"]["out"]
    first = min(site["layers"])
    return pointer.parent / site["tensor_files"][str(first)], pointer


def test_paths_layout() -> None:
    _save(_lens())
    ts, sc = _paths()
    assert sc.parts[-4:] == ("lens", "local", "j", "manifest.json")
    assert ts.name.startswith("out.layer-0.gen-") and ts.suffix == ".safetensors"
    assert safe_model_id(_MODEL) in str(ts)


def test_save_load_round_trip() -> None:
    lens = _lens()
    _save(lens)

    loaded = load_lens(_MODEL)
    assert loaded is not None
    got, sidecar = loaded
    assert got.n_prompts == 7
    assert got.d_model == _D
    assert got.source_layers == [0, 1, 2]
    # fp32 storage is lossless across the artifact boundary.
    for layer in got.source_layers:
        assert got.jacobians[layer].dtype == torch.float32
        assert torch.equal(got.jacobians[layer], lens.jacobians[layer])
    assert sidecar["format_version"] == LENS_FORMAT_VERSION
    assert sidecar["dtype"] == "float32"
    assert sidecar["corpus_spec"] == "test-corpus"
    assert sidecar["corpus_sha256"] == _digest("abc123")
    assert sidecar["kind"] == "J" and got.key == LensKey("J", "out")
    assert sidecar["corpus_hash_kind"] == "token_ids_v1"
    assert sidecar["estimator_policy"]["skip_first_positions"] == 16
    assert sidecar["estimator_policy"]["direction"] == "output"
    site = sidecar["sites"]["out"]
    assert site["layers"] == [0, 1, 2]
    assert set(site["tensor_files"]) == {"0", "1", "2"}
    assert set(site["tensor_sha256"]) == {"0", "1", "2"}
    assert all(
        filename.startswith(f"out.layer-{layer}.gen-")
        for layer, filename in site["tensor_files"].items()
    )
    for layer, filename in site["tensor_files"].items():
        shard = load_file(str(_paths()[1].parent / filename))
        assert shard[f"layer_{layer}"].dtype == torch.float32


def test_missing_layer_topup_reuses_immutable_existing_shards() -> None:
    initial = _lens(n_layers=2)
    _save(initial)
    _, sc_path = _paths()
    before = json.loads(sc_path.read_text())
    before_stats = {
        layer: (sc_path.parent / filename).stat()
        for layer, filename in before["sites"]["out"]["tensor_files"].items()
    }
    extra = _lens(n_layers=3).jacobians[2]
    merged = JacobianLens(
        {**initial.jacobians, 2: extra},
        n_prompts=initial.n_prompts, d_model=_D,
    )

    _save_lens(
        merged, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())
    assert after["sites"]["out"]["tensor_files"]["0"] == before["sites"]["out"]["tensor_files"]["0"]
    assert after["sites"]["out"]["tensor_files"]["1"] == before["sites"]["out"]["tensor_files"]["1"]
    assert after["sites"]["out"]["tensor_files"]["2"] not in set(before["sites"]["out"]["tensor_files"].values())
    for layer in ("0", "1"):
        stat = (sc_path.parent / after["sites"]["out"]["tensor_files"][layer]).stat()
        old = before_stats[layer]
        assert (stat.st_ino, stat.st_mtime_ns, stat.st_size) == (
            old.st_ino, old.st_mtime_ns, old.st_size,
        )
    loaded = load_lens(_MODEL)
    assert loaded is not None and loaded[0].source_layers == [0, 1, 2]


def test_unverified_shard_reuse_falls_back_to_payload_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.base.integrity as integrity

    initial = _lens(n_layers=2)
    _save(initial)
    merged = JacobianLens(
        {**initial.jacobians, 2: torch.eye(_D)},
        n_prompts=initial.n_prompts,
        d_model=_D,
    )
    real_hash = integrity.hash_file
    hashed: list[Path] = []

    def _count_hash(path: Path) -> str:
        hashed.append(path)
        return real_hash(path)

    monkeypatch.setattr(integrity, "hash_file", _count_hash)
    _save_lens(
        merged, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    assert len(hashed) == 2
    assert {path.name for path in hashed} == {
        json.loads(_paths()[1].read_text())["sites"]["out"]["tensor_files"][layer]
        for layer in ("0", "1")
    }


def test_shard_reuse_is_refused_when_fit_identity_changes() -> None:
    lens = _lens(n_layers=2)
    _save(lens)
    _, sc_path = _paths()
    before = json.loads(sc_path.read_text())["sites"]["out"]["tensor_files"]

    _save_lens(
        lens, _MODEL,
        corpus_spec="other-corpus", corpus_sha256=_digest("different"),
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())["sites"]["out"]["tensor_files"]
    assert set(after.values()).isdisjoint(before.values())


def test_artifact_size_refreshes_a_stale_sidecar_snapshot() -> None:
    _save(_lens(n_layers=2))
    stale = load_lens_sidecar(_MODEL)
    assert stale is not None
    _save(_lens(n_layers=3, n_prompts=9))
    current = load_lens_sidecar(_MODEL)
    assert current is not None
    expected = sum(
        (_paths()[1].parent / filename).stat().st_size
        for filename in current["sites"]["out"]["tensor_files"].values()
    )

    assert lens_artifact_size(_MODEL, name="j") == expected


def test_corrupt_reusable_shard_is_rewritten_while_valid_shard_is_kept() -> None:
    initial = _lens(n_layers=2)
    _save(initial)
    _, sc_path = _paths()
    before = json.loads(sc_path.read_text())["sites"]["out"]["tensor_files"]
    corrupt = sc_path.parent / before["0"]
    payload = bytearray(corrupt.read_bytes())
    payload[-1] ^= 1
    corrupt.write_bytes(payload)
    merged = JacobianLens(
        {**initial.jacobians, 2: torch.eye(_D)},
        n_prompts=initial.n_prompts, d_model=_D,
    )

    _save_lens(
        merged, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())["sites"]["out"]["tensor_files"]
    assert after["0"] != before["0"]
    assert after["1"] == before["1"]
    loaded = load_lens(_MODEL)
    assert loaded is not None and loaded[0].source_layers == [0, 1, 2]


def test_save_load_checkpoint_round_trip() -> None:
    partial = _lens(n_layers=2, n_prompts=12)
    _save_checkpoint(
        partial, _MODEL,
        corpus_spec="test-corpus",
        corpus_sha256=_digest("abc123"),
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        consumed_prefix_sha256=_digest("consumed-prefix"),
        raw_corpus_sha256=_digest("raw456"),
        raw_prompt_count=13,
        usable_prompt_count=12,
    )

    loaded = load_lens_checkpoint(_MODEL, "j")
    assert loaded is not None
    got, sidecar = loaded
    assert got.n_prompts == 12
    assert got.source_layers == [0, 1]
    assert sidecar["checkpoint"] is True
    assert sidecar["consumed_prefix_sha256"] == _digest("consumed-prefix")
    for layer in got.source_layers:
        assert torch.allclose(got.jacobians[layer], partial.jacobians[layer])
    assert sidecar["raw_corpus_sha256"] == _digest("raw456")
    assert sidecar["raw_prompt_count"] == 13
    assert sidecar["usable_prompt_count"] == 12


def test_load_lens_sidecar_validates_tensor_header_without_materializing() -> None:
    _save(_lens())
    sidecar = load_lens_sidecar(_MODEL)
    assert sidecar is not None
    assert sidecar["sites"]["out"]["layers"] == [0, 1, 2]
    assert sidecar["d_model"] == _D

    ts_path, _ = _paths()
    ts_path.write_bytes(ts_path.read_bytes()[:32])
    assert load_lens_sidecar(_MODEL) is None


def test_save_lens_preserves_existing_tensor_on_failed_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _lens(n_prompts=3)
    _save(first)
    ts_path, _ = _paths()
    before = ts_path.read_bytes()

    def _fail_save(_fd: int) -> None:
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr("drowse.lens.store.os.fsync", _fail_save)
    with pytest.raises(RuntimeError, match="simulated"):
        _save(_lens(n_prompts=9))
    assert ts_path.read_bytes() == before
    assert not ts_path.with_suffix(ts_path.suffix + ".tmp").exists()

    loaded = load_lens(_MODEL)
    assert loaded is not None
    assert loaded[0].n_prompts == 3


def test_checkpoint_promotion_sidecar_failure_preserves_both_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    final = _lens(n_layers=2, n_prompts=3)
    _save_lens(
        final, _MODEL, corpus_spec="old", corpus_sha256=_digest("old-sha"),
        seq_len=128, dim_batch=8, skip_first=16,
        model_fingerprint="weights",
    )
    checkpoint = _lens(n_layers=2, n_prompts=5)
    _save_checkpoint(
        checkpoint, _MODEL,
        corpus_spec="new", corpus_sha256=_digest("new-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _final_tensor, final_sc = _paths()
    real_write = lens_io.write_json_atomic

    def _fail_final_pointer(path: Path, payload: object) -> None:
        if path == final_sc:
            raise OSError("simulated final pointer failure")
        real_write(path, payload)

    monkeypatch.setattr(lens_io, "write_json_atomic", _fail_final_pointer)
    with pytest.raises(OSError, match="pointer failure"):
        promote_lens_checkpoint(
            _MODEL, name="j", n_prompts=5, source_layers=[0, 1],
            corpus_sha256=_digest("new-sha"), corpus_hash_kind="token_ids_v1",
            seq_len=128, d_model=_D, model_fingerprint="weights",
        )

    still_final = load_lens(_MODEL)
    still_checkpoint = load_lens_checkpoint(_MODEL, "j")
    assert still_final is not None and still_final[0].n_prompts == 3
    assert still_checkpoint is not None and still_checkpoint[0].n_prompts == 5


def test_checkpoint_promotion_rejects_corrupt_shard_and_keeps_pointers() -> None:
    _save_lens(
        _lens(n_layers=2, n_prompts=3), _MODEL,
        corpus_spec="old", corpus_sha256=_digest("old-sha"),
        seq_len=128, dim_batch=8, skip_first=16,
        model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=2, n_prompts=5), _MODEL,
        corpus_spec="new", corpus_sha256=_digest("new-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    checkpoint_ts, checkpoint_sc = _paths(checkpoint=True)
    payload = bytearray(checkpoint_ts.read_bytes())
    payload[-1] ^= 1
    checkpoint_ts.write_bytes(payload)

    assert not promote_lens_checkpoint(
        _MODEL, name="j", n_prompts=5, source_layers=[0, 1],
        corpus_sha256=_digest("new-sha"), corpus_hash_kind="token_ids_v1",
        seq_len=128, d_model=_D, model_fingerprint="weights",
    )
    final = load_lens(_MODEL)
    assert final is not None and final[0].n_prompts == 3
    assert checkpoint_sc.exists() and checkpoint_ts.exists()


def test_subsumed_checkpoint_recovery_handles_subset_layers() -> None:
    final = _lens(n_layers=3, n_prompts=7)
    _save_lens(
        final, _MODEL, corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    # A checkpoint is self-contained: its progress is its own n_prompts.
    checkpoint = JacobianLens(
        {1: torch.eye(_D)}, n_prompts=5, d_model=_D,
    )
    _save_checkpoint(
        checkpoint, _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert remove_subsumed_lens_checkpoint(_MODEL, name="j")
    assert load_lens_checkpoint(_MODEL, "j") is None
    loaded = load_lens(_MODEL)
    assert loaded is not None and loaded[0].source_layers == [0, 1, 2]


def test_subsumed_checkpoint_recovery_preserves_farther_progress() -> None:
    _save_lens(
        _lens(n_layers=2, n_prompts=3), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=2, n_prompts=5), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert not remove_subsumed_lens_checkpoint(_MODEL, name="j")
    assert load_lens_checkpoint(_MODEL, "j") is not None


def test_subsumed_checkpoint_recovery_preserves_other_corpus() -> None:
    _save_lens(
        _lens(n_layers=2, n_prompts=7), _MODEL,
        corpus_spec="final", corpus_sha256=_digest("final-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=1, n_prompts=2), _MODEL,
        corpus_spec="checkpoint", corpus_sha256=_digest("other-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert not remove_subsumed_lens_checkpoint(_MODEL, name="j")
    assert load_lens_checkpoint(_MODEL, "j") is not None


def test_subsumed_checkpoint_recovery_preserves_recovery_point_if_final_corrupt() -> None:
    _save_lens(
        _lens(n_layers=2, n_prompts=7), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=1, n_prompts=2), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    final_tensor, _ = _paths()
    payload = bytearray(final_tensor.read_bytes())
    payload[-1] ^= 1
    final_tensor.write_bytes(payload)

    assert not remove_subsumed_lens_checkpoint(_MODEL, name="j")
    assert load_lens_checkpoint(_MODEL, "j") is not None


def test_remove_waits_for_fit_transaction() -> None:
    _save(_lens())
    started = threading.Event()
    finished = threading.Event()

    def _remove() -> None:
        started.set()
        remove_lens(_MODEL, "j")
        finished.set()

    with lens_fit_lock(_MODEL, "j"):
        worker = threading.Thread(target=_remove)
        worker.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05)
        assert load_lens(_MODEL) is not None
    worker.join(timeout=1.0)
    assert finished.is_set()
    assert load_lens(_MODEL) is None


def test_load_missing_returns_none() -> None:
    assert load_lens(_MODEL) is None
    assert load_lens_sidecar(_MODEL) is None


def test_load_previous_format_version_returns_none() -> None:
    _save(_lens())
    _, sc_path = _paths()
    sidecar = json.loads(sc_path.read_text())
    sidecar["format_version"] = LENS_FORMAT_VERSION - 1
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens(_MODEL) is None


def test_load_changed_estimator_policy_returns_none() -> None:
    _save(_lens())
    _, sc_path = _paths()
    sidecar = json.loads(sc_path.read_text())
    sidecar["estimator_policy"]["doc_chars"] += 1
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens_sidecar(_MODEL) is None
    assert load_lens(_MODEL) is None


def test_lens_fitted_at_another_skip_first_loads_and_keeps_its_policy() -> None:
    """``skip_first`` is a real knob: a recorded value other than the default
    is a valid artifact (the readout never consumes it), never "no lens"."""
    lens = _lens()
    _save_lens(
        lens, _MODEL, corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=8,
    )
    loaded = load_lens(_MODEL)
    assert loaded is not None
    assert loaded[1]["estimator_policy"]["skip_first_positions"] == 8
    assert torch.equal(loaded[0].jacobians[1], lens.jacobians[1])


def test_lens_name_must_carry_the_artifact_kind() -> None:
    """``local:r`` is always an R-lens: no fit lands in another kind's slot."""
    with pytest.raises(ValueError, match="cannot save a J lens as local:r"):
        save_lens(
            _lens(), _MODEL,
            LensProvenance("c", _digest("abc123"), 128, 8), name="r",
        )
    _save(_lens())
    directory = _paths()[1].parent
    moved = directory.parent / "r"
    directory.rename(moved)
    from drowse.lens.store import load_local_lens_sidecar

    assert load_local_lens_sidecar(_MODEL, "r") is None


def test_load_non_finite_returns_none() -> None:
    lens = _lens()
    lens.jacobians[1][0, 0] = float("inf")
    _save(lens)
    assert load_lens(_MODEL) is None


def test_load_rejects_same_shape_finite_payload_corruption() -> None:
    _save(_lens())
    ts_path, _ = _paths()
    payload = bytearray(ts_path.read_bytes())
    payload[-1] ^= 1
    ts_path.write_bytes(payload)
    assert load_lens(_MODEL) is None


def test_load_wrong_tensor_shape_returns_none() -> None:
    _save(_lens())
    ts_path, _sc_path = _paths()
    save_file({"layer_0": torch.randn(_D, _D + 1)}, str(ts_path))
    assert load_lens(_MODEL) is None


def test_load_sidecar_layer_mismatch_returns_none() -> None:
    _save(_lens())
    _ts_path, sc_path = _paths()
    sidecar = json.loads(sc_path.read_text())
    sidecar["sites"]["out"]["layers"] = [0, 1, 9]
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens(_MODEL) is None


def test_load_corrupt_sidecar_returns_none() -> None:
    _save(_lens())
    _, sc_path = _paths()
    sc_path.write_text("{not json")
    assert load_lens(_MODEL) is None


def test_remove_lens() -> None:
    _save(_lens())
    _save_checkpoint(
        _lens(n_layers=1), _MODEL,
        corpus_spec="test-corpus",
        corpus_sha256=_digest("abc123"),
        seq_len=128,
        dim_batch=8,
        skip_first=16,
    )
    ckpt_ts, ckpt_sc = _paths(checkpoint=True)
    assert ckpt_ts.exists() and ckpt_sc.exists()
    assert remove_lens(_MODEL, "j") is True
    assert load_lens(_MODEL) is None
    assert load_lens_checkpoint(_MODEL, "j") is None
    assert remove_lens(_MODEL, "j") is False


def test_remove_lens_keeps_a_different_active_local_lens() -> None:
    """Removing one local lens must not unpublish another that stays on disk."""
    from drowse.lens.sources import LENS_SOURCES

    _save(_lens())
    relp = _lens()
    relp.key = LensKey("R")
    _save_lens(
        relp, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
    )
    active = LENS_SOURCES.active(_MODEL, "output")
    assert active is not None and active.name == "r"

    assert remove_lens(_MODEL, "j") is True
    active = LENS_SOURCES.active(_MODEL, "output")
    assert active is not None and active.name == "r"
    assert load_lens(_MODEL) is not None

    assert remove_lens(_MODEL, "r") is True
    assert LENS_SOURCES.active(_MODEL, "output") is None


def test_remove_lens_reaps_crash_left_streaming_temp() -> None:
    generation, _ = _paths()
    orphan = generation.parent / "out.layer-1.gen-deadbeef.safetensors.tmp"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"partial shard")

    assert remove_lens(_MODEL, "j") is True
    assert not orphan.exists()


def test_checkpoint_pointer_unlink_is_durable_before_shard_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    _save_checkpoint(
        _lens(n_layers=1), _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
    )
    events: list[str] = []
    real_unlink = Path.unlink
    real_cleanup = lens_io._cleanup_unreferenced_generations

    def record_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        events.append("pointer" if path.suffix == ".json" else "shard")
        real_unlink(path, *args, **kwargs)

    def record_cleanup(path: Path) -> None:
        events.append("gc")
        real_cleanup(path)

    monkeypatch.setattr(Path, "unlink", record_unlink)
    monkeypatch.setattr(lens_io, "fsync_directory", lambda _path: events.append("barrier"))
    monkeypatch.setattr(lens_io, "_cleanup_unreferenced_generations", record_cleanup)

    assert remove_lens_checkpoint(_MODEL, "j")
    assert events[0:3] == ["pointer", "barrier", "gc"]
    assert events.index("barrier") < events.index("shard")


def test_subsumed_checkpoint_unlink_is_durable_before_shard_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    _save_lens(
        _lens(n_layers=2, n_prompts=7), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=1, n_prompts=2), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    events: list[str] = []
    real_unlink = Path.unlink
    real_cleanup = lens_io._cleanup_unreferenced_generations

    def record_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        events.append("pointer" if path.suffix == ".json" else "shard")
        real_unlink(path, *args, **kwargs)

    def record_cleanup(path: Path) -> None:
        events.append("gc")
        real_cleanup(path)

    monkeypatch.setattr(Path, "unlink", record_unlink)
    monkeypatch.setattr(lens_io, "fsync_directory", lambda _path: events.append("barrier"))
    monkeypatch.setattr(lens_io, "_cleanup_unreferenced_generations", record_cleanup)

    assert remove_subsumed_lens_checkpoint(_MODEL, name="j")
    pointer = events.index("pointer")
    barrier = events.index("barrier", pointer + 1)
    gc_event = events.index("gc")
    assert pointer < barrier < gc_event < events.index("shard")


def test_full_lens_pointer_unlinks_are_durable_before_shard_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    _save(_lens())
    _save_checkpoint(
        _lens(n_layers=1), _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
    )
    events: list[str] = []
    real_unlink = Path.unlink

    def record_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        events.append("pointer" if path.suffix == ".json" else "shard")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", record_unlink)
    monkeypatch.setattr(lens_io, "fsync_directory", lambda _path: events.append("barrier"))

    assert remove_lens(_MODEL, "j")
    first_barrier = events.index("barrier")
    assert events[:first_barrier] == ["pointer", "pointer"]
    assert first_barrier < events.index("shard")


def test_payload_and_pointer_directory_barriers_precede_generation_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    events: list[str] = []
    real_write = lens_io.write_json_atomic
    real_cleanup = lens_io._cleanup_unreferenced_generations

    def record_write(path: Path, payload: object) -> None:
        real_write(path, payload)
        events.append("pointer")

    def record_barrier(_path: Path) -> None:
        events.append("barrier")

    def record_cleanup(path: Path) -> None:
        events.append("gc")
        real_cleanup(path)

    monkeypatch.setattr(lens_io, "write_json_atomic", record_write)
    monkeypatch.setattr(lens_io, "fsync_directory", record_barrier)
    monkeypatch.setattr(
        lens_io, "_cleanup_unreferenced_generations", record_cleanup,
    )

    _save(_lens())

    assert events[0:3] == ["barrier", "pointer", "barrier"]
    assert events.index("barrier", 2) < events.index("gc")


def test_checkpoint_payload_is_durable_before_pointer_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    events: list[str] = []
    real_payload = lens_io._save_fp32_square_safetensors_atomic
    real_pointer = lens_io.write_json_atomic

    def record_payload(*args: object, **kwargs: object) -> str:
        assert kwargs.get("durable") is True
        digest = real_payload(*args, **kwargs)  # type: ignore[arg-type]
        events.append("payload")
        return digest

    def record_pointer(path: Path, payload: object) -> None:
        events.append("pointer")
        real_pointer(path, payload)

    monkeypatch.setattr(
        lens_io, "_save_fp32_square_safetensors_atomic", record_payload,
    )
    monkeypatch.setattr(lens_io, "write_json_atomic", record_pointer)
    _save_checkpoint(
        _lens(n_layers=2, n_prompts=2), _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"), seq_len=128,
        dim_batch=8, skip_first=16,
    )
    assert events == ["payload", "payload", "pointer"]


def test_exception_after_lens_pointer_replace_preserves_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    _save(_lens(n_layers=2, n_prompts=3))
    anchor, pointer = _paths()
    prior_files = set(json.loads(pointer.read_text())["sites"]["out"]["tensor_files"].values())
    real_write = lens_io.write_json_atomic

    def write_then_fail(path: Path, payload: object) -> None:
        real_write(path, payload)
        raise OSError("after lens pointer replace")

    monkeypatch.setattr(lens_io, "write_json_atomic", write_then_fail)
    with pytest.raises(OSError, match="after lens pointer"):
        _save(_lens(n_layers=2, n_prompts=11))

    monkeypatch.setattr(lens_io, "write_json_atomic", real_write)
    loaded = load_lens(_MODEL)
    assert loaded is not None
    assert loaded[0].n_prompts == 11
    assert not any((anchor.parent / name).exists() for name in prior_files)


def test_lens_generation_gc_fails_closed_on_transient_pointer_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    import drowse.lens.store as lens_io

    _save(_lens(n_layers=2))
    anchor, pointer = _paths()
    live_files = set(json.loads(pointer.read_text())["sites"]["out"]["tensor_files"].values())
    orphan = anchor.parent / "out.layer-99.gen-orphan.safetensors"
    orphan.write_bytes(b"orphan")
    real_open = builtins.open

    def fail_pointer(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == pointer:
            raise OSError("transient pointer read")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_pointer)
    lens_io._cleanup_unreferenced_generations(anchor.parent)

    assert orphan.exists()
    assert all((anchor.parent / name).exists() for name in live_files)


def test_checkpoint_promotion_directory_barrier_precedes_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    checkpoint = _lens(n_layers=2, n_prompts=5)
    _save_checkpoint(
        checkpoint, _MODEL,
        corpus_spec="new", corpus_sha256=_digest("new-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    final_pointer = lens_manifest_path(_MODEL, "j")
    checkpoint_pointer = lens_manifest_path(_MODEL, "j", checkpoint=True)
    events: list[str] = []
    real_fsync = lens_io.os.fsync
    real_write = lens_io.write_json_atomic
    real_unlink = Path.unlink
    real_cleanup = lens_io._cleanup_unreferenced_generations

    def record_fsync(fd: int) -> None:
        events.append("file")
        real_fsync(fd)

    def record_dir(path: Path) -> None:
        events.append("dir")
        del path

    def record_pointer(path: Path, payload: object) -> None:
        if path == final_pointer:
            events.append("pointer")
        real_write(path, payload)

    def record_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == checkpoint_pointer:
            events.append("checkpoint-unlink")
        real_unlink(path, *args, **kwargs)

    def record_cleanup(path: Path) -> None:
        events.append("gc")
        real_cleanup(path)

    monkeypatch.setattr(lens_io.os, "fsync", record_fsync)
    monkeypatch.setattr(lens_io, "fsync_directory", record_dir)
    monkeypatch.setattr(lens_io, "write_json_atomic", record_pointer)
    monkeypatch.setattr(Path, "unlink", record_unlink)
    monkeypatch.setattr(
        lens_io, "_cleanup_unreferenced_generations", record_cleanup,
    )

    assert promote_lens_checkpoint(
        _MODEL, name="j", n_prompts=5, source_layers=[0, 1],
        corpus_sha256=_digest("new-sha"), corpus_hash_kind="token_ids_v1",
        seq_len=128, d_model=_D, model_fingerprint="weights",
    )
    pointer_index = events.index("pointer")
    assert "file" in events[:pointer_index]
    assert events[pointer_index - 1] == "dir"
    unlink_index = events.index("checkpoint-unlink")
    assert events[unlink_index + 1] == "dir"
    assert unlink_index < events.index("gc")


def test_remove_unpublishes_before_best_effort_tensor_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(_lens())
    real_unlink = Path.unlink

    def _fail_tensor_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".safetensors" and ".gen-" in path.name:
            raise OSError("simulated tensor GC failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _fail_tensor_unlink)

    assert remove_lens(_MODEL, "j")
    assert load_lens(_MODEL) is None
    assert load_lens_checkpoint(_MODEL, "j") is None


def test_lens_rejects_noncanonical_identity_digest() -> None:
    _save(_lens())
    _anchor, sidecar_path = _paths()
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["corpus_sha256"] = "not-a-digest"
    sidecar_path.write_text(json.dumps(sidecar))
    assert load_lens(_MODEL) is None


def test_checkpoint_progress_requires_consumed_prefix_identity() -> None:
    _save_checkpoint(
        _lens(n_layers=2, n_prompts=2), _MODEL,
        corpus_spec="test", corpus_sha256=_digest("test"),
        seq_len=8, dim_batch=2, skip_first=0,
    )
    _anchor, sidecar_path = _paths(checkpoint=True)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["consumed_prefix_sha256"] = None
    sidecar_path.write_text(json.dumps(sidecar))
    assert load_lens_checkpoint(_MODEL, "j") is None
