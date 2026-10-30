"""CPU round-trip tests for the per-model Jacobian-lens artifact."""

from __future__ import annotations

import hashlib

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from drowse.lens.jlens import JacobianLens, JacobianLensCancelled
from drowse.lens.store import (
    LENS_FORMAT_VERSION,
    lens_artifact_size,
    lens_checkpoint_paths,
    lens_fit_lock,
    lens_paths,
    load_lens,
    load_lens_checkpoint,
    load_lens_sidecar,
    remove_lens,
    remove_lens_checkpoint,
    remove_subsumed_lens_checkpoint,
    promote_lens_checkpoint,
    save_lens,
    save_lens_checkpoint_accumulator,
    stream_lens_corpus,
)
from drowse.base.paths import safe_model_id

_MODEL = "test-org/tiny-model"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_default_corpus_cancel_does_not_wait_for_blocked_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web cancel button must not depend on a blocked Hub read returning."""
    import drowse.lens.store as lens_io

    provider_entered = threading.Event()
    cancelled = threading.Event()

    class BlockedProcess:
        alive = True
        terminated = False
        joined = False

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float = 0) -> None:
            del timeout
            self.joined = True

    class EmptyReceiver:
        closed = False

        def poll(self, timeout: float) -> bool:
            provider_entered.set()
            time.sleep(timeout)
            return False

        def recv(self) -> object:
            raise AssertionError("blocked provider has no result")

        def close(self) -> None:
            self.closed = True

    process = BlockedProcess()
    receiver = EmptyReceiver()
    monkeypatch.setattr(
        lens_io,
        "_start_lens_corpus_process",
        lambda _n, _corpus: (process, receiver),
    )

    def request_cancel() -> None:
        assert provider_entered.wait(timeout=1)
        cancelled.set()

    threading.Thread(target=request_cancel, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(JacobianLensCancelled, match="streaming its corpus"):
        stream_lens_corpus(1, cancel_event=cancelled)
    assert time.monotonic() - started < 1.0
    assert process.terminated and process.joined
    assert receiver.closed


def test_default_corpus_cancel_wins_a_simultaneously_ready_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel arriving during poll must not start the estimator afterward."""
    import drowse.lens.store as lens_io

    cancelled = threading.Event()

    class FinishedProcess:
        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            raise AssertionError("completed worker must not be terminated")

        def kill(self) -> None:
            raise AssertionError("completed worker must not be killed")

        def join(self, timeout: float = 0) -> None:
            del timeout

    class ReadyReceiver:
        closed = False

        def poll(self, timeout: float) -> bool:
            del timeout
            cancelled.set()
            return True

        def recv(self) -> object:
            raise AssertionError("cancel must win over the queued result")

        def close(self) -> None:
            self.closed = True

    receiver = ReadyReceiver()
    monkeypatch.setattr(
        lens_io,
        "_start_lens_corpus_process",
        lambda _n, _corpus: (FinishedProcess(), receiver),
    )

    with pytest.raises(JacobianLensCancelled, match="streaming its corpus"):
        stream_lens_corpus(1, cancel_event=cancelled)
    assert receiver.closed


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


def _save(lens: JacobianLens) -> None:
    save_lens(
        lens, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
    )


def _save_checkpoint(
    partial: JacobianLens, model_id: str, *, base_n_prompts: int, **kwargs: Any,
) -> Path:
    """Publish a self-contained current checkpoint from estimator sums."""
    base = (
        JacobianLens(
            dict(partial.jacobians), n_prompts=base_n_prompts,
            d_model=partial.d_model,
        )
        if base_n_prompts else None
    )
    sums = {
        layer: matrix * partial.n_prompts
        for layer, matrix in partial.jacobians.items()
    }
    kwargs.setdefault("consumed_prefix_sha256", _digest("consumed-prefix"))
    return save_lens_checkpoint_accumulator(
        sums, partial.n_prompts, partial.d_model, model_id,
        base=base, **kwargs,
    )


def test_paths_layout() -> None:
    ts, sc = lens_paths(_MODEL)
    assert ts.suffix == ".safetensors" and sc.suffix == ".json"
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
    assert sidecar["corpus_hash_kind"] == "text_v1"
    assert sidecar["skip_first_positions"] == 16
    assert set(sidecar["tensor_files"]) == {"0", "1", "2"}
    assert set(sidecar["tensor_sha256"]) == {"0", "1", "2"}
    assert all(
        ".gen-" in filename and f"layer-{layer}" in filename
        for layer, filename in sidecar["tensor_files"].items()
    )
    assert lens_paths(_MODEL)[0].name == sidecar["tensor_files"]["0"]
    for layer, filename in sidecar["tensor_files"].items():
        shard = load_file(str(lens_paths(_MODEL)[1].parent / filename))
        assert shard[f"layer_{layer}"].dtype == torch.float32


def test_missing_layer_topup_reuses_immutable_existing_shards() -> None:
    initial = _lens(n_layers=2)
    _save(initial)
    _, sc_path = lens_paths(_MODEL)
    before = json.loads(sc_path.read_text())
    before_stats = {
        layer: (sc_path.parent / filename).stat()
        for layer, filename in before["tensor_files"].items()
    }
    extra = _lens(n_layers=3).jacobians[2]
    merged = JacobianLens(
        {**initial.jacobians, 2: extra},
        n_prompts=initial.n_prompts, d_model=_D,
    )

    save_lens(
        merged, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())
    assert after["tensor_files"]["0"] == before["tensor_files"]["0"]
    assert after["tensor_files"]["1"] == before["tensor_files"]["1"]
    assert after["tensor_files"]["2"] not in set(before["tensor_files"].values())
    for layer in ("0", "1"):
        stat = (sc_path.parent / after["tensor_files"][layer]).stat()
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
    save_lens(
        merged, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    assert len(hashed) == 2
    assert {path.name for path in hashed} == {
        json.loads(lens_paths(_MODEL)[1].read_text())["tensor_files"][layer]
        for layer in ("0", "1")
    }


def test_shard_reuse_is_refused_when_fit_identity_changes() -> None:
    lens = _lens(n_layers=2)
    _save(lens)
    _, sc_path = lens_paths(_MODEL)
    before = json.loads(sc_path.read_text())["tensor_files"]

    save_lens(
        lens, _MODEL,
        corpus_spec="other-corpus", corpus_sha256=_digest("different"),
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())["tensor_files"]
    assert set(after.values()).isdisjoint(before.values())


def test_artifact_size_refreshes_a_stale_sidecar_snapshot() -> None:
    _save(_lens(n_layers=2))
    stale = load_lens_sidecar(_MODEL)
    assert stale is not None
    _save(_lens(n_layers=3, n_prompts=9))
    current = load_lens_sidecar(_MODEL)
    assert current is not None
    expected = sum(
        (lens_paths(_MODEL)[1].parent / filename).stat().st_size
        for filename in current["tensor_files"].values()
    )

    assert lens_artifact_size(_MODEL, stale) == expected


def test_corrupt_reusable_shard_is_rewritten_while_valid_shard_is_kept() -> None:
    initial = _lens(n_layers=2)
    _save(initial)
    _, sc_path = lens_paths(_MODEL)
    before = json.loads(sc_path.read_text())["tensor_files"]
    corrupt = sc_path.parent / before["0"]
    payload = bytearray(corrupt.read_bytes())
    payload[-1] ^= 1
    corrupt.write_bytes(payload)
    merged = JacobianLens(
        {**initial.jacobians, 2: torch.eye(_D)},
        n_prompts=initial.n_prompts, d_model=_D,
    )

    save_lens(
        merged, _MODEL,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"),
        seq_len=128, dim_batch=8, skip_first=16,
        reuse_layers={0, 1},
    )

    after = json.loads(sc_path.read_text())["tensor_files"]
    assert after["0"] != before["0"]
    assert after["1"] == before["1"]
    loaded = load_lens(_MODEL)
    assert loaded is not None and loaded[0].source_layers == [0, 1, 2]


def test_save_load_checkpoint_round_trip() -> None:
    partial = _lens(n_layers=2, n_prompts=5)
    _save_checkpoint(
        partial, _MODEL,
        base_n_prompts=7,
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

    loaded = load_lens_checkpoint(_MODEL)
    assert loaded is not None
    got, sidecar = loaded
    assert got.n_prompts == 12
    assert got.source_layers == [0, 1]
    assert sidecar["checkpoint"] is True
    assert sidecar["base_n_prompts"] == 0
    assert sidecar["partial_n_prompts"] == 12
    assert sidecar["raw_corpus_sha256"] == _digest("raw456")
    assert sidecar["raw_prompt_count"] == 13
    assert sidecar["usable_prompt_count"] == 12


def test_checkpoint_accumulator_is_self_contained_and_merges_prefix() -> None:
    base = _lens(n_layers=2, n_prompts=3)
    tail = _lens(n_layers=2, n_prompts=2)
    sums = {layer: J * tail.n_prompts for layer, J in tail.jacobians.items()}

    save_lens_checkpoint_accumulator(
        sums, tail.n_prompts, _D, _MODEL,
        base=base,
        corpus_spec="test-corpus",
        corpus_sha256=_digest("abc123"),
        seq_len=128,
        dim_batch=8,
        skip_first=16,
        model_layer_count=3,
        consumed_prefix_sha256=_digest("consumed-prefix"),
    )

    loaded = load_lens_checkpoint(_MODEL)
    assert loaded is not None
    got, sidecar = loaded
    expected = JacobianLens.merge([base, tail])
    assert got.n_prompts == expected.n_prompts == 5
    assert sidecar["base_n_prompts"] == 0
    assert sidecar["partial_n_prompts"] == 5
    assert sidecar["model_layer_count"] == 3
    for layer in got.source_layers:
        assert torch.allclose(
            got.jacobians[layer], expected.jacobians[layer], atol=2e-3,
        )


def test_load_lens_sidecar_validates_tensor_header_without_materializing() -> None:
    _save(_lens())
    sidecar = load_lens_sidecar(_MODEL)
    assert sidecar is not None
    assert sidecar["source_layers"] == [0, 1, 2]
    assert sidecar["d_model"] == _D

    ts_path, _ = lens_paths(_MODEL)
    ts_path.write_bytes(ts_path.read_bytes()[:32])
    assert load_lens_sidecar(_MODEL) is None


def test_save_lens_preserves_existing_tensor_on_failed_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _lens(n_prompts=3)
    _save(first)
    ts_path, _ = lens_paths(_MODEL)
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
    save_lens(
        final, _MODEL, corpus_spec="old", corpus_sha256=_digest("old-sha"),
        seq_len=128, dim_batch=8, skip_first=16,
        model_fingerprint="weights",
    )
    checkpoint = _lens(n_layers=2, n_prompts=5)
    _save_checkpoint(
        checkpoint, _MODEL, base_n_prompts=0,
        corpus_spec="new", corpus_sha256=_digest("new-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _final_tensor, final_sc = lens_paths(_MODEL)
    real_write = lens_io.write_json_atomic

    def _fail_final_pointer(path: Path, payload: object) -> None:
        if path == final_sc:
            raise OSError("simulated final pointer failure")
        real_write(path, payload)

    monkeypatch.setattr(lens_io, "write_json_atomic", _fail_final_pointer)
    with pytest.raises(OSError, match="pointer failure"):
        promote_lens_checkpoint(
            _MODEL, n_prompts=5, source_layers=[0, 1],
            corpus_sha256=_digest("new-sha"), corpus_hash_kind="token_ids_v1",
            seq_len=128, d_model=_D, model_fingerprint="weights",
        )

    still_final = load_lens(_MODEL)
    still_checkpoint = load_lens_checkpoint(_MODEL)
    assert still_final is not None and still_final[0].n_prompts == 3
    assert still_checkpoint is not None and still_checkpoint[0].n_prompts == 5


def test_checkpoint_promotion_rejects_corrupt_shard_and_keeps_pointers() -> None:
    save_lens(
        _lens(n_layers=2, n_prompts=3), _MODEL,
        corpus_spec="old", corpus_sha256=_digest("old-sha"),
        seq_len=128, dim_batch=8, skip_first=16,
        model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=2, n_prompts=5), _MODEL, base_n_prompts=0,
        corpus_spec="new", corpus_sha256=_digest("new-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    checkpoint_ts, checkpoint_sc = lens_checkpoint_paths(_MODEL)
    payload = bytearray(checkpoint_ts.read_bytes())
    payload[-1] ^= 1
    checkpoint_ts.write_bytes(payload)

    assert not promote_lens_checkpoint(
        _MODEL, n_prompts=5, source_layers=[0, 1],
        corpus_sha256=_digest("new-sha"), corpus_hash_kind="token_ids_v1",
        seq_len=128, d_model=_D, model_fingerprint="weights",
    )
    final = load_lens(_MODEL)
    assert final is not None and final[0].n_prompts == 3
    assert checkpoint_sc.exists() and checkpoint_ts.exists()


def test_subsumed_checkpoint_recovery_handles_base_progress_and_subset_layers() -> None:
    final = _lens(n_layers=3, n_prompts=7)
    save_lens(
        final, _MODEL, corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    # Legacy-compatible progress semantics: 3 durable base prompts plus a
    # 2-prompt partial means this checkpoint reaches prompt 5, not prompt 2.
    checkpoint = JacobianLens(
        {1: torch.eye(_D)}, n_prompts=2, d_model=_D,
    )
    _save_checkpoint(
        checkpoint, _MODEL, base_n_prompts=3,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert remove_subsumed_lens_checkpoint(_MODEL)
    assert load_lens_checkpoint(_MODEL) is None
    loaded = load_lens(_MODEL)
    assert loaded is not None and loaded[0].source_layers == [0, 1, 2]


def test_subsumed_checkpoint_recovery_preserves_farther_progress() -> None:
    save_lens(
        _lens(n_layers=2, n_prompts=3), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=2, n_prompts=2), _MODEL, base_n_prompts=3,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert not remove_subsumed_lens_checkpoint(_MODEL)
    assert load_lens_checkpoint(_MODEL) is not None


def test_subsumed_checkpoint_recovery_preserves_other_corpus() -> None:
    save_lens(
        _lens(n_layers=2, n_prompts=7), _MODEL,
        corpus_spec="final", corpus_sha256=_digest("final-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=1, n_prompts=2), _MODEL, base_n_prompts=0,
        corpus_spec="checkpoint", corpus_sha256=_digest("other-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )

    assert not remove_subsumed_lens_checkpoint(_MODEL)
    assert load_lens_checkpoint(_MODEL) is not None


def test_subsumed_checkpoint_recovery_preserves_recovery_point_if_final_corrupt() -> None:
    save_lens(
        _lens(n_layers=2, n_prompts=7), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=1, n_prompts=2), _MODEL, base_n_prompts=0,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    final_tensor, _ = lens_paths(_MODEL)
    payload = bytearray(final_tensor.read_bytes())
    payload[-1] ^= 1
    final_tensor.write_bytes(payload)

    assert not remove_subsumed_lens_checkpoint(_MODEL)
    assert load_lens_checkpoint(_MODEL) is not None


def test_remove_waits_for_fit_transaction() -> None:
    _save(_lens())
    started = threading.Event()
    finished = threading.Event()

    def _remove() -> None:
        started.set()
        remove_lens(_MODEL)
        finished.set()

    with lens_fit_lock(_MODEL):
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
    _, sc_path = lens_paths(_MODEL)
    sidecar = json.loads(sc_path.read_text())
    sidecar["format_version"] = LENS_FORMAT_VERSION - 1
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens(_MODEL) is None


def test_load_changed_estimator_policy_returns_none() -> None:
    _save(_lens())
    _, sc_path = lens_paths(_MODEL)
    sidecar = json.loads(sc_path.read_text())
    sidecar["estimator_policy"]["skip_first_positions"] += 1
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens_sidecar(_MODEL) is None
    assert load_lens(_MODEL) is None


def test_load_non_finite_returns_none() -> None:
    lens = _lens()
    lens.jacobians[1][0, 0] = float("inf")
    _save(lens)
    assert load_lens(_MODEL) is None


def test_load_rejects_same_shape_finite_payload_corruption() -> None:
    _save(_lens())
    ts_path, _ = lens_paths(_MODEL)
    payload = bytearray(ts_path.read_bytes())
    payload[-1] ^= 1
    ts_path.write_bytes(payload)
    assert load_lens(_MODEL) is None


def test_load_wrong_tensor_shape_returns_none() -> None:
    _save(_lens())
    ts_path, _sc_path = lens_paths(_MODEL)
    save_file({"layer_0": torch.randn(_D, _D + 1)}, str(ts_path))
    assert load_lens(_MODEL) is None


def test_load_sidecar_layer_mismatch_returns_none() -> None:
    _save(_lens())
    _ts_path, sc_path = lens_paths(_MODEL)
    sidecar = json.loads(sc_path.read_text())
    sidecar["source_layers"] = [0, 1, 9]
    sc_path.write_text(json.dumps(sidecar))
    assert load_lens(_MODEL) is None


def test_load_corrupt_sidecar_returns_none() -> None:
    _save(_lens())
    _, sc_path = lens_paths(_MODEL)
    sc_path.write_text("{not json")
    assert load_lens(_MODEL) is None


def test_remove_lens() -> None:
    _save(_lens())
    _save_checkpoint(
        _lens(n_layers=1), _MODEL,
        base_n_prompts=0,
        corpus_spec="test-corpus",
        corpus_sha256=_digest("abc123"),
        seq_len=128,
        dim_batch=8,
        skip_first=16,
    )
    ckpt_ts, ckpt_sc = lens_checkpoint_paths(_MODEL)
    assert ckpt_ts.exists() and ckpt_sc.exists()
    assert remove_lens(_MODEL) is True
    assert load_lens(_MODEL) is None
    assert load_lens_checkpoint(_MODEL) is None
    assert remove_lens(_MODEL) is False


def test_remove_lens_reaps_crash_left_streaming_temp() -> None:
    generation, _ = lens_paths(_MODEL)
    orphan = generation.parent / "jlens.layer-1.gen-deadbeef.safetensors.tmp"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"partial shard")

    assert remove_lens(_MODEL) is True
    assert not orphan.exists()


def test_checkpoint_pointer_unlink_is_durable_before_shard_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    _save_checkpoint(
        _lens(n_layers=1), _MODEL, base_n_prompts=0,
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

    assert remove_lens_checkpoint(_MODEL)
    assert events[0:3] == ["pointer", "barrier", "gc"]
    assert events.index("barrier") < events.index("shard")


def test_subsumed_checkpoint_unlink_is_durable_before_shard_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    save_lens(
        _lens(n_layers=2, n_prompts=7), _MODEL,
        corpus_spec="same", corpus_sha256=_digest("same-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _save_checkpoint(
        _lens(n_layers=1, n_prompts=2), _MODEL, base_n_prompts=0,
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

    assert remove_subsumed_lens_checkpoint(_MODEL)
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
        _lens(n_layers=1), _MODEL, base_n_prompts=0,
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

    assert remove_lens(_MODEL)
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
        _lens(n_layers=2, n_prompts=2), _MODEL, base_n_prompts=0,
        corpus_spec="test-corpus", corpus_sha256=_digest("abc123"), seq_len=128,
        dim_batch=8, skip_first=16,
    )
    assert events == ["payload", "payload", "pointer"]


def test_exception_after_lens_pointer_replace_preserves_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    _save(_lens(n_layers=2, n_prompts=3))
    anchor, pointer = lens_io._lens_anchor_paths(_MODEL)
    prior_files = set(json.loads(pointer.read_text())["tensor_files"].values())
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
    anchor, pointer = lens_io._lens_anchor_paths(_MODEL)
    live_files = set(json.loads(pointer.read_text())["tensor_files"].values())
    orphan = anchor.parent / f"{anchor.stem}.layer-99.gen-orphan.safetensors"
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
        checkpoint, _MODEL, base_n_prompts=0,
        corpus_spec="new", corpus_sha256=_digest("new-sha"),
        corpus_hash_kind="token_ids_v1", seq_len=128, dim_batch=8,
        skip_first=16, model_fingerprint="weights",
    )
    _final_anchor, final_pointer = lens_io._lens_anchor_paths(_MODEL)
    _checkpoint_anchor, checkpoint_pointer = lens_io._checkpoint_anchor_paths(
        _MODEL,
    )
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
        _MODEL, n_prompts=5, source_layers=[0, 1],
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

    assert remove_lens(_MODEL)
    assert load_lens(_MODEL) is None
    assert load_lens_checkpoint(_MODEL) is None


def test_lens_rejects_noncanonical_identity_digest() -> None:
    _save(_lens())
    _anchor, sidecar_path = lens_paths(_MODEL)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["corpus_sha256"] = "not-a-digest"
    sidecar_path.write_text(json.dumps(sidecar))
    assert load_lens(_MODEL) is None


def test_checkpoint_progress_requires_consumed_prefix_identity() -> None:
    _save_checkpoint(
        _lens(n_layers=2, n_prompts=2), _MODEL, base_n_prompts=0,
        corpus_spec="test", corpus_sha256=_digest("test"),
        seq_len=8, dim_batch=2, skip_first=0,
    )
    _anchor, sidecar_path = lens_checkpoint_paths(_MODEL)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["consumed_prefix_sha256"] = None
    sidecar_path.write_text(json.dumps(sidecar))
    assert load_lens_checkpoint(_MODEL) is None


# ------------------------------------------------- Hugging Face lens corpora --

_SHA = "0123456789abcdef0123456789abcdef01234567"


def _source(raw: str) -> Any:
    from drowse.lens.store import HfLensCorpus, parse_lens_corpus

    corpus = parse_lens_corpus([raw])
    assert isinstance(corpus, HfLensCorpus)
    return corpus


def test_hf_corpus_spec_records_repo_config_commit_and_preprocessing() -> None:
    from drowse.lens.store import (
        DEFAULT_LENS_CORPUS,
        LENS_CORPUS_PREPROCESS_VERSION,
        LENS_DOC_CHARS,
        HfLensCorpus,
    )

    tail = f";preprocess={LENS_CORPUS_PREPROCESS_VERSION};doc_chars={LENS_DOC_CHARS}"
    corpus = _source(f"HuggingFaceFW/dclm_100BT-shuffled:default@{_SHA}")
    assert corpus.pinned
    assert corpus.spec == f"hf:HuggingFaceFW/dclm_100BT-shuffled/default@{_SHA}{tail}"
    # The default corpus keeps the exact spec its existing fits recorded.
    assert DEFAULT_LENS_CORPUS == _source("HuggingFaceFW/fineweb-edu:sample-10BT")
    assert HfLensCorpus("HuggingFaceFW/fineweb-edu", "sample-10BT", _SHA).spec == (
        f"hf:HuggingFaceFW/fineweb-edu/sample-10BT@{_SHA}{tail}"
    )
    # A non-default column and a spread read are part of the corpus identity.
    assert _source(f"o/r:c@{_SHA},text_field=content").spec == (
        f"hf:o/r/c@{_SHA}{tail};text_field=content"
    )
    assert _source(f"o/r:c@{_SHA},files=64,seed=3,text_field=body").spec == (
        f"hf:o/r/c@{_SHA}{tail};text_field=body;files=64;seed=3"
    )
    assert _source(f"o/r:c@{_SHA},files=64").spec.endswith(";files=64;seed=0")


def test_hf_corpus_parse_round_trips_and_rejects_malformed_specs() -> None:
    from drowse.lens.store import DEFAULT_LENS_CORPUS, HfLensCorpus, parse_lens_corpus

    assert parse_lens_corpus(None) is DEFAULT_LENS_CORPUS
    assert parse_lens_corpus([]) is DEFAULT_LENS_CORPUS
    assert _source("org/data") == HfLensCorpus("org/data")
    assert _source("org/data@v1.0") == HfLensCorpus("org/data", None, "v1.0")
    # Branch names may contain slashes (e.g. the Hub's parquet conversion).
    parquet = _source("org/data:cfg@refs/convert/parquet")
    assert parquet == HfLensCorpus("org/data", "cfg", "refs/convert/parquet")
    assert _source("org/data,files=8,seed=2,text_field=body") == HfLensCorpus(
        "org/data", text_field="body", files=8, seed=2,
    )
    for raw in (
        "org/data", "org/data:cfg", f"org/data:cfg@{_SHA}", "org/data@main",
        "org/data:cfg@main,text_field=body,files=8,seed=2",
    ):
        assert str(_source(raw)) == raw
    for unpinned in ("org/data", "org/data:cfg", f"org/data@{_SHA}", "org/data:cfg@main"):
        corpus = _source(unpinned)
        assert not corpus.pinned
        with pytest.raises(ValueError, match="not pinned"):
            _ = corpus.spec
    for bad in ("", "  ", ":cfg", "org/data:", "org/data@", "org/data@main:cfg", "org data"):
        with pytest.raises(ValueError, match="REPO"):
            parse_lens_corpus([bad])
    for bad, needle in (
        ("org/data,foo=1", "bad corpus option"),
        ("org/data,files=1,files=2", "bad corpus option"),
        ("org/data,files=", "bad corpus option"),
        ("org/data,text_field=", "bad corpus option"),
        ("org/data,files=x", "integers"),
        ("org/data,files=0", "at least 1"),
        ("org/data,seed=1", "only with files"),
        ("org/data,seed=0", "only with files"),
        ("org/data,files=2,seed=-1", "negative"),
        ("org/data,weight=2", "only to a mix"),
    ):
        with pytest.raises(ValueError, match=needle):
            parse_lens_corpus([bad])


def test_corpus_mix_records_its_parts_and_weights_in_schedule_order() -> None:
    from drowse.lens.store import HfLensCorpus, LensCorpusMix, parse_lens_corpus

    mix = parse_lens_corpus([
        f"en/web:default@{_SHA},weight=3", f"zh/web:cmn@{_SHA},files=64",
    ])
    en = HfLensCorpus("en/web", "default", _SHA)
    zh = HfLensCorpus("zh/web", "cmn", _SHA, files=64)
    assert mix == LensCorpusMix((en, zh), (3, 1))
    assert mix.pinned
    assert mix.spec == f"mix:3*{en.spec}|1*{zh.spec}"
    assert str(mix) == f"{en},weight=3 + {zh},weight=1"
    assert not parse_lens_corpus(["en/web", "zh/web"]).pinned
    with pytest.raises(ValueError, match="at least 1"):
        parse_lens_corpus(["en/web,weight=0", "zh/web"])
    with pytest.raises(ValueError, match="one weight per corpus"):
        LensCorpusMix((en,), (1,))


def test_weighted_schedule_is_smooth_and_prefix_stable() -> None:
    import random

    from drowse.lens.store import _weighted_schedule

    assert _weighted_schedule([3, 1], 8, {}) == [0, 0, 1, 0, 0, 0, 1, 0]
    assert _weighted_schedule([1, 1, 1], 7, {}) == [0, 1, 2, 0, 1, 2, 0]
    # A dry stream leaves the schedule; the rest continue at their weights.
    assert _weighted_schedule([1, 1], 6, {1: 2}) == [0, 1, 0, 1, 0, 0]
    assert _weighted_schedule([1, 1], 9, {0: 1, 1: 2}) == [0, 1, 1]
    rng = random.Random(0)
    for _ in range(200):
        k = rng.randint(1, 5)
        weights = [rng.randint(1, 4) for _ in range(k)]
        caps = {i: rng.randint(0, 6) for i in range(k) if rng.random() < 0.5}
        full = _weighted_schedule(weights, 60, caps)
        for n in range(61):
            assert _weighted_schedule(weights, n, caps) == full[:n]
        for i, cap in caps.items():
            assert full.count(i) <= cap


class _FixedDocs:
    def __init__(self, docs: list[str]) -> None:
        self.docs = docs

    def prefix(self, n: int) -> list[str]:
        return self.docs[:n]


def test_interleave_is_prefix_stable_when_streams_run_dry() -> None:
    from drowse.lens.store import _Interleave

    streams = [
        _FixedDocs([f"a{i}" for i in range(9)]),
        _FixedDocs(["b0", "b1"]),
        _FixedDocs([f"c{i}" for i in range(5)]),
    ]
    mix = _Interleave(streams, [2, 1, 1])
    everything = mix.prefix(100)
    assert sorted(everything) == sorted(d for s in streams for d in s.docs)
    for n in range(len(everything) + 3):
        assert mix.prefix(n) == everything[:n]
    # Each stream's documents keep their own order.
    for stream in streams:
        assert [d for d in everything if d in stream.docs] == stream.docs
    assert everything[:4] == ["a0", "b0", "c0", "a1"]


def _fake_spread_dataset(
    monkeypatch: pytest.MonkeyPatch, files: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Serve ``files`` (repository path → texts) as one dataset's train files."""
    import datasets

    opened: list[dict[str, Any]] = []

    def fake_builder(repo: str, name: str | None = None, revision: str | None = None) -> Any:
        urls = [f"hf://datasets/{repo}@{revision}/{path}" for path in files]
        config = type("Config", (), {"name": name, "data_files": {"train": urls}})()
        return type("Builder", (), {"config": config})()

    def fake_load_dataset(repo: str, **kwargs: Any) -> Any:
        opened.append({"repo": repo, **kwargs})
        (path,) = kwargs["data_files"]["train"]
        return iter({"text": text} for text in files[path])

    monkeypatch.setattr(datasets, "load_dataset_builder", fake_builder)
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return opened


def test_spread_files_are_ordered_by_a_seeded_path_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens.store import HfLensCorpus, _spread_files

    paths = [f"data/cmn/train/{i:03d}.parquet" for i in range(10)]
    _fake_spread_dataset(monkeypatch, {path: [] for path in paths})

    def expected(seed: int, k: int) -> list[str]:
        return sorted(
            paths,
            key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).digest(),
        )[:k]

    corpus = HfLensCorpus("org/zh", "cmn", _SHA, files=3, seed=7)
    assert _spread_files(corpus) == expected(7, 3)
    assert _spread_files(HfLensCorpus("org/zh", "cmn", _SHA, files=3)) == expected(0, 3)
    assert expected(0, 3) != expected(7, 3)
    many = _spread_files(HfLensCorpus("org/zh", "cmn", _SHA, files=99))
    assert sorted(many) == paths


def test_spread_read_round_robins_its_files_and_outlasts_a_dry_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens.store import HfLensCorpus, _collect_lens_corpus, _spread_files

    files = {
        "f/0.parquet": ["x0", "", "x1"],
        "f/1.parquet": ["y0", "y1", "y2", "y3"],
        "f/2.parquet": ["z0"],
    }
    opened = _fake_spread_dataset(monkeypatch, files)
    corpus = HfLensCorpus("org/zh", "cmn", _SHA, files=2, seed=1)
    first, second = _spread_files(corpus)
    left = [d for d in files[first] if d]
    right = [d for d in files[second] if d]
    expected: list[str] = []
    while left or right:
        for stream in (left, right):
            if stream:
                expected.append(stream.pop(0))
    docs, spec = _collect_lens_corpus(6, corpus)
    assert docs == expected[:6]
    assert spec == corpus.spec
    assert {o["data_files"]["train"][0] for o in opened} <= {first, second}
    assert all(
        (o["name"], o["revision"], o["split"], o["streaming"])
        == ("cmn", _SHA, "train", True)
        for o in opened
    )
    for n in range(8):
        assert _collect_lens_corpus(n, corpus)[0] == _collect_lens_corpus(7, corpus)[0][:n]


def test_mix_read_interleaves_its_corpora_by_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    from drowse.lens.store import HfLensCorpus, LensCorpusMix, _collect_lens_corpus

    rows = {
        "en/web": [f"en{i}" for i in range(20)],
        "zh/web": ["", *(f"zh{i}" for i in range(3))],
    }
    monkeypatch.setattr(
        datasets, "load_dataset",
        lambda repo, **_kwargs: iter({"text": text} for text in rows[repo]),
    )
    mix = LensCorpusMix(
        (HfLensCorpus("en/web", "default", _SHA), HfLensCorpus("zh/web", "default", _SHA)),
        (3, 1),
    )
    docs, spec = _collect_lens_corpus(8, mix)
    assert docs == ["en0", "en1", "zh0", "en2", "en3", "en4", "zh1", "en5"]
    assert spec == mix.spec
    # The Chinese corpus runs dry after three documents; English fills in.
    long, _spec = _collect_lens_corpus(20, mix)
    assert long == [
        *docs, "en6", "en7", "zh2", *(f"en{i}" for i in range(8, 17)),
    ]


def test_pin_resolves_revision_to_a_commit_and_names_the_default_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets
    import huggingface_hub

    from drowse.lens.store import HfLensCorpus, pin_lens_corpus

    asked: list[tuple[str, str | None]] = []

    class FakeApi:
        def dataset_info(self, repo: str, revision: str | None = None) -> Any:
            asked.append((repo, revision))
            return type("Info", (), {"sha": _SHA})()

    def fake_builder(repo: str, revision: str | None = None) -> Any:
        assert revision == _SHA  # the config is read at the pinned commit
        config = type("Config", (), {"name": "default"})()
        return type("Builder", (), {"config": config})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(datasets, "load_dataset_builder", fake_builder)
    pinned = pin_lens_corpus(HfLensCorpus("org/data", None, "main", "body"))
    assert pinned == HfLensCorpus("org/data", "default", _SHA, "body")
    assert asked == [("org/data", "main")]
    # Pinning is idempotent and makes no further Hub call.
    assert pin_lens_corpus(pinned) is pinned
    assert asked == [("org/data", "main")]


def test_read_lens_documents_skips_empty_text_without_counting_it() -> None:
    from drowse.lens.jlens import JacobianLensError
    from drowse.lens.store import LENS_DOC_CHARS, _read_lens_documents

    rows = [
        {"text": ""}, {"text": "  first  "}, {"text": None}, {"text": " \n\t"},
        {"text": "second"}, {"text": "x" * (LENS_DOC_CHARS + 10)}, {"text": "fourth"},
    ]
    assert _read_lens_documents(rows, 2) == ["first", "second"]
    assert _read_lens_documents(rows, 3)[-1] == "x" * LENS_DOC_CHARS
    assert _read_lens_documents(rows, 99) == [
        "first", "second", "x" * LENS_DOC_CHARS, "fourth",
    ]
    assert _read_lens_documents([{"body": " b ", "text": "t"}], 1, "body") == ["b"]
    with pytest.raises(JacobianLensError, match="no 'body' field.*text"):
        _read_lens_documents([{"text": "t"}], 1, "body")
    with pytest.raises(JacobianLensError, match="holds list, not text"):
        _read_lens_documents([{"text": ["a", "b"]}], 1)


def _fake_hub_dataset(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Serve ``rows`` as every streamed dataset, recording each open."""
    import datasets

    opened: list[dict[str, Any]] = []

    def fake_load_dataset(repo: str, **kwargs: Any) -> Any:
        opened.append({"repo": repo, **kwargs})
        return iter(rows)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return opened


def test_hf_corpus_read_is_the_pinned_train_split_by_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens.store import HfLensCorpus, _collect_lens_corpus

    rows = [{"text": t} for t in ("a doc", "", "b doc", None, "c doc", "d doc")]
    opened = _fake_hub_dataset(monkeypatch, rows)
    corpus = HfLensCorpus("org/shuffled", "default", _SHA)
    small, small_spec = _collect_lens_corpus(2, corpus)
    large, large_spec = _collect_lens_corpus(4, corpus)
    assert small == ["a doc", "b doc"]
    assert large == ["a doc", "b doc", "c doc", "d doc"]
    assert large[: len(small)] == small
    assert small_spec == large_spec == corpus.spec
    assert opened == [{
        "repo": "org/shuffled", "name": "default", "split": "train",
        "streaming": True, "revision": _SHA,
    }] * 2


def test_corpus_stream_terminates_a_worker_stuck_after_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a cancel event the read still runs in the worker, and a worker
    that never exits (pyarrow's pool after an early-stopped stream) is
    terminated once its result arrives."""
    import drowse.lens.store as lens_io
    from drowse.lens.store import HfLensCorpus

    corpus = HfLensCorpus("org/data", "default", _SHA)

    class StuckProcess:
        terminated = False

        def is_alive(self) -> bool:
            return not self.terminated

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            raise AssertionError("terminate suffices")

        def join(self, timeout: float = 0) -> None:
            del timeout

    class ResultReceiver:
        closed = False

        def poll(self, timeout: float) -> bool:
            del timeout
            return True

        def recv(self) -> object:
            return ("ok", (["doc"], corpus.spec))

        def close(self) -> None:
            self.closed = True

    process, receiver = StuckProcess(), ResultReceiver()
    started: list[tuple[int, HfLensCorpus]] = []

    def start(n: int, started_corpus: HfLensCorpus) -> tuple[Any, Any]:
        started.append((n, started_corpus))
        return process, receiver

    monkeypatch.setattr(lens_io, "_start_lens_corpus_process", start)
    assert stream_lens_corpus(5, corpus) == (["doc"], corpus.spec)
    assert started == [(5, corpus)]
    assert process.terminated and receiver.closed
