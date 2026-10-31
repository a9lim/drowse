"""Hub corpus specs, schedules, pinning, and the isolated corpus streamer."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

import pytest

from drowse.lens.corpus import stream_lens_corpus
from drowse.lens.jlens import JacobianLensCancelled


def test_default_corpus_cancel_does_not_wait_for_blocked_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web cancel button must not depend on a blocked Hub read returning."""
    import drowse.lens.corpus as lens_io

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
        lambda _job, *_args: (process, receiver),
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
    import drowse.lens.corpus as lens_io

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
        lambda _job, *_args: (FinishedProcess(), receiver),
    )

    with pytest.raises(JacobianLensCancelled, match="streaming its corpus"):
        stream_lens_corpus(1, cancel_event=cancelled)
    assert receiver.closed


# ------------------------------------------------- Hugging Face lens corpora --

_SHA = "0123456789abcdef0123456789abcdef01234567"


def _source(raw: str) -> Any:
    from drowse.lens.corpus import HfLensCorpus, parse_lens_corpus

    corpus = parse_lens_corpus([raw])
    assert isinstance(corpus, HfLensCorpus)
    return corpus


def test_hf_corpus_spec_records_repo_config_commit_and_preprocessing() -> None:
    from drowse.lens.corpus import (
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
    from drowse.lens.corpus import DEFAULT_LENS_CORPUS, HfLensCorpus, parse_lens_corpus

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
    from drowse.lens.corpus import HfLensCorpus, LensCorpusMix, parse_lens_corpus

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

    from drowse.lens.corpus import _weighted_schedule

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
    from drowse.lens.corpus import _Interleave

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
    from drowse.lens.corpus import HfLensCorpus, _spread_files

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
    from drowse.lens.corpus import HfLensCorpus, _collect_lens_corpus, _spread_files

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

    from drowse.lens.corpus import HfLensCorpus, LensCorpusMix, _collect_lens_corpus

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
    """The resolution the corpus worker runs (in process here)."""
    import datasets
    import huggingface_hub

    from drowse.lens.corpus import HfLensCorpus, _pin_in_process

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
    pinned = _pin_in_process(HfLensCorpus("org/data", None, "main", "body"))
    assert pinned == HfLensCorpus("org/data", "default", _SHA, "body")
    assert asked == [("org/data", "main")]
    # Pinning is idempotent and makes no further Hub call.
    assert _pin_in_process(pinned) is pinned
    assert asked == [("org/data", "main")]


def test_read_lens_documents_skips_empty_text_without_counting_it() -> None:
    from drowse.lens.jlens import JacobianLensError
    from drowse.lens.corpus import LENS_DOC_CHARS, _read_lens_documents

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
    from drowse.lens.corpus import HfLensCorpus, _collect_lens_corpus

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
    import drowse.lens.corpus as lens_io
    from drowse.lens.corpus import HfLensCorpus

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
    started: list[tuple[Any, tuple[Any, ...]]] = []

    def start(job: Any, *args: Any) -> tuple[Any, Any]:
        started.append((job, args))
        return process, receiver

    monkeypatch.setattr(lens_io, "_start_lens_corpus_process", start)
    assert stream_lens_corpus(5, corpus) == (["doc"], corpus.spec)
    assert started == [(lens_io._collect_lens_corpus, (5, corpus))]
    assert process.terminated and receiver.closed


class _ResultWorker:
    """A corpus worker process (and its pipe end) that already sent ``result``."""

    closed = False

    def __init__(self, result: Any) -> None:
        self.result = result

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float = 0) -> None:
        del timeout

    def poll(self, timeout: float) -> bool:
        del timeout
        return True

    def recv(self) -> object:
        return ("ok", self.result)

    def close(self) -> None:
        self.closed = True


def test_pin_resolves_an_unpinned_corpus_in_the_corpus_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub resolution never runs in the caller (the fit route's server
    process): an unpinned corpus is pinned by the isolated worker."""
    import drowse.base.hf as base_hf
    import drowse.lens.corpus as lens_io
    from drowse.lens.corpus import HfLensCorpus, LensCorpusMix, pin_lens_corpus

    def in_process_hub(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("Hub resolution must run in the corpus worker")

    corpus = LensCorpusMix((
        HfLensCorpus("org/data", "cfg", "main"),
        HfLensCorpus("org/zh", None, _SHA),
    ), (2, 1))
    pinned = LensCorpusMix((
        HfLensCorpus("org/data", "cfg", _SHA),
        HfLensCorpus("org/zh", "default", _SHA),
    ), (2, 1))
    started: list[tuple[Any, tuple[Any, ...]]] = []
    worker = _ResultWorker(pinned)

    def start(job: Any, *args: Any) -> tuple[Any, Any]:
        started.append((job, args))
        return worker, worker

    monkeypatch.setattr(base_hf, "_hf_api", in_process_hub)
    monkeypatch.setattr(lens_io, "_start_lens_corpus_process", start)
    assert pin_lens_corpus(corpus) == pinned
    assert started == [(lens_io._pin_in_process, (corpus,))]
    assert worker.closed
    # An already pinned corpus starts no worker.
    started.clear()
    assert pin_lens_corpus(pinned) is pinned
    assert started == []


def test_corpus_worker_target_runs_its_job_and_sends_a_picklable_result() -> None:
    """The spawn target runs ``job(*args)`` and sends the result; both jobs
    are module-level functions, so they cross the spawn boundary by name."""
    import pickle

    import drowse.lens.corpus as lens_io
    from drowse.lens.corpus import HfLensCorpus

    class Sender:
        sent: list[object] = []
        closed = False

        def send(self, payload: object) -> None:
            self.sent.append(payload)

        def close(self) -> None:
            self.closed = True

    pinned = HfLensCorpus("org/data", "cfg", _SHA)
    sender = Sender()
    lens_io._lens_corpus_process(lens_io._pin_in_process, (pinned,), sender)
    assert sender.sent == [("ok", pinned)] and sender.closed
    for job in (lens_io._pin_in_process, lens_io._collect_lens_corpus):
        assert pickle.loads(pickle.dumps(job)) is job


def test_pin_cancel_abandons_a_blocked_hub_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.corpus as lens_io
    from drowse.lens.corpus import HfLensCorpus, pin_lens_corpus

    cancelled = threading.Event()

    class BlockedWorker:
        alive = True
        terminated = False

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float = 0) -> None:
            del timeout

        def poll(self, timeout: float) -> bool:
            cancelled.set()  # the click lands while the Hub call blocks
            return False

        def recv(self) -> object:
            raise AssertionError("blocked resolution has no result")

        def close(self) -> None:
            pass

    worker = BlockedWorker()
    monkeypatch.setattr(
        lens_io, "_start_lens_corpus_process", lambda _job, *_args: (worker, worker),
    )
    with pytest.raises(JacobianLensCancelled, match="resolving its corpus"):
        pin_lens_corpus(HfLensCorpus("org/data", "cfg", "main"), cancel_event=cancelled)
    assert worker.terminated


def test_corpus_file_reads_text_lines_and_refuses_missing_or_empty_files(
    tmp_path: Any,
) -> None:
    """The ``lens fit --corpus`` / ``sae train --corpus`` reader: JSON lines
    contribute their ``text``, blanks are skipped, and a missing or empty
    file is a typed error (exit 1 / exit 2 at the CLI)."""
    from drowse.errors import RequestError
    from drowse.lens.corpus import EmptyCorpusError, read_corpus_file

    path = tmp_path / "corpus.txt"
    path.write_text('first\n\n{"text": "second"}\nthird\n')
    assert read_corpus_file(path, 2) == ["first", "second"]
    with pytest.raises(FileNotFoundError, match="corpus file not found"):
        read_corpus_file(tmp_path / "missing.txt", 2)
    path.write_text("\n  \n")
    with pytest.raises(EmptyCorpusError, match="no non-empty documents") as err:
        read_corpus_file(path, 2)
    assert isinstance(err.value, RequestError) and err.value.user_message()[0] == 400
