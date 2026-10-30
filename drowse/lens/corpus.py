"""Hub text corpora for lens fits and SAE training.

A corpus is the first N non-empty documents of a Hugging Face dataset's train
split (:class:`HfLensCorpus`), or several interleaved by weight
(:class:`LensCorpusMix`), pinned to a commit before it is read so the
recorded ``corpus_spec`` is exact provenance.  Streaming always runs in an
isolated spawned process (:func:`stream_lens_corpus`).  A local file corpus
(:func:`read_corpus_file`) applies the same per-document slice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast, overload

from drowse.errors import SourceSpecError

log = logging.getLogger(__name__)

#: Bumped when document preprocessing changes, so a fit's recorded policy
#: tells resumed and re-run fits apart.
LENS_CORPUS_PREPROCESS_VERSION = 1


#: Documents are sliced to this many characters before tokenization — the fit
#: truncates to ``seq_len`` tokens anyway, so tokenizing a full web page is
#: waste.
LENS_DOC_CHARS = 4000
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_DATASETS_MISSING = (
    "Hugging Face lens corpora stream via the optional `datasets` "
    "dependency — `pip install 'drowse.ai[hf]'`, or supply a corpus "
    "file (one document per line)"
)
_SOURCE_OPTIONS = ("text_field", "files", "seed", "weight")
#: Streams an interleave reads at once.  Opening a Hub data file costs one
#: round trip plus its first row group (seconds each), so a read spread over
#: dozens of files is latency-bound; every stream's documents are fixed by
#: its own position in the data, so the concurrency never changes the result.
_CORPUS_READ_THREADS = 8


@dataclass(frozen=True)
class HfLensCorpus:
    """A Hugging Face dataset read as a lens-fit corpus.

    The corpus is the first N non-empty ``text_field`` values of the train
    split, each stripped and sliced to :data:`LENS_DOC_CHARS`.  With
    ``files=K`` they come round-robin from K of the split's data files
    instead, ordered by a ``seed``-keyed hash of their paths, which spreads a
    sample over a dataset whose rows are not shuffled.  Either way the first
    N documents of a longer read are exactly those of a shorter one, so a fit
    grows or resumes by token-id prefix; on a globally shuffled dataset the
    head is a random sample and growing a fit adds a nested one.  ``revision``
    is any branch, tag, or commit until :func:`pin_lens_corpus` fixes it to a
    commit and names the config, which makes :attr:`spec` the recorded
    provenance.
    """

    repo: str
    config: str | None = None
    revision: str | None = None
    text_field: str = "text"
    files: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.text_field:
            raise SourceSpecError("the corpus text field must not be empty")
        if self.files is not None and self.files < 1:
            raise SourceSpecError("files must be at least 1")
        if self.seed < 0:
            raise SourceSpecError("seed must not be negative")
        if self.seed and self.files is None:
            raise SourceSpecError("seed applies only with files=K")

    @property
    def pinned(self) -> bool:
        return self.config is not None and bool(
            _COMMIT_SHA.fullmatch(self.revision or "")
        )

    @property
    def spec(self) -> str:
        """Provenance recorded as the fit's ``corpus_spec``.

        ``hf:REPO/CONFIG@COMMIT;preprocess=…;doc_chars=…``, plus
        ``;text_field=…`` when the text is not in ``text`` and
        ``;files=…;seed=…`` for a read spread over files, so the default
        corpus keeps the spec its existing fits recorded.
        """
        if not self.pinned:
            raise ValueError(f"{self} is not pinned to a config and commit")
        spec = (
            f"hf:{self.repo}/{self.config}@{self.revision};preprocess="
            f"{LENS_CORPUS_PREPROCESS_VERSION};doc_chars={LENS_DOC_CHARS}"
        )
        if self.text_field != "text":
            spec += f";text_field={self.text_field}"
        if self.files is not None:
            spec += f";files={self.files};seed={self.seed}"
        return spec

    def __str__(self) -> str:
        text = self.repo
        if self.config is not None:
            text += f":{self.config}"
        if self.revision is not None:
            text += f"@{self.revision}"
        if self.text_field != "text":
            text += f",text_field={self.text_field}"
        if self.files is not None:
            text += f",files={self.files},seed={self.seed}"
        return text


@dataclass(frozen=True)
class LensCorpusMix:
    """Several Hub corpora interleaved by integer weight.

    Documents are drawn by smooth weighted round-robin, ties going to the
    earlier corpus, so weights 3,1 read A A B A A A B A …; a corpus that runs
    out leaves the schedule.  The schedule never depends on N, so the mix
    keeps the prefix property of its parts and a fit on it resumes and grows
    the same way.
    """

    parts: tuple[HfLensCorpus, ...]
    weights: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.parts) < 2 or len(self.weights) != len(self.parts):
            raise SourceSpecError("a corpus mix needs one weight per corpus, ≥ 2")
        if any(weight < 1 for weight in self.weights):
            raise SourceSpecError("corpus weights must be at least 1")

    @property
    def pinned(self) -> bool:
        return all(part.pinned for part in self.parts)

    @property
    def spec(self) -> str:
        """``mix:W*SPEC|W*SPEC…``, each part's own spec in schedule order."""
        return "mix:" + "|".join(
            f"{weight}*{part.spec}"
            for part, weight in zip(self.parts, self.weights)
        )

    def __str__(self) -> str:
        return " + ".join(
            f"{part},weight={weight}"
            for part, weight in zip(self.parts, self.weights)
        )


LensCorpus = HfLensCorpus | LensCorpusMix

#: Default web-text corpus for a lens fit — the paper-parity pretraining-like
#: sample.  Shared by CLI ``lens fit`` / ``sae train`` and the server's fit and
#: train routes.
DEFAULT_LENS_CORPUS = HfLensCorpus("HuggingFaceFW/fineweb-edu", "sample-10BT")


def parse_lens_corpus(sources: Sequence[str] | None) -> LensCorpus:
    """The corpus named by CLI ``--hf-corpus`` values or the fit route's list.

    Each source is ``REPO[:CONFIG][@REV][,KEY=VALUE…]`` with keys
    ``text_field`` (default ``text``), ``files`` (spread the read over that
    many data files), ``seed`` (their order, default 0), and ``weight`` (a mix
    member's share, default 1).  Several sources make a
    :class:`LensCorpusMix`; none is the default corpus.  Raises
    :class:`~drowse.errors.SourceSpecError` for a malformed source.
    """
    if not sources:
        return DEFAULT_LENS_CORPUS
    parsed = [_parse_source(raw) for raw in sources]
    if len(parsed) == 1:
        corpus, weight = parsed[0]
        if weight is not None:
            raise SourceSpecError("weight applies only to a mix of several corpora")
        return corpus
    return LensCorpusMix(
        tuple(corpus for corpus, _weight in parsed),
        tuple(1 if weight is None else weight for _corpus, weight in parsed),
    )


def _parse_source(raw: str) -> tuple[HfLensCorpus, int | None]:
    source, *options = raw.strip().split(",")
    head, at, revision = source.partition("@")
    repo, colon, config = head.partition(":")
    if (
        not repo or any(ch.isspace() for ch in source)
        or (colon and not config) or (at and not revision)
        or ":" in revision
    ):
        raise SourceSpecError(
            f"bad Hugging Face corpus {raw!r} (want "
            "REPO[:CONFIG][@REV][,KEY=VALUE…], e.g. "
            "HuggingFaceFW/dclm_100BT-shuffled)"
        )
    values: dict[str, str] = {}
    for option in options:
        key, eq, value = option.partition("=")
        if key not in _SOURCE_OPTIONS or not eq or not value or key in values:
            raise SourceSpecError(
                f"bad corpus option {option!r} in {raw!r} (want KEY=VALUE, "
                f"KEY one of {', '.join(_SOURCE_OPTIONS)}, each at most once)"
            )
        values[key] = value
    try:
        ints = {
            key: int(values[key])
            for key in ("files", "seed", "weight") if key in values
        }
    except ValueError:
        raise SourceSpecError(
            f"files, seed, and weight take integers, in {raw!r}"
        ) from None
    weight = ints.get("weight")
    if weight is not None and weight < 1:
        raise SourceSpecError("corpus weights must be at least 1")
    corpus = HfLensCorpus(
        repo, config or None, revision or None,
        values.get("text_field", "text"), ints.get("files"),
        ints.get("seed", 0),
    )
    if "seed" in ints and corpus.files is None:
        raise SourceSpecError("seed applies only with files=K")
    return corpus, weight


@overload
def pin_lens_corpus(corpus: HfLensCorpus = ...) -> HfLensCorpus: ...


@overload
def pin_lens_corpus(corpus: LensCorpusMix) -> LensCorpusMix: ...


def pin_lens_corpus(corpus: LensCorpus = DEFAULT_LENS_CORPUS) -> LensCorpus:
    """Fix ``corpus`` (each part of a mix) to a Hub commit and named config.

    An already pinned source is returned as is, without a Hub call.  An
    omitted config resolves to the dataset's default at that commit, which
    needs the optional ``datasets`` dependency.
    """
    if isinstance(corpus, LensCorpusMix):
        return replace(
            corpus, parts=tuple(_pin_source(part) for part in corpus.parts),
        )
    return _pin_source(corpus)


def _pin_source(corpus: HfLensCorpus) -> HfLensCorpus:
    from drowse.lens.jlens import JacobianLensError

    if corpus.pinned:
        return corpus
    try:
        from drowse.base.hf import _hf_api

        revision = _hf_api().dataset_info(
            corpus.repo, revision=corpus.revision,
        ).sha
    except Exception as exc:
        raise JacobianLensError(
            f"could not resolve an immutable revision for {corpus}: {exc}"
        ) from exc
    if not revision:
        raise JacobianLensError(
            f"Hugging Face returned no immutable dataset revision for {corpus}"
        )
    config = corpus.config
    if config is None:
        try:
            from datasets import load_dataset_builder
        except ImportError as e:
            raise JacobianLensError(_DATASETS_MISSING) from e
        try:
            config = load_dataset_builder(
                corpus.repo, revision=revision,
            ).config.name
        except Exception as exc:
            raise JacobianLensError(
                f"could not resolve the default config of {corpus}; name one "
                f"as REPO:CONFIG ({exc})"
            ) from exc
    return replace(corpus, config=str(config), revision=str(revision))


def stream_lens_corpus(
    n: int,
    corpus: LensCorpus = DEFAULT_LENS_CORPUS,
    *,
    cancel_event: threading.Event | None = None,
) -> tuple[list[str], str]:
    """Stream the first ``n`` documents of a Hub corpus (default web text).

    Returns ``(documents, corpus_spec)``.  Needs the optional ``datasets``
    dependency (``pip install 'drowse.ai[hf]'``); raises
    :class:`~drowse.lens.jlens.JacobianLensError` without it so both the CLI
    and the fit route surface the same actionable message.  Hub resolution and
    streaming always run in an isolated spawned process, which is terminated
    once it has sent its result.  A cancellation event can therefore abandon a
    blocked provider read without trapping the server in ``cancelling…``, and
    no caller inherits the provider's threads: after a stream stops early,
    pyarrow's I/O pool can block interpreter exit forever (every time on
    ``HuggingFaceFW/dclm_100BT-shuffled``), even though the read returned.
    """
    from drowse.lens.jlens import JacobianLensCancelled, JacobianLensError

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if cancelled():
        raise JacobianLensCancelled(
            "Jacobian-lens fit cancelled while preparing its corpus"
        )

    process, receiver = _start_lens_corpus_process(n, corpus)
    try:
        while True:
            if cancelled():
                raise JacobianLensCancelled(
                    "Jacobian-lens fit cancelled while streaming its corpus"
                )
            ready = receiver.poll(0.1)
            # A click that lands while ``poll`` is waiting must win over a
            # simultaneously-ready result; once the user asks to cancel, do
            # not unexpectedly advance into the hours-long estimator phase.
            if cancelled():
                raise JacobianLensCancelled(
                    "Jacobian-lens fit cancelled while streaming its corpus"
                )
            if ready:
                payload = receiver.recv()
                kind = payload[0]
                if kind == "ok":
                    return cast(tuple[list[str], str], payload[1])
                if kind == "typed_error":
                    raise JacobianLensError(str(payload[1]))
                raise RuntimeError(
                    f"lens corpus worker failed ({payload[1]})"
                )
            if not process.is_alive():
                raise RuntimeError(
                    "lens corpus worker exited without a result"
                )
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        else:
            process.join(timeout=0)


def _collect_lens_corpus(
    n: int, corpus: LensCorpus,
) -> tuple[list[str], str]:
    """Direct provider read; runs inside the isolated corpus worker."""
    from drowse.lens.jlens import JacobianLensError

    try:
        import datasets  # noqa: F401 — fail early with the install hint
    except ImportError as e:
        raise JacobianLensError(_DATASETS_MISSING) from e

    pinned = pin_lens_corpus(corpus)
    return _document_stream(pinned).prefix(n), pinned.spec


class _DocumentStream(Protocol):
    def prefix(self, n: int) -> list[str]:
        """The stream's first ``n`` documents, fewer once it runs dry."""
        ...


def _document_stream(corpus: LensCorpus) -> _DocumentStream:
    if isinstance(corpus, LensCorpusMix):
        return _Interleave(
            [_document_stream(part) for part in corpus.parts], corpus.weights,
        )
    if corpus.files is None:
        return _Documents(corpus)
    files = _spread_files(corpus)
    return _Interleave(
        [_Documents(corpus, data_file) for data_file in files], [1] * len(files),
    )


def _spread_files(corpus: HfLensCorpus) -> list[str]:
    """The first ``files`` train data files of a pinned source, in the order
    of a ``seed``-keyed hash of their repository paths."""
    from datasets import load_dataset_builder

    from drowse.lens.jlens import JacobianLensError

    try:
        data_files = load_dataset_builder(
            corpus.repo, name=corpus.config, revision=corpus.revision,
        ).config.data_files
        urls = [str(url) for url in (data_files or {}).get("train", [])]
    except Exception as exc:
        raise JacobianLensError(
            f"could not list the train files of {corpus}: {exc}"
        ) from exc
    if not urls:
        raise JacobianLensError(f"{corpus} has no train data files to spread over")
    marker = f"@{corpus.revision}/"
    paths = [url.split(marker, 1)[1] if marker in url else url for url in urls]
    paths.sort(
        key=lambda path: hashlib.sha256(
            f"{corpus.seed}:{path}".encode("utf-8")
        ).digest()
    )
    assert corpus.files is not None
    return paths[: corpus.files]


class _Documents:
    """A pinned source's non-empty documents, from its whole train split or
    one of its data files; a longer request rereads from the start."""

    def __init__(self, corpus: HfLensCorpus, data_file: str | None = None) -> None:
        self._corpus = corpus
        self._data_file = data_file
        self._docs: list[str] = []
        self._exhausted = False

    def prefix(self, n: int) -> list[str]:
        if len(self._docs) < n and not self._exhausted:
            self._docs = _read_lens_documents(
                self._rows(), n, self._corpus.text_field,
            )
            self._exhausted = len(self._docs) < n
        return self._docs[:n]

    def _rows(self) -> Iterable[Mapping[str, Any]]:
        from datasets import load_dataset

        from drowse.lens.jlens import JacobianLensError

        corpus = self._corpus
        extra: dict[str, Any] = {}
        where = str(corpus)
        if self._data_file is not None:
            extra["data_files"] = {"train": [self._data_file]}
            where = f"{self._data_file} of {corpus}"
        try:
            return load_dataset(
                corpus.repo, name=corpus.config, split="train", streaming=True,
                revision=corpus.revision, **extra,
            )
        except Exception as exc:
            raise JacobianLensError(f"could not open {where}: {exc}") from exc


class _Interleave:
    """Document streams merged by :func:`_weighted_schedule`."""

    def __init__(
        self, streams: Sequence[_DocumentStream], weights: Sequence[int],
    ) -> None:
        self._streams = list(streams)
        self._weights = list(weights)

    def prefix(self, n: int) -> list[str]:
        # Schedule assuming every stream is long enough; a stream that comes
        # up short is capped at what it has and the schedule is redrawn, which
        # equals drawing lazily and dropping a stream when it runs dry.
        from concurrent.futures import ThreadPoolExecutor

        caps: dict[int, int] = {}
        while True:
            order = _weighted_schedule(self._weights, n, caps)
            counts = Counter(order)
            with ThreadPoolExecutor(
                max_workers=max(1, min(_CORPUS_READ_THREADS, len(counts))),
                thread_name_prefix="drowse-lens-corpus",
            ) as pool:
                docs = dict(zip(counts, pool.map(
                    lambda item: self._streams[item[0]].prefix(item[1]),
                    counts.items(),
                )))
            short = {k: len(docs[k]) for k, c in counts.items() if len(docs[k]) < c}
            if not short:
                break
            caps.update(short)
        taken = Counter[int]()
        merged: list[str] = []
        for k in order:
            merged.append(docs[k][taken[k]])
            taken[k] += 1
        return merged


def _weighted_schedule(
    weights: Sequence[int], n: int, caps: Mapping[int, int],
) -> list[int]:
    """The stream index of each of the first ``n`` merged documents.

    Smooth weighted round-robin: each pick credits every live stream with its
    weight, takes the stream with the most credit (the earlier on a tie), and
    charges it the live total.  A stream that has given ``caps[k]`` documents
    is dry: it leaves the schedule and the pick is retaken among the rest.
    """
    credit = [0] * len(weights)
    taken = [0] * len(weights)
    live = list(range(len(weights)))
    order: list[int] = []
    while len(order) < n and live:
        for k in live:
            credit[k] += weights[k]
        while live:
            best = max(live, key=lambda k: (credit[k], -k))
            if best in caps and taken[best] >= caps[best]:
                live.remove(best)
                continue
            credit[best] -= sum(weights[k] for k in live)
            taken[best] += 1
            order.append(best)
            break
    return order


def _read_lens_documents(
    rows: Iterable[Mapping[str, Any]], n: int, text_field: str = "text",
) -> list[str]:
    """The first ``n`` non-empty ``text_field`` values of ``rows``.

    Each is stripped and sliced to :data:`LENS_DOC_CHARS`; empty, blank, and
    null values are skipped and do not count toward ``n``.
    """
    from drowse.lens.jlens import JacobianLensError

    docs: list[str] = []
    for row in rows:
        if text_field not in row:
            raise JacobianLensError(
                f"corpus rows have no {text_field!r} field "
                f"(fields: {', '.join(map(str, row))})"
            )
        value = row[text_field]
        if value is None:
            continue
        if not isinstance(value, str):
            raise JacobianLensError(
                f"corpus field {text_field!r} holds "
                f"{type(value).__name__}, not text"
            )
        text = value.strip()
        if text:
            docs.append(text[:LENS_DOC_CHARS])
        if len(docs) >= n:
            break
    return docs


def _lens_corpus_process(n: int, corpus: LensCorpus, sender: Any) -> None:
    """Spawn target that transports only JSON/pickle-safe result payloads."""
    from drowse.lens.jlens import JacobianLensError

    try:
        payload: tuple[str, object] = ("ok", _collect_lens_corpus(n, corpus))
    except JacobianLensError as exc:
        payload = ("typed_error", str(exc))
    except BaseException as exc:  # noqa: BLE001 — preserve worker liveness
        log.exception("lens corpus worker failed")
        payload = ("error", type(exc).__name__)
    try:
        sender.send(payload)
    except (BrokenPipeError, EOFError, OSError):
        # Parent cancellation closes the receive end before terminating us.
        # Do not turn that expected race into a second worker traceback.
        pass
    finally:
        sender.close()


def _start_lens_corpus_process(
    n: int, corpus: LensCorpus,
) -> tuple[Any, Any]:
    """Start a spawn-only provider process, safe beside initialized MPS."""
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_lens_corpus_process,
        args=(n, corpus, sender),
        name="drowse-jlens-corpus",
    )
    process.start()
    sender.close()
    return process, receiver


def read_corpus_file(path: Path, n: int) -> list[str]:
    """The first ``n`` documents of a local corpus file, one per line.

    A JSON-object line contributes its ``text`` field; blank lines are
    skipped; each document is sliced to :data:`LENS_DOC_CHARS`, exactly like
    a streamed Hub document.
    """
    docs: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    line = str(json.loads(line).get("text", "")) or line
                except json.JSONDecodeError:
                    pass
            docs.append(line[:LENS_DOC_CHARS])
            if len(docs) >= n:
                break
    return docs
