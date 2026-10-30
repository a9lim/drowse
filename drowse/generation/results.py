from __future__ import annotations
import csv
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from drowse.readout.types import ProbeReading

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TokenAlt:
    """One alternative the model considered at a given position.

    Captured at decode time when ``SamplingConfig.return_top_k > 0`` (or
    when OpenAI ``logprobs`` is set to a positive int).  ``logprob`` is the
    post-sampler, post-temperature natural-log probability under the same
    distribution the chosen token was drawn from — that's the calibrated
    quantity for "how surprising was this token to the configured sampler."
    """
    id: int
    text: str
    logprob: float


@dataclass
class ProbeReadings:
    """One probe's aggregate readings across a *series* of generations.

    The multi-run summary shape (:class:`ProbeReading`, singular, is the
    single-read shape — per token or the end-of-generation aggregate).
    Produced by :meth:`RunSet.probe_summary` — which returns exactly the
    ``dict[str, ProbeReadings]`` that
    ``drowse.notebook.plots.plot_trait_history`` consumes, so the plot is
    one call off a batch or sweep.

    Vectorized for the coordinate readout: a probe is a rank-``R`` flat
    subspace (``R = 1`` for a 2-node concept axis), so each per-generation
    sample is the aggregate coordinate ``tuple[float, ...]`` of length
    ``R`` and the summary statistics are reported per axis.  ``mean`` /
    ``std`` / ``min`` / ``max`` / ``delta_per_gen`` are each an
    ``R``-tuple aligned with the coordinate axes; ``per_generation`` is
    the list of per-run coordinate tuples.  A scalar consumer reads axis
    0 (``mean[0]``).

    ``delta_per_gen`` is the **drift rate**: ``(last − first) / (n − 1)``
    per axis, the mean change per generation across the series (``0.0`` for
    a single run).  ``std`` is the population standard deviation.
    """
    per_generation: list[tuple[float, ...]]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    min: tuple[float, ...]
    max: tuple[float, ...]
    delta_per_gen: tuple[float, ...]

    @classmethod
    def from_coords(
        cls, samples: Iterable[Iterable[float]],
    ) -> "ProbeReadings":
        """Fold a series of per-generation coordinate tuples into a summary.

        Ragged input is truncated to the narrowest sample's rank so every
        statistic stays axis-aligned — a probe re-attached at a different
        rank mid-series is a caller error, not something to interpolate.
        """
        rows = [tuple(float(v) for v in sample) for sample in samples]
        rows = [row for row in rows if row]
        if not rows:
            empty: tuple[float, ...] = ()
            return cls(
                per_generation=[], mean=empty, std=empty,
                min=empty, max=empty, delta_per_gen=empty,
            )
        rank = min(len(row) for row in rows)
        rows = [row[:rank] for row in rows]
        n = len(rows)
        columns = [[row[axis] for row in rows] for axis in range(rank)]
        mean = tuple(sum(col) / n for col in columns)
        std = tuple(
            (sum((v - m) ** 2 for v in col) / n) ** 0.5
            for col, m in zip(columns, mean)
        )
        delta = tuple(
            (col[-1] - col[0]) / (n - 1) if n > 1 else 0.0 for col in columns
        )
        return cls(
            per_generation=rows,
            mean=mean,
            std=std,
            min=tuple(min(col) for col in columns),
            max=tuple(max(col) for col in columns),
            delta_per_gen=delta,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Result of a generation call."""
    text: str
    tokens: list[int]
    token_count: int
    tok_per_sec: float
    elapsed: float
    # Active-steering coefficient map at generation time: term -> coefficient.
    steering_alphas: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    finish_reason: str = "stop"
    # Per-completion-token ``(token_id, logprob, top_alts)`` — populated
    # only when chosen-logprob capture is live (any ``on_token`` consumer
    # or an explicit logprobs request). ``top_alts`` is a list of
    # :class:`TokenAlt` carrying ``(id, text, logprob)`` triples; empty
    # list when no top-K alternatives were requested (``return_top_k == 0``
    # and OpenAI ``logprobs`` was set to 0). Inner-tuple shape replaces the
    # legacy ``list[tuple[int, float]]`` pair shape; renderers that consume
    # this field read ``alt.text`` directly rather than re-decoding.
    # A forced token outside the sampler's support has zero probability;
    # its non-finite logprob is represented by None on the JSON-safe surface.
    logprobs: list[tuple[int, float | None, list[TokenAlt]]] | None = None
    # Steering expression applied to this generation, stringified via
    # :func:`drowse.steering.grammar.format_expr` for round-trip
    # reproduction.  ``None`` when no steering was active.  Receipts /
    # ``drowse replay`` land on this single field.
    applied_steering: str | None = None
    # Per-generated-token, per-layer residual-stream captures. ``None``
    # when SamplingConfig.return_hidden was False (default).  When set,
    # keyed by absolute layer index; each value is a ``[T, D]`` CPU
    # tensor where ``T == len(tokens)`` and ``D == hidden_size``. Dtype
    # matches the model's working dtype (fp16/bf16/fp32). ``to_dict``
    # deliberately omits this field — tensors don't serialize cleanly to
    # the JSON path; persist explicitly with ``torch.save`` if needed.
    hidden_states: dict[int, torch.Tensor] | None = None
    # End-of-generation manifold-probe aggregates, populated by
    # ``Monitor.score_aggregate`` when at least one probe is attached
    # (the same :class:`ProbeReading` shape the live per-token stream
    # carries, pooled at the last-content token).  Empty dict otherwise.
    # Keyed by registered probe name; round-trips through ``to_dict`` as a
    # nested mapping.
    probe_readings: dict[str, "ProbeReading"] = field(default_factory=dict)
    # The ``scope="aggregate"`` measurement envelope for this generation —
    # the same versioned record the per-token frames carry, built once at
    # finalize (``readout/measurements.py``).  It keeps each family's NATIVE
    # reading shape (geometry's full ``ProbeReading``, the single-axis
    # families' ``ScalarReading`` with its unit and depth basis), which
    # ``probe_readings`` above cannot: that field is the cross-family
    # compatibility dict keyed by one reading type, and it is what the
    # OpenAI / Ollama ``x-drowse-probe-readings`` vendor extension
    # serializes.  ``None`` when nothing was measured.
    measurements: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "token_count": self.token_count,
            "tok_per_sec": self.tok_per_sec,
            "elapsed": self.elapsed,
            "steering_alphas": dict(self.steering_alphas),
            "prompt_tokens": self.prompt_tokens,
            "finish_reason": self.finish_reason,
            "applied_steering": self.applied_steering,
            "probe_readings": {
                k: v.to_dict() for k, v in self.probe_readings.items()
            },
            "measurements": self.measurements,
        }


class RunSet(list[GenerationResult]):
    """Ordered set of generation results plus experiment metadata.

    ``RunSet`` is intentionally list-like: batch/sweep callers can iterate,
    index, and take ``len(...)``. Single-run callers use :attr:`first`.
    """

    def __init__(
        self,
        results: Iterable[GenerationResult] = (),
        *,
        node_ids: Iterable[str | None] | None = None,
        grid: Iterable[dict[str, Any]] | None = None,
        kind: str = "generation",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(results)
        self.node_ids: list[str | None] = (
            list(node_ids) if node_ids is not None else [None] * len(self)
        )
        if len(self.node_ids) < len(self):
            self.node_ids.extend([None] * (len(self) - len(self.node_ids)))
        self.grid: list[dict[str, Any]] = (
            [dict(row) for row in grid] if grid is not None else [{} for _ in self]
        )
        if len(self.grid) < len(self):
            self.grid.extend({} for _ in range(len(self) - len(self.grid)))
        self.kind = kind
        self.metrics: dict[str, Any] = dict(metrics or {})

    @property
    def first(self) -> GenerationResult:
        if not self:
            raise IndexError("RunSet is empty")
        return self[0]

    @property
    def node_id(self) -> str | None:
        return self.node_ids[0] if self.node_ids else None

    def probe_summary(self) -> dict[str, ProbeReadings]:
        """Fold each probe's per-run aggregate coordinates into a summary.

        The multi-run counterpart of ``GenerationResult.probe_readings``:
        one :class:`ProbeReadings` per probe name, in run order, ready for
        ``drowse.notebook.plots.plot_trait_history``.  Probes are keyed by
        the name they were registered under; a probe attached partway
        through the set contributes only the runs that measured it.
        """
        series: dict[str, list[tuple[float, ...]]] = {}
        for result in self:
            for name, reading in result.probe_readings.items():
                if reading.coords:
                    series.setdefault(name, []).append(tuple(reading.coords))
        return {
            name: ProbeReadings.from_coords(samples)
            for name, samples in series.items()
        }

    def to_collector(self) -> "ResultCollector":
        collector = ResultCollector()
        for idx, result in enumerate(self):
            tags: dict[str, Any] = {"run_idx": idx}
            if idx < len(self.node_ids) and self.node_ids[idx] is not None:
                tags["node_id"] = self.node_ids[idx]
            if idx < len(self.grid):
                tags.update(self.grid[idx])
            collector.add(result, **tags)
        return collector

    def to_dataframe(self) -> Any:
        return self.to_collector().to_dataframe()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "node_ids": list(self.node_ids),
            "grid": [dict(row) for row in self.grid],
            "metrics": dict(self.metrics),
            "results": [result.to_dict() for result in self],
        }


@dataclass
class TokenEvent:
    """Single token yielded during streaming generation."""
    text: str
    token_id: int
    index: int
    thinking: bool = False
    logprob: float | None = None
    # Top-K alternatives the model considered at this position. ``None``
    # when no top-K was requested (``return_top_k == 0`` and OpenAI
    # ``logprobs`` was None or 0). Replaces the legacy
    # ``top_logprobs: list[tuple[int, float]]`` pair shape — entries now
    # carry decoded text so consumers don't re-tokenize.
    top_alts: list[TokenAlt] | None = None
    finish_reason: str | None = None
    # Per-probe subspace readings computed inline against the latest
    # captured hidden state — the full coordinate readout
    # (:class:`ProbeReading`: coords + fraction + nearest + residual,
    # plus per-layer traces) for every attached probe, flat or curved (a
    # 2-node concept axis is the ``R = 1`` case; ``coords`` is a 1-tuple).
    # Populated by ``generate_stream`` only when the session has active
    # probes; otherwise None.
    probe_readings: dict[str, "ProbeReading"] | None = None
    # Perplexity of the configured sampler distribution after temperature,
    # top-k, and top-p renormalization — ``exp`` of Shannon entropy in nats.
    # Bounded above by sampler support size; a confident prediction
    # approaches 1. Consumers take ``log`` to recover entropy-nats.
    perplexity: float | None = None
    # The 5.x measurement envelope for this step (scope ``"token"``) — the one
    # wire record carrying every read-side channel: geometry / lens / SAE
    # ``instruments`` plus the flat ``scores`` / ``per_layer_scores`` views
    # (:func:`drowse.readout.measurements.build_measurements`).  It replaces the
    # former ``lens_readout`` / ``lens_aggregate`` / ``sae_readout`` fields and
    # the loom-owned ``captured`` record.  ``None`` when nothing was measured
    # this step (no probes, live readouts off).  ``probe_readings`` above stays
    # a separate typed field: it feeds the OpenAI/Ollama
    # ``x-drowse-probe-readings`` vendor extension
    # (compat protocols), carrying the merged per-probe readings from all
    # families as :class:`ProbeReading` objects rather than the serialized
    # envelope shape.
    measurements: dict[str, Any] | None = None
    # The ``generated_ids`` index of the token that triggered this emission
    # (the last buffered id for a multi-byte flush): the join key a logit fork
    # slices ``raw_token_ids`` with.  ``None`` before any token was decoded.
    raw_index: int | None = None

class ResultCollector:
    """Accumulates GenerationResults with tags for batch export."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    @property
    def results(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def add(self, result: GenerationResult, **tags: Any) -> None:
        row: dict[str, Any] = {
            "text": result.text,
            "token_count": result.token_count,
            "tok_per_sec": result.tok_per_sec,
            "elapsed": result.elapsed,
        }
        for probe_name, reading in result.probe_readings.items():
            for axis, value in enumerate(reading.coords):
                suffix = "" if len(reading.coords) == 1 else f"_{axis}"
                row[f"probe_{probe_name}_coord{suffix}"] = value
            row[f"probe_{probe_name}_fraction"] = reading.fraction
            row[f"probe_{probe_name}_residual"] = reading.residual
            row[f"probe_{probe_name}_membership"] = reading.membership
        for term_name, alpha in result.steering_alphas.items():
            row[f"steering_{term_name}_alpha"] = alpha
        row.update(tags)
        self._rows.append(row)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in self._rows)

    def to_csv(self, path: str) -> None:
        if not self._rows:
            return
        all_keys = list(dict.fromkeys(k for row in self._rows for k in row))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self._rows)

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            ) from e
        return pd.DataFrame(self._rows)
