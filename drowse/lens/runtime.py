"""The lens runtime: the resident lens, its identity, and the readout math.

:class:`LensRuntime` holds the one resident lens a session reads through and
the identity that binds it to its disk artifact.  It also owns the readout
primitives every lens surface shares: device transport stacks, the
``(layer, hidden)`` -> logits front half, per-row top-k, the layer aggregate,
the offline prompt readout, J-space decomposition, and token directions.

The runtime is plain state plus math.  Residency *transactions* (refresh from
disk, adopt, evict) belong to :class:`~drowse.lens.instrument.LensInstrument`,
which serializes them under its state lock and rebuilds the probe and live
state that depends on the resident lens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from drowse.base.lru import BoundedLRU
from drowse.errors import RequestError
from drowse.lens.jlens import (
    JacobianLens,
    aggregate_readout,
    aggregate_readout_from_probabilities,
    resolve_token_atom,
    topk_probabilities,
    transport_rows,
)
from drowse.model.layers import LayerSelector, resolve_layer_selector
from drowse.lens.store import lens_identity, lens_layers
from drowse.model.fingerprint import loaded_model_fingerprint

#: Device transport stacks kept per runtime (each is ``[L, d, d]`` fp32).
DEVICE_STACK_CACHE_MAX = 2
#: Decoded single-token strings kept per runtime.
DECODE_CACHE_MAX = 4096
#: Depth lists and selector tensors kept per runtime.
SELECTOR_CACHE_MAX = 64
#: Upper bound on one readout chunk's ``[rows, vocab]`` fp32 logits.
READOUT_CHUNK_BYTES = 256 * 1024**2

#: ``capture_hidden(input_ids, layer_indices=..., pool_index=...)`` -> per-layer
#: hidden states: the session's single capture forward.
CaptureHidden = Callable[..., dict[int, torch.Tensor]]


def matches_loaded_model(
    sidecar: Mapping[str, Any] | None, model: Any, model_id: str,
) -> bool:
    """Whether a lens sidecar was fitted on (or published for) these weights.

    A local fit records the exact loaded-weight fingerprint.  An external
    binding records the model id, the base-model commit, and its layer
    roster, which must agree with the live config.
    """
    if sidecar is None:
        return False
    fitted_fingerprint = sidecar.get("model_fingerprint")
    if isinstance(fitted_fingerprint, str) and fitted_fingerprint:
        return fitted_fingerprint == loaded_model_fingerprint(model, model_id)
    source = sidecar.get("_source")
    if not isinstance(source, Mapping) or source.get("kind") != "huggingface":
        return False
    if str(source.get("model_id", "")).casefold() != model_id.casefold():
        return False
    base = getattr(model, "_orig_mod", model)
    config = getattr(base, "config", getattr(model, "config", None))
    live_commit = (
        getattr(config, "_commit_hash", None)
        or getattr(getattr(config, "text_config", None), "_commit_hash", None)
    )
    if not isinstance(live_commit, str) or live_commit != source.get("model_revision"):
        return False
    text_config = getattr(config, "text_config", config)
    hidden = getattr(text_config, "hidden_size", getattr(text_config, "n_embd", None))
    n_layers = getattr(
        text_config, "num_hidden_layers", getattr(text_config, "n_layer", None),
    )
    layers = lens_layers(sidecar)
    return bool(
        isinstance(hidden, int)
        and not isinstance(hidden, bool)
        and int(sidecar.get("d_model", -1)) == hidden
        and isinstance(n_layers, int)
        and not isinstance(n_layers, bool)
        and layers
        and all(0 <= layer < n_layers for layer in layers)
    )


def select_rows(tensor: torch.Tensor, rows: Sequence[int]) -> torch.Tensor:
    """Select rows without copying when the request is identity/contiguous."""
    row_list = [int(row) for row in rows]
    if not row_list:
        return tensor[:0]
    if row_list == list(range(int(tensor.shape[0]))):
        return tensor
    start = row_list[0]
    if row_list == list(range(start, start + len(row_list))):
        return tensor[start:start + len(row_list)]
    return tensor.index_select(
        0, torch.tensor(row_list, device=tensor.device, dtype=torch.long),
    )


def _missing_layers_error(lens: JacobianLens, missing: Sequence[int]) -> RequestError:
    return RequestError(
        f"layers {list(missing)} not in the fitted lens "
        f"(fitted: {lens.source_layers[0]}..{lens.source_layers[-1]})"
    )


class LensRuntime:
    """The resident lens plus the readout primitives over it.

    ``lens`` / ``identity`` are the resident lens and the digest of the
    artifact pointer it was loaded from (``None`` when nothing is resident).
    Only :meth:`install` and :meth:`clear` change them; the caller holds the
    lens-state lock around both.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_id: str,
        device: torch.device,
        n_layers: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.n_layers = int(n_layers)
        self.lens: JacobianLens | None = None
        self.identity: str | None = None
        # Only the transport stacks depend on the resident lens; every other
        # cache is keyed by value and survives a lens swap.
        self._stacks: BoundedLRU[tuple[int, str, tuple[int, ...]], torch.Tensor] = (
            BoundedLRU(DEVICE_STACK_CACHE_MAX)
        )
        self._decoded: BoundedLRU[int, str] = BoundedLRU(DECODE_CACHE_MAX)
        self._depths: BoundedLRU[tuple[int, ...], list[float]] = BoundedLRU(
            SELECTOR_CACHE_MAX,
        )
        self._depth_tensors: BoundedLRU[tuple[str, tuple[int, ...]], torch.Tensor] = (
            BoundedLRU(SELECTOR_CACHE_MAX)
        )
        self._long_tensors: BoundedLRU[tuple[str, tuple[int, ...]], torch.Tensor] = (
            BoundedLRU(SELECTOR_CACHE_MAX)
        )
        self._readout_modules: tuple[torch.Tensor, torch.nn.Module] | None = None

    # ------------------------------------------------------------ residency

    def matches(self, sidecar: Mapping[str, Any] | None) -> bool:
        """Whether ``sidecar`` describes a lens for the loaded weights."""
        return matches_loaded_model(sidecar, self.model, self.model_id)

    def install(
        self, lens: JacobianLens, sidecar: Mapping[str, Any] | None,
    ) -> None:
        """Make ``lens`` resident under ``sidecar``'s identity."""
        self.lens = lens
        self.identity = lens_identity(sidecar)
        self._stacks.clear()

    def clear(self) -> None:
        """Drop the resident lens and its device stacks."""
        self.lens = None
        self.identity = None
        self._stacks.clear()

    def close(self) -> None:
        """Release every cache (session teardown)."""
        self.clear()
        self._decoded.clear()
        self._depths.clear()
        self._depth_tensors.clear()
        self._long_tensors.clear()
        self._readout_modules = None

    # ------------------------------------------------------------ layers

    def resolve_layers(
        self, lens: JacobianLens, layers: LayerSelector,
    ) -> list[int]:
        """Resolve a readout selector over ``lens``'s fitted layers.

        Raises :class:`~drowse.errors.RequestError` when an explicit layer is not fitted.
        """
        req = resolve_layer_selector(
            layers, lens.source_layers, n_layers=self.n_layers,
        )
        missing = [layer for layer in req if layer not in lens.jacobians]
        if missing:
            raise _missing_layers_error(lens, missing)
        return req

    # ------------------------------------------------------------ caches

    def transport_stack(
        self,
        lens: JacobianLens,
        layers: Sequence[int],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """``[L, d, d]`` fp32 device stack of ``lens``'s matrices for ``layers``."""
        target = self.device if device is None else device
        layer_tuple = tuple(int(layer) for layer in layers)
        return self._stacks.get_or_build(
            (id(lens), str(target), layer_tuple),
            lambda: torch.stack([
                lens.jacobians[layer].to(device=target, dtype=torch.float32)
                for layer in layer_tuple
            ]),
        )

    def readout_modules(self) -> tuple[torch.Tensor, torch.nn.Module]:
        """The unembedding and the model-specific final readout path."""
        if self._readout_modules is None:
            from drowse.model.readout import LogitReadout, get_unembedding

            self._readout_modules = (
                get_unembedding(self.model), LogitReadout(self.model),
            )
        return self._readout_modules

    def decode_id(self, token_id: int) -> str:
        """Cached single-token decode shared by every lens readout."""
        return self._decoded.get_or_build(
            int(token_id), lambda: str(self.tokenizer.decode([int(token_id)])),
        )

    def depths(self, layers: Sequence[int]) -> list[float]:
        """Normalized layer depths ``layer / (n_layers - 1)`` (0 = first block)."""
        layer_tuple = tuple(int(layer) for layer in layers)
        denom = max(self.n_layers - 1, 1)
        return self._depths.get_or_build(
            layer_tuple, lambda: [layer / denom for layer in layer_tuple],
        )

    def depth_tensor(
        self, layers: Sequence[int], device: torch.device,
    ) -> torch.Tensor:
        """``[L, 1]`` fp32 device depth column, cached across decode steps."""
        layer_tuple = tuple(int(layer) for layer in layers)
        return self._depth_tensors.get_or_build(
            (str(device), layer_tuple),
            lambda: torch.tensor(
                self.depths(layer_tuple), dtype=torch.float32, device=device,
            ).reshape(len(layer_tuple), 1),
        )

    def long_tensor(
        self, values: Sequence[int], device: torch.device,
    ) -> torch.Tensor:
        """Device long selector tensor cached by value tuple and device."""
        value_tuple = tuple(int(value) for value in values)
        return self._long_tensors.get_or_build(
            (str(device), value_tuple),
            lambda: torch.tensor(value_tuple, dtype=torch.long, device=device),
        )

    # ------------------------------------------------------------ rows

    def logits_rows(
        self,
        lens: JacobianLens,
        rows: Sequence[tuple[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Full-vocab lens logits ``[n_rows, vocab]`` for ``(layer, hidden)`` rows.

        The shared front half of the per-layer top-k and the layer aggregate:
        one bmm plus one readout serves both.
        """
        unembed, readout = self.readout_modules()
        device = unembed.device
        unique_layers = sorted({layer for layer, _ in rows})
        stack = self.transport_stack(lens, unique_layers, device)
        layer_to_row = {layer: idx for idx, layer in enumerate(unique_layers)}
        jacobian_rows = select_rows(stack, [layer_to_row[layer] for layer, _ in rows])
        hidden = torch.stack([row.detach() for _, row in rows])
        return readout(transport_rows(jacobian_rows, hidden))

    def aggregate_rows(
        self,
        logits: torch.Tensor | None,
        layers: Sequence[int],
        *,
        top_k: int,
        probabilities: torch.Tensor | None = None,
    ) -> list[tuple[str, float, float, float]]:
        """Layer-aggregate per-layer lens rows into decoded chips
        ``[(token, strength, com, spread), ...]`` (see
        :func:`drowse.lens.jlens.aggregate_readout`)."""
        if probabilities is None:
            if logits is None:
                raise ValueError("logits are required when probabilities are absent")
            rows = aggregate_readout(
                logits.float(),
                self.depths(layers),
                top_k=top_k,
                depth_tensor=self.depth_tensor(layers, logits.device),
            )
        else:
            rows = aggregate_readout_from_probabilities(
                probabilities,
                self.depths(layers),
                top_k=top_k,
                depth_tensor=self.depth_tensor(layers, probabilities.device),
            )
        return [
            (self.decode_id(token_id), strength, com, spread)
            for token_id, strength, com, spread in rows
        ]

    def topk_rows(
        self,
        lens: JacobianLens,
        rows: Sequence[tuple[int, torch.Tensor]],
        *,
        top_k: int,
        logits: torch.Tensor | None = None,
    ) -> list[list[tuple[str, float, int]]]:
        """Per-row top-k as ``[(token, p_l, vocab_id), ...]``.

        The value is the per-layer readout probability, the one unit every
        lens surface reports.
        """
        if not rows:
            return []
        if logits is None:
            logits = self.logits_rows(lens, rows)
        vals, idxs = topk_probabilities(logits, top_k)
        out: list[list[tuple[str, float, int]]] = []
        for vrow, irow in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            out.append([
                (self.decode_id(int(idx)), float(value), int(idx))
                for value, idx in zip(vrow, irow)
            ])
        return out

    # ------------------------------------------------------------ readouts

    def readout(
        self,
        lens: JacobianLens,
        prompt: str,
        *,
        capture_hidden: CaptureHidden,
        layers: LayerSelector = None,
        positions: Sequence[int] | None = None,
        top_k: int = 10,
        aggregate: bool = False,
    ) -> (
        dict[int, list[list[tuple[str, float]]]]
        | tuple[
            dict[int, list[list[tuple[str, float]]]],
            list[list[tuple[str, float, float, float]]],
        ]
    ):
        """Top-``top_k`` readout per (layer, position) of a raw prompt.

        One capture forward, then the vocabulary readout in position chunks
        sized so one chunk's ``[layers x positions, vocab]`` logits stay under
        :data:`READOUT_CHUNK_BYTES` (a full-position sweep over a large
        vocabulary would otherwise materialize every row at once).  The
        caller holds the model-exclusive guard.
        """
        req = self.resolve_layers(lens, layers)
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        n_pos = int(ids.shape[1])
        if positions is None:
            pos = [n_pos - 1]
            hidden = capture_hidden(ids, layer_indices=req, pool_index=n_pos - 1)
            per_position = {layer: hidden[layer].reshape(1, -1) for layer in req}
        else:
            pos = [p if p >= 0 else n_pos + p for p in positions]
            hidden = capture_hidden(ids, layer_indices=req, pool_index=None)
            per_position = {layer: hidden[layer][0, pos, :] for layer in req}
        out: dict[int, list[list[tuple[str, float]]]] = {
            layer: [[] for _ in pos] for layer in req
        }
        agg: list[list[tuple[str, float, float, float]]] = []
        vocab = int(self.readout_modules()[0].shape[0])
        row_bytes = max(len(req) * vocab * 4, 1)
        chunk = max(1, READOUT_CHUNK_BYTES // row_bytes)
        for start in range(0, len(pos), chunk):
            stop = min(start + chunk, len(pos))
            # Layer-major rows, so the logits view as ``[L, C, vocab]``.
            rows = [
                (layer, per_position[layer][index])
                for layer in req
                for index in range(start, stop)
            ]
            logits = self.logits_rows(lens, rows)
            decoded = self.topk_rows(lens, rows, top_k=top_k, logits=logits)
            width = stop - start
            for row_index, row in enumerate(decoded):
                layer = req[row_index // width]
                out[layer][start + row_index % width] = [
                    (tok, p_l) for tok, p_l, _ in row
                ]
            if aggregate:
                grid = logits.reshape(len(req), width, -1)
                for offset in range(width):
                    agg.append(self.aggregate_rows(grid[:, offset], req, top_k=top_k))
        return (out, agg) if aggregate else out

    def decompose(
        self,
        lens: JacobianLens,
        directions: Mapping[int, torch.Tensor],
        *,
        label: str,
        k: int = 16,
        layers: LayerSelector = None,
    ) -> dict[int, tuple[float, list[tuple[str, float]]]]:
        """Split a per-layer direction into its J-space component.

        Returns ``{layer: (share, [(token, coeff), ...])}``: ``share`` is the
        fraction of the direction's variance carried by its best sparse
        nonnegative combination of ``k`` lens token vectors.
        """
        from drowse.lens.decompose import TokensOut, sparse_decompose
        from drowse.model.readout import get_output_projection

        unembed = self.readout_modules()[0]
        projection = get_output_projection(self.model)
        req = [
            layer
            for layer in resolve_layer_selector(
                layers, lens.source_layers, n_layers=self.n_layers,
            )
            if layer in directions and layer in lens.jacobians
        ]
        if not req:
            raise RequestError(
                f"no overlap between {label!r}'s layers and the fitted lens layers"
            )
        out: dict[int, tuple[float, list[tuple[str, float]]]] = {}
        for layer in req:
            jacobian = lens.jacobians[layer]
            if projection is not None:
                jacobian = (
                    projection.detach().to(device=jacobian.device, dtype=torch.float32)
                    @ jacobian
                )
            dec = sparse_decompose(
                directions[layer],
                TokensOut(
                    jacobian, unembed,
                    atom_norms=(
                        lens.atom_norms(layer, unembed) if projection is None else None
                    ),
                ),
                layer=layer, k=k,
            )
            out[layer] = (
                dec.share,
                [(str(self.tokenizer.decode([tok])), coeff) for tok, coeff in dec.atoms],
            )
        return out

    def token_directions(
        self, lens: JacobianLens, word: str,
    ) -> dict[int, torch.Tensor]:
        """``W_U[v] @ J_l`` for a single-token word (or ``#<id>``) on every
        fitted layer, landed on the session device."""
        from drowse.model.readout import get_output_projection

        unembed = self.readout_modules()[0]
        token_id = resolve_token_atom(
            self.tokenizer, word, vocab_size=int(unembed.shape[0]),
        )
        fitted = sorted(int(layer) for layer in lens.source_layers)
        directions = lens.token_direction(
            token_id, unembed, layers=fitted,
            output_projection=get_output_projection(self.model),
        )
        return {layer: d.to(self.device) for layer, d in directions.items()}


__all__ = [
    "CaptureHidden",
    "DECODE_CACHE_MAX",
    "DEVICE_STACK_CACHE_MAX",
    "LensRuntime",
    "READOUT_CHUNK_BYTES",
    "SELECTOR_CACHE_MAX",
    "matches_loaded_model",
    "select_rows",
]
