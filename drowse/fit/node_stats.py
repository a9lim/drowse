"""Per-node activation statistics for a manifold fit — the capture driver.

:func:`compute_manifold_node_stats` runs the model forwards that pool each
node's corpus into per-layer centroids (and, for curved fits, retains the
per-response rows in an :class:`~drowse.store.manifold_tensors.
ActivationRowStore` spool); :func:`compute_store_reduced_covariances` streams
that spool into each node's reduced within-node covariance for the sigma
field.  The geometry these feed is pure tensor math in
:mod:`drowse.geometry`; this module is where the model and the store meet it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

from drowse.errors import is_out_of_memory_error, ManifoldSpecError
from drowse.fit import capture
from drowse.store.manifold_tensors import ActivationRowStore

if TYPE_CHECKING:
    from drowse.geometry.subspace import LayerSubspace


def compute_manifold_node_stats(
    model: torch.nn.Module,
    tokenizer: object,
    layers: torch.nn.ModuleList,
    device: torch.device,
    node_groups: "Sequence[tuple[str, list[str]]]",
    prompts: "Sequence[str | list[dict[str, str]]]",
    *,
    roles: "Sequence[str | None]",
    model_type: str | None = None,
    layer_indices: Sequence[int] | None = None,
    retain_rows: bool = False,
    prepared_rows: "Sequence[tuple[torch.Tensor, int]] | None" = None,
    capture_context: "Any | None" = None,
) -> tuple[dict[int, torch.Tensor], ActivationRowStore | None]:
    """Fit-wide batched capture for all manifold nodes.

    Rows carry their node/within-node indices through one stream, so short
    template corpora share batches across node boundaries instead of paying one
    underfilled model forward per node.  Standard 48-response nodes retain the
    same chunking, while OOM backoff halves the active batch and cautiously grows
    it again. Returns layer-major ``(K, D)`` fp32 centroid stacks plus optional
    retained rows in the capture source dtype.  Keeping the result layer-major
    matches every fit consumer and avoids rebuilding each stack from ``K`` row
    views immediately after capture.
    """
    if not prompts:
        raise ValueError("conversational capture needs at least one baseline prompt")
    if len(node_groups) != len(roles):
        raise ValueError("node_groups and roles must be aligned")
    capture_layers = (
        list(range(len(layers)))
        if layer_indices is None else [int(idx) for idx in layer_indices]
    )
    k = len(prompts)
    flat: list[
        tuple[int, int, str | list[dict[str, str]], str, str | None]
    ] = []
    node_sizes: list[int] = []
    for node_idx, ((_label, responses), role) in enumerate(
        zip(node_groups, roles, strict=True),
    ):
        if not responses or len(responses) % k != 0:
            raise ManifoldSpecError(
                f"node corpus ({len(responses)} responses) must be a non-empty "
                f"multiple of the baseline prompt set ({k})"
            )
        node_sizes.append(len(responses))
        flat.extend(
            (node_idx, row_idx, prompts[row_idx % k], response, role)
            for row_idx, response in enumerate(responses)
        )

    sums: dict[int, torch.Tensor] = {}
    retained = ActivationRowStore(node_sizes) if retain_rows else None
    # Render/tokenize once, then group similar lengths so right-padding does not
    # make every row pay the longest response's quadratic attention cost.  Tiny
    # CPU test tokenizers are intentionally non-callable and keep the legacy
    # seam; production HF tokenizers always take the prepared path.
    prepared: list[tuple[torch.Tensor, int]] | None = (
        list(prepared_rows) if prepared_rows is not None else None
    )
    if prepared is not None and len(prepared) != len(flat):
        raise ValueError("prepared_rows must align with the flattened manifold corpus")
    if prepared is None and callable(tokenizer):
        prepared = capture._prepare_capture_batch(
            tokenizer,
            [row[2] for row in flat],
            [row[3] for row in flat],
            roles=[row[4] for row in flat],
            model_type=model_type,
        )
    order = (
        sorted(range(len(flat)), key=lambda i: int(prepared[i][0].shape[1]))
        if prepared is not None else list(range(len(flat)))
    )
    # Optimistically try the proven maximum and halve on OOM. Starting at the
    # old conservative 16 made a 107-node fit spend four successful forwards
    # merely climbing to the width that the same device had already shown it
    # could run.
    active_batch = capture._CAPTURE_BATCH_MAX
    start = 0
    is_mps = getattr(device, "type", None) == "mps"
    while start < len(order):
        end = min(start + active_batch, len(order))
        chunk_indices = order[start:end]
        chunk = [flat[i] for i in chunk_indices]
        try:
            per_layer = capture._encode_and_capture_all_batch(
                model, tokenizer,
                [row[2] for row in chunk],
                [row[3] for row in chunk],
                layers, device,
                roles=[row[4] for row in chunk],
                model_type=model_type,
                layer_indices=capture_layers,
                rendered=(
                    [prepared[i] for i in chunk_indices]
                    if prepared is not None else None
                ),
                promote_pooled=not retain_rows,
                capture_context=capture_context,
            )
        except RuntimeError as exc:
            if not is_out_of_memory_error(exc) or active_batch <= 1:
                raise
            active_batch = max(1, active_batch // 2)
            if is_mps:
                torch.mps.empty_cache()
            elif getattr(device, "type", None) == "cuda":
                torch.cuda.empty_cache()
            continue

        node_ids_cpu = torch.tensor([row[0] for row in chunk], dtype=torch.long)
        flat_indices_cpu = torch.tensor(chunk_indices, dtype=torch.long)
        unique = inverse = None
        if retained is None:
            node_ids = node_ids_cpu.to(device)
            unique, inverse = torch.unique(
                node_ids, sorted=True, return_inverse=True,
            )
        for idx in capture_layers:
            captured = per_layer[idx].detach()
            if idx not in sums:
                sums[idx] = torch.zeros(
                    len(node_groups), captured.shape[1], dtype=torch.float32,
                )
            if retained is not None:
                # Raw rows must cross the boundary for the later sigma fit, but
                # stay in source dtype in the mmap.  Accumulate their promoted
                # host values while they are already present.
                host = captured.to(device="cpu")
                retained.write(idx, flat_indices_cpu, host)
                sums[idx].index_add_(
                    0, node_ids_cpu, host.to(torch.float32),
                )
            else:
                # Centroid-only fits need one partial sum per node, not every
                # response row.  Reduce on-device in fp32 and transfer U×D
                # (usually 1×D) instead of B×D.
                assert unique is not None and inverse is not None
                partial = torch.zeros(
                    unique.numel(), captured.shape[1],
                    dtype=torch.float32, device=captured.device,
                )
                partial.index_add_(0, inverse, captured.to(torch.float32))
                sums[idx].index_add_(
                    0, unique.to(device="cpu"), partial.to(device="cpu"),
                )
        del per_layer
        start = end
        if is_mps:
            torch.mps.empty_cache()

    divisors = torch.tensor(node_sizes, dtype=torch.float32).reshape(-1, 1)
    # The accumulators already have the final layer-major K×D shape and sole
    # ownership here. Normalize them in place instead of allocating a second
    # complete fp32 centroid roster at the capture/fitting boundary.
    for idx in capture_layers:
        sums[idx].div_(divisors)
    return sums, retained


def compute_store_reduced_covariances(
    store: ActivationRowStore,
    layer_subs: "dict[int, LayerSubspace]",
    *,
    row_chunk: int = 2048,
) -> list[dict[int, torch.Tensor]]:
    """Project a layer-major activation spool into per-node covariances.

    Streams each layer of the layer-major spool once in bounded row chunks,
    performs large projection GEMMs, and segments only the small ``(N,R)``
    results by the store's contiguous node boundaries (``N − 1`` denominator;
    zeros for a single-sample node).  Covariance is translation-invariant, so
    projecting raw rows is exactly equivalent to subtracting ``sub.mean`` first
    and then centering the reduced rows.
    """
    if row_chunk <= 0:
        raise ValueError("row_chunk must be > 0")
    if not set(layer_subs) <= set(store.layer_indices):
        raise ValueError(
            "activation-row store must cover every fitted subspace layer"
        )

    n_nodes = len(store.node_sizes)
    out: list[dict[int, torch.Tensor]] = [dict() for _ in range(n_nodes)]
    boundaries = [
        (offset, offset + size)
        for offset, size in zip(store.offsets, store.node_sizes, strict=True)
    ]
    for idx, sub in layer_subs.items():
        basis_t = sub.basis.to(device="cpu", dtype=torch.float32).transpose(0, 1)
        rank = sub.rank
        rows = store.flat_rows(idx)
        reduced_rows = torch.empty(store.total_rows, rank, dtype=torch.float32)
        mean = sub.mean.to(device="cpu", dtype=torch.float32)
        centered_buffer = torch.empty(
            min(row_chunk, store.total_rows), rows.shape[1], dtype=torch.float32,
        )
        for start in range(0, store.total_rows, row_chunk):
            end = min(start + row_chunk, store.total_rows)
            centered = centered_buffer[:end - start]
            # Center before projection for numerical stability at the large
            # common-mode offsets residual streams can carry. ``out=`` keeps
            # the operation bounded and out-of-place even when the source row
            # store is already fp32 (where ``.to(float32)`` would alias it).
            torch.sub(rows[start:end], mean, out=centered)
            reduced_rows[start:end] = centered @ basis_t

        for k, (start, end) in enumerate(boundaries):
            size = end - start
            if size <= 1:
                cov = torch.zeros(rank, rank, dtype=torch.float32)
            else:
                node_reduced = reduced_rows[start:end]
                centered = node_reduced - node_reduced.mean(dim=0, keepdim=True)
                cov = centered.transpose(0, 1) @ centered / float(size - 1)
            out[k][idx] = cov
    return out


__all__ = ["compute_manifold_node_stats", "compute_store_reduced_covariances"]
