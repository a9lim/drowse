"""Token-span helpers: which ids are structural, where content ends.

The single definition of the "last content token" every pooling site reads
from — extraction capture, the monitor's aggregate, the incremental scorer and
the batch rows — so the discipline can't drift per site.  Tokenizer-only; no
model.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def special_token_ids(tokenizer: Any) -> set[int]:
    """Token ids that are structural rather than content.

    The canonical "non-content" set every last-content-token pooling site
    shares: the tokenizer's ``all_special_ids`` *plus* everything in
    ``added_tokens_encoder``.  The second arm matters for tokenizers that
    don't promote chat-boundary markers to ``all_special_ids`` (talkie's
    ``<|user|>``/``<|end|>``/``<|assistant|>`` are added tokens but not
    "special") — without it, pooling lands on a structural turn marker
    whose hidden state is disconnected from the content.
    """
    skip = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    skip.update(int(v) for v in added.values())
    return skip


def last_content_index(
    token_ids: Sequence[int], tokenizer: Any, *, skip: set[int] | None = None,
) -> int:
    """Index of the final token belonging to the assistant content span.

    Chat templates may append whitespace *after* their closing special token
    (Smol ChatML renders ``...content\n<|im_end|>\n``). Trim terminal
    whitespace only when it follows a closing id, then walk backward past ids from
    :func:`special_token_ids` (``all_special_ids`` + ``added_tokens_encoder``).
    Whitespace before the closing marker remains part of the assistant span;
    its state is the one that predicts the marker, matching browser capture.
    The result floors at 0 so an all-structural sequence still yields a valid
    index.  This is the single definition of "last content token" shared by
    extraction (:func:`drowse.fit.capture._prepare_capture_batch`), the
    aggregate vector probe (:meth:`Monitor.score_per_token`), the
    incremental scoring path (``ReadPlane.finalize``), and the
    manifold aggregate (:meth:`Monitor.score_aggregate`) — every
    reported single-state value pools from here so the discipline can't
    drift per-site.
    """
    idx = len(token_ids) - 1
    if idx < 0:
        return 0
    skip = special_token_ids(tokenizer) if skip is None else skip

    # A template-owned newline can follow the closing marker. It is not an
    # added/special token, so the old special-only walk incorrectly pooled it.
    # Decode only the short suffix we inspect; scanning the whole vocabulary to
    # precompute whitespace ids would be substantially more work and is not
    # needed on this cold capture path.
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        suffix_start = idx
        while suffix_start > 0:
            piece = decode([int(token_ids[suffix_start])], skip_special_tokens=False)
            if not isinstance(piece, str) or not piece or not piece.isspace():
                break
            suffix_start -= 1
        if int(token_ids[suffix_start]) in skip:
            idx = suffix_start
    if skip:
        while idx > 0 and int(token_ids[idx]) in skip:
            idx -= 1
    return idx
