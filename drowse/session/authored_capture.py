"""Measure authored prompt text the first time a generation re-reads it.

Authored loom turns (typed, committed or edited text: ``recipe is None``) have
no per-token measurement rows of their own.  When a later generation's prompt
renders them, the prefill forward already computes their hidden states, so the
engine scores those positions once and attaches the rows to the authored node
(``LoomTree.set_authored_token_scores``).  This module owns that work:

- :func:`plan_authored_capture` decides, before the prefill, whether anything
  will measure the rows (the geometry monitor, a live J-lens or SAE consumer)
  and only then tokenizes the uncaptured authored channels on the path;
- :meth:`AuthoredPromptCapture.match` locates them in the rendered prompt and
  maps each token to the capture row that produced it;
- :meth:`AuthoredPromptCapture.persist` scores those rows through the monitor
  and the lens/SAE instruments and writes the token rows to the tree.

Row and measurement-envelope shapes are the loom token-row contract the
browser runtime replays; they must not change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch

from drowse.generation.token_payloads import TokenProbePayload

if TYPE_CHECKING:
    from drowse.lens.instrument import LensInstrument
    from drowse.loom.tree import LoomTree
    from drowse.readout.monitor import Monitor
    from drowse.sae.instrument import SaeInstrument

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class AuthoredPromptTarget:
    """One visible authored node channel awaiting its first prompt capture."""

    node_id: str
    thinking: bool
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]


@dataclass(slots=True)
class AuthoredPromptMatch:
    """A target located in the full rendered prompt token sequence."""

    target: AuthoredPromptTarget
    token_positions: tuple[int, ...]
    capture_columns: tuple[int | None, ...] = ()


def _as_flat_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu").tolist()
    if value and isinstance(value[0], list) and len(value) == 1:
        value = value[0]
    return list(value)


def tokenize_authored_text(
    tokenizer: Any,
    node_id: str,
    text: str,
    *,
    thinking: bool,
) -> AuthoredPromptTarget | None:
    """Tokenize visible authored text without changing its rendered bytes.

    One fast-tokenizer call yields both the ids and their offsets, which let
    the loom keep the user's exact whitespace while assigning each span to the
    model token that consumed it.  A tokenizer without offsets falls back to
    one-token decodes, used only when they concatenate byte-for-byte to the
    original text; otherwise the channel is left plain instead of replacing
    authored copy with a normalized reconstruction.
    """
    token_ids: tuple[int, ...] | None = None
    pieces: tuple[str, ...] | None = None
    try:
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
        )
        token_ids = tuple(int(tid) for tid in _as_flat_list(encoded["input_ids"]))
        offsets = _as_flat_list(encoded["offset_mapping"])
        cursor = 0
        spans: list[str] = []
        valid = len(offsets) == len(token_ids)
        for raw_start, raw_end in offsets:
            start, end = int(raw_start), int(raw_end)
            if start < 0 or end < start or end > len(text) or end < cursor:
                valid = False
                break
            # Attach gaps (usually leading spaces omitted by a word token's
            # formal offset) to the following visible token.
            spans.append(text[cursor:end])
            cursor = end
        if valid and spans:
            spans[-1] += text[cursor:]
            if "".join(spans) == text:
                pieces = tuple(spans)
    except (KeyError, TypeError, ValueError, NotImplementedError):
        token_ids = None
    if token_ids is None:
        token_ids = tuple(
            int(tid) for tid in tokenizer.encode(text, add_special_tokens=False)
        )
    if not token_ids:
        return None

    if pieces is None:
        decoded: list[str] = []
        try:
            for tid in token_ids:
                try:
                    piece = tokenizer.decode(
                        [tid],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    piece = tokenizer.decode([tid], skip_special_tokens=False)
                decoded.append(str(piece))
        except (AttributeError, TypeError, ValueError):
            return None
        if "".join(decoded) != text:
            return None
        pieces = tuple(decoded)

    return AuthoredPromptTarget(
        node_id=node_id,
        thinking=thinking,
        token_ids=token_ids,
        token_texts=pieces,
    )


def pending_targets(
    tree: "LoomTree",
    tokenizer: Any,
    assistant_node_id: str | None,
    *,
    raw: bool,
) -> list[AuthoredPromptTarget]:
    """Uncaptured authored channels on the prompt path, in render order."""
    if assistant_node_id is None:
        return []
    generated = tree.nodes.get(assistant_node_id)
    if generated is None or generated.parent_id is None:
        return []
    targets: list[AuthoredPromptTarget] = []
    for node in tree.path_to(generated.parent_id):
        if node.id == tree.root_id or node.recipe is not None:
            continue
        # Raw/flat rendering ignores thinking_text; chat rendering places
        # authored thinking before the visible response content.
        if not raw and node.thinking_text and node.thinking_tokens is None:
            target = tokenize_authored_text(
                tokenizer, node.id, node.thinking_text, thinking=True,
            )
            if target is not None:
                targets.append(target)
        if node.text and node.tokens is None:
            target = tokenize_authored_text(
                tokenizer, node.id, node.text, thinking=False,
            )
            if target is not None:
                targets.append(target)
    return targets


def rfind_token_subsequence(
    haystack: list[int],
    needle: tuple[int, ...],
    end: int,
) -> int:
    """Rightmost exact token subsequence ending at or before ``end``."""
    width = len(needle)
    for start in range(min(end, len(haystack)) - width, -1, -1):
        if tuple(haystack[start:start + width]) == needle:
            return start
    return -1


def match_targets(
    tokenizer: Any,
    targets: list[AuthoredPromptTarget],
    input_ids: torch.Tensor,
    *,
    cache_position_offset: int,
    search_end: int | None = None,
) -> tuple[list[AuthoredPromptMatch], list[int]]:
    """Locate visible text in the full prompt and map producer rows.

    Matching walks backward because the generation prompt follows the last
    visible message and duplicate authored lines are common during rerolls.
    Positions are first resolved in the full prompt, then translated into
    the suffix-local coordinates of a KV-prefix hit.  The first suffix token
    has no producer row in the new forward and is therefore left unscored.
    """
    full_ids = [int(tid) for tid in input_ids[0].detach().to("cpu").tolist()]
    limit = min(
        len(full_ids),
        int(search_end) if search_end is not None else len(full_ids),
    )
    reversed_matches: list[AuthoredPromptMatch] = []
    for target in reversed(targets):
        start = rfind_token_subsequence(full_ids, target.token_ids, limit)
        # Gemma-family templates trim message content before tokenizing.
        # Preserve the loom's verbatim authored whitespace, but if the exact
        # standalone token sequence is absent, retry the stripped content and
        # attach the trimmed edge bytes to its first/last visible token span.
        # Whitespace the template removed has no model position of its own,
        # so this is the only faithful display mapping.
        if start < 0:
            original_text = "".join(target.token_texts)
            stripped_text = original_text.strip()
            if stripped_text and stripped_text != original_text:
                stripped_target = tokenize_authored_text(
                    tokenizer, target.node_id, stripped_text,
                    thinking=target.thinking,
                )
                if stripped_target is not None:
                    pieces = list(stripped_target.token_texts)
                    left = original_text[:len(original_text) - len(
                        original_text.lstrip()
                    )]
                    right = original_text[len(original_text.rstrip()):]
                    pieces[0] = left + pieces[0]
                    pieces[-1] += right
                    stripped_target.token_texts = tuple(pieces)
                    stripped_start = rfind_token_subsequence(
                        full_ids, stripped_target.token_ids, limit,
                    )
                    if stripped_start >= 0:
                        target = stripped_target
                        start = stripped_start
        if start < 0:
            _log.debug(
                "authored prompt capture skipped node %s: content tokens "
                "not found verbatim in rendered prompt",
                target.node_id,
            )
            continue
        positions = tuple(range(start, start + len(target.token_ids)))
        reversed_matches.append(AuthoredPromptMatch(target, positions))
        limit = start
    matches = list(reversed(reversed_matches))

    effective_producers = sorted({
        token_position - 1 - cache_position_offset
        for match in matches
        for token_position in match.token_positions
        if token_position - 1 >= cache_position_offset
    })
    producer_to_column = {
        producer: column for column, producer in enumerate(effective_producers)
    }
    for match in matches:
        match.capture_columns = tuple(
            producer_to_column.get(token_position - 1 - cache_position_offset)
            if token_position - 1 >= cache_position_offset
            else None
            for token_position in match.token_positions
        )
    return matches, effective_producers


@dataclass(slots=True)
class AuthoredPromptCapture:
    """One generation's plan to measure the authored text its prompt re-reads.

    The engine calls :meth:`match` once the prompt is rendered (it returns the
    capture rows to retain during prefill) and :meth:`persist` after the
    prefill forward with those retained rows.
    """

    tree: "LoomTree"
    tokenizer: Any
    targets: list[AuthoredPromptTarget]
    monitor: "Monitor"
    lens: "LensInstrument"
    sae: "SaeInstrument"
    monitor_active: bool
    lens_active: bool
    sae_active: bool
    readout_top_k: int
    persist_per_layer_scores: bool
    steering: str | None
    search_end: int | None = None
    matches: list[AuthoredPromptMatch] = field(default_factory=list)

    def match(self, input_ids: torch.Tensor, *, cache_position_offset: int) -> list[int]:
        """Locate the targets in ``input_ids``; return the producer rows to keep."""
        self.matches, producers = match_targets(
            self.tokenizer,
            self.targets,
            input_ids,
            cache_position_offset=cache_position_offset,
            search_end=self.search_end,
        )
        return producers

    def persist(self, prompt_rows: dict[int, torch.Tensor]) -> None:
        """Score the retained prefill rows and attach them to authored nodes."""
        matches = self.matches
        if not matches or not prompt_rows:
            return
        lens_live = self.lens.live
        sae_live = self.sae.live
        lens_source = lens_live.get("source") if lens_live is not None else None
        sae_source = sae_live.get("source") if sae_live is not None else None
        sae_layer = int(sae_live["layer"]) if sae_live is not None else None

        output_rows: list[list[dict[str, Any]]] = [
            [
                {
                    "token_id": int(tid),
                    "text": text,
                    "logprob": None,
                    "perplexity": None,
                }
                for tid, text in zip(
                    match.target.token_ids, match.target.token_texts, strict=True,
                )
            ]
            for match in matches
        ]
        ordered: list[tuple[int, int, int, int]] = []
        for match_index, match in enumerate(matches):
            for token_index, (token_position, column) in enumerate(zip(
                match.token_positions, match.capture_columns, strict=True,
            )):
                if column is not None:
                    ordered.append((token_position, match_index, token_index, column))

        populated_matches: set[int] = set()
        for _position, match_index, token_index, column in sorted(ordered):
            hidden = {
                layer: rows[column]
                for layer, rows in prompt_rows.items()
                if column < int(rows.shape[0])
            }
            if not hidden:
                continue
            payload = TokenProbePayload()
            if self.monitor_active:
                payload.merge_readings(
                    self.monitor.score_single_token(hidden),
                    family="geometry",
                    per_layer=self.persist_per_layer_scores,
                )

            lens_payload = lens_aggregate = lens_token_ids = None
            if self.lens_active:
                lens_capture = self.lens.authored_capture(
                    hidden, top_k=self.readout_top_k,
                )
                if lens_capture is not None:
                    (
                        lens_payload,
                        lens_aggregate,
                        lens_token_ids,
                        lens_readings,
                    ) = lens_capture
                    payload.merge_readings(
                        lens_readings,
                        family="lens",
                        per_layer=self.persist_per_layer_scores,
                    )

            sae_payload = None
            if self.sae_active:
                sae_capture = self.sae.authored_capture(
                    hidden, top_k=self.readout_top_k,
                )
                if sae_capture is not None:
                    sae_payload, sae_readings = sae_capture
                    payload.merge_readings(
                        sae_readings,
                        family="sae",
                        per_layer=self.persist_per_layer_scores,
                    )

            shaped = payload.to_token_payload(
                lens=lens_payload,
                lens_aggregate=lens_aggregate,
                lens_token_ids=lens_token_ids,
                lens_source=lens_source,
                sae=sae_payload,
                sae_source=sae_source,
                sae_layer=sae_layer,
                steering=self.steering,
            )
            measurements = shaped.get("measurements")
            if not measurements:
                continue
            row = output_rows[match_index][token_index]
            if payload.scores:
                row["probes"] = {
                    name: round(float(value), 6)
                    for name, value in payload.scores.items()
                }
            if payload.per_layer_scores:
                row["per_layer_scores"] = payload.per_layer_scores
            row["measurements"] = measurements
            populated_matches.add(match_index)

        for match_index in sorted(populated_matches):
            match = matches[match_index]
            self.tree.set_authored_token_scores(
                match.target.node_id,
                output_rows[match_index],
                thinking=match.target.thinking,
            )


def plan_authored_capture(
    tree: "LoomTree",
    tokenizer: Any,
    assistant_node_id: str | None,
    *,
    raw: bool,
    monitor: "Monitor",
    lens: "LensInstrument",
    sae: "SaeInstrument",
    monitor_active: bool,
    lens_active: bool,
    sae_active: bool,
    readout_top_k: int,
    persist_per_layer_scores: bool,
    content_end: Callable[[], int],
) -> AuthoredPromptCapture | None:
    """The authored-prompt capture this generation needs, or None.

    Nothing is tokenized unless some channel will measure the rows: with no
    live monitor probe and no lens or SAE consumer the plan is None before
    the path is walked.  ``content_end`` renders the prompt without its
    generation header and returns its length (the search bound that keeps
    the header out of the match); it runs only when there are chat-rendered
    targets to find.
    """
    if not (monitor_active or lens_active or sae_active):
        return None
    targets = pending_targets(tree, tokenizer, assistant_node_id, raw=raw)
    if not targets:
        return None
    node = tree.nodes.get(assistant_node_id) if assistant_node_id is not None else None
    recipe = node.recipe if node is not None else None
    return AuthoredPromptCapture(
        tree=tree,
        tokenizer=tokenizer,
        targets=targets,
        monitor=monitor,
        lens=lens,
        sae=sae,
        monitor_active=monitor_active,
        lens_active=lens_active,
        sae_active=sae_active,
        readout_top_k=readout_top_k,
        persist_per_layer_scores=persist_per_layer_scores,
        steering=recipe.steering if recipe is not None else None,
        search_end=None if raw else int(content_end()),
    )


__all__ = [
    "AuthoredPromptCapture",
    "AuthoredPromptMatch",
    "AuthoredPromptTarget",
    "match_targets",
    "pending_targets",
    "plan_authored_capture",
    "rfind_token_subsequence",
    "tokenize_authored_text",
]
