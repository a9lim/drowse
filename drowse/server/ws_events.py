"""Native WebSocket event-shaping helpers."""

from __future__ import annotations

from typing import Any

from drowse.generation.results import TokenEvent
from drowse.session.session import DrowseSession


def build_token_event(
    session: DrowseSession,
    node_holder: list[str | None],
    event: TokenEvent,
) -> dict[str, Any]:
    """Build one native WS ``token`` frame from the token tap's event."""
    node_id = node_holder[0]
    if node_id is None:
        node_id = session.tree.active_node_id
        node_holder[0] = node_id

    frame: dict[str, Any] = {
        "type": "token",
        "text": event.text,
        "thinking": bool(event.thinking),
        "token_id": int(event.token_id),
        "node_id": node_id,
    }
    if event.logprob is not None:
        frame["logprob"] = float(event.logprob)
    if event.perplexity is not None:
        frame["perplexity"] = float(event.perplexity)
    if event.top_alts:
        frame["top_alts"] = [
            {"id": int(a.id), "text": a.text, "logprob": float(a.logprob)}
            for a in event.top_alts
        ]
    if event.raw_index is not None:
        frame["raw_index"] = int(event.raw_index)

    # The measurement envelope is the single read-side wire record: geometry /
    # lens / SAE ``instruments`` plus the flat ``scores`` / ``per_layer_scores``
    # views (:func:`drowse.readout.measurements.build_measurements`).  The token
    # tap builds it once and stores the same envelope on the loom row, so live
    # and rehydrated tokens share one authority.
    if event.measurements:
        frame["measurements"] = event.measurements

    return frame
