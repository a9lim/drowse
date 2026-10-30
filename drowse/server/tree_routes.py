"""Native loom-tree route group (``/drowse/v1/sessions/{id}/tree/*``)."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from drowse.server.runtime import run_in_thread, session_lock, session_router
from drowse.server.response_models import (
    ActivePathJSON,
    CastMemberResponse,
    CastRosterResponse,
    EdgeLabelResponse,
    FilterMatchesJSON,
    JointLogprobsJSON,
    LoomNodeDetailJSON,
    LoomTreeJSON,
    NodeDiffJSON,
    TranscriptLoadResponseJSON,
    TranscriptResponse,
    TreeBranchResponse,
    TreeDeleteResponse,
    TreeRestoreResponse,
)
from drowse.server.tree_models import (
    CastMemberRequest,
    JointLogprobsRequest,
    TreeBranchRequest,
    TreeDiffRequest,
    TreeEditRequest,
    TreeNavigateRequest,
    TreeNoteRequest,
    TreeRestoreRequest,
    TreeStarRequest,
    TreeTranscriptLoadRequest,
    TreeTranscriptRequest,
    active_path_json,
    node_json,
    cast_json,
    tree_to_json,
)

_JOINT_LOGPROB_CACHE_MAX = 8


def _cached_joint_logprobs(cache: dict[Any, Any], key: Any) -> Any | None:
    hit = cache.pop(key, None)
    if hit is not None:
        cache[key] = hit
    return hit


def _remember_joint_logprobs(cache: dict[Any, Any], key: Any, value: Any) -> None:
    cache.pop(key, None)
    while len(cache) >= _JOINT_LOGPROB_CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = value


def register_tree_routes(app: FastAPI) -> None:
    """Mount the loom ``tree/*`` routes."""
    session = app.state.session
    router = session_router()

    @router.get("/tree")
    def get_tree() -> LoomTreeJSON:
        """Full tree as JSON, token rows included.

        :meth:`LoomTree.to_dict` with the configured ``cast`` replaced by the
        effective roster (each member annotated with its ``origin``).
        Surfaces hydrate their state from this on bootstrap and reconcile
        via the WS ``tree_mutated`` delta stream after.
        """
        return tree_to_json(session)

    @router.put("/tree")
    async def restore_tree(
        req: TreeRestoreRequest,
    ) -> TreeRestoreResponse:
        """Atomically restore a complete Loom tree from its JSON form.

        This is the inverse of the full-tree GET and backs the dashboard's
        save/load affordance.  Structural/schema validation is delegated to
        :meth:`LoomTree.from_dict`; model mismatch is refused because saved
        token ids and stamped generation recipes are model-specific.
        """
        async with session_lock(app):
            session.restore_tree(req.tree)
        return {
            "rev": session.tree.rev,
            "root_id": session.tree.root_id,
            "active_node_id": session.tree.active_node_id,
            "nodes": len(session.tree.nodes),
        }

    @router.get("/tree/active")
    def get_tree_active() -> ActivePathJSON:
        """Active path: chat messages + parallel node-id list.

        Cheaper than the full tree for surfaces that only need the
        currently-rendered conversation. The node-id list is parallel to
        ``messages`` so a click on message ``i`` maps to ``node_ids[i]``.
        """
        return active_path_json(session)

    @router.post("/tree/navigate")
    async def tree_navigate(
        req: TreeNavigateRequest,
    ) -> ActivePathJSON:
        """Re-point the active node.

        Free relative to in-flight generation: the gen continues attached
        to its original target, the user simply sees a different active
        path.
        """
        session.tree.navigate(req.node_id)
        return active_path_json(session)

    @router.post("/tree/edit")
    async def tree_edit(
        req: TreeEditRequest,
    ) -> LoomNodeDetailJSON:
        """In-place text replacement.

        409 when the node is in the reservation of an in-flight
        generation (mapped via ``DrowseError.user_message``); 404 on
        unknown id; 400 on root-edit or other invalid ops.
        """
        session.tree.edit(req.node_id, req.text)
        return node_json(session, req.node_id)

    @router.post("/tree/branch")
    async def tree_branch(
        req: TreeBranchRequest,
    ) -> TreeBranchResponse:
        """Always-sibling — create a new node next to ``node_id``.

        Allowed during in-flight generation; the new sibling sits on the
        same user-parent as the gen target without disturbing it.
        Returns ``{node_id, node, active_path}`` so the caller can place
        the new node and (if it became active) re-render the chat
        without a follow-up fetch.
        """
        new_id = session.tree.branch(
            req.node_id, req.text, role=req.role,
        )
        return {
            "node_id": new_id,
            "node": node_json(session, new_id),
            "active_path": active_path_json(session),
        }

    @router.delete("/tree/{node_id}")
    async def tree_delete(
        node_id: str,
    ) -> TreeDeleteResponse:
        """Subtree delete.

        400 for the root delete; 409 when the subtree intersects an
        in-flight generation's reservation; 404 on unknown id.  When
        the active node sits inside the deleted subtree, the engine
        re-seats the active pointer on the surviving parent and emits
        the new ``active_node_id`` on the mutation event.  Returns
        ``{removed: <count>}``.
        """
        removed = session.tree.delete_subtree(node_id)
        return {"removed": removed}

    @router.post("/tree/star")
    async def tree_star(
        req: TreeStarRequest,
    ) -> LoomNodeDetailJSON:
        """Toggle a node's ``starred`` flag.

        Decoration-only; never raises a concurrency conflict.
        """
        session.tree.star(req.node_id, req.on)
        return node_json(session, req.node_id)

    @router.post("/tree/note")
    async def tree_note(
        req: TreeNoteRequest,
    ) -> LoomNodeDetailJSON:
        """Set a node's free-text ``notes`` annotation.

        Decoration-only; never raises a concurrency conflict.
        """
        session.tree.annotate(req.node_id, req.text)
        return node_json(session, req.node_id)

    @router.get("/tree/cast")
    def tree_cast() -> CastRosterResponse:
        """The tree's effective cast roster: label → member (recipe + notes
        + origin).

        Also rides the full-tree GET (``cast`` key) and every
        ``tree_mutated`` frame — this endpoint is the cheap standalone read.
        """
        return {"cast": cast_json(session)}

    @router.put("/tree/cast/{label}")
    async def tree_cast_put(
        label: str, req: CastMemberRequest,
    ) -> CastMemberResponse:
        """Create or replace the cast member under ``label``.

        Validates the label as a role slug and the steering expression's
        syntax up front (400 via ``DrowseError.user_message`` on either).
        Decoration-tier — never a concurrency conflict.
        """
        member = session.set_cast_member(
            label,
            steering=req.steering,
            thinking=req.thinking,
            seed=req.seed,
            notes=req.notes,
        )
        return {"label": label, "member": member.to_dict()}

    @router.delete("/tree/cast/{label}", status_code=204)
    async def tree_cast_delete(label: str):
        """Drop the cast member under ``label`` (no-op when absent)."""
        session.remove_cast_member(label)
        return None

    @router.post("/tree/reset", status_code=204)
    async def tree_reset():
        """Drop the entire tree and rebuild a fresh root.

        Equivalent to ``session.clear_history()``; 409 when a generation
        is in flight (per the concurrency invariant — ``reset`` cannot
        race the gen path because the gen path owns the streaming target
        in the tree itself).
        """
        async with session_lock(app):
            session.clear_history()
        return Response(status_code=204)

    @router.post("/tree/transcript")
    def tree_transcript(
        req: TreeTranscriptRequest,
    ) -> TranscriptResponse:
        """Render the path ending at ``node_id`` (or active) as transcript YAML.

        Uses :meth:`Transcript.from_path` so probe sha256 hashes are real
        and the YAML round-trips through :meth:`Transcript.from_yaml`
        cleanly.  Returns
        ``{"yaml": "<text>", "node_id": "<leaf-of-rendered-path>"}``.
        """
        from drowse.loom.transcript import Transcript

        leaf = req.node_id if req.node_id is not None else session.tree.active_node_id
        # Validate the id before touching the renderer so the 404 lands
        # cleanly through the existing ``DrowseError`` handler.
        session.tree.get(leaf)
        transcript = Transcript.from_path(leaf, session)
        return {"yaml": transcript.to_yaml(), "node_id": leaf}

    @router.post("/tree/transcript/load")
    async def tree_transcript_load(
        req: TreeTranscriptLoadRequest,
    ) -> TranscriptLoadResponseJSON:
        """Import a transcript YAML into the live session tree.

        Wraps :meth:`Transcript.from_yaml` + :meth:`Transcript.import_into`
        (one atomic graft: one revision and one ``import`` delta).
        Modes are ``"default"`` / ``"here"`` / ``"merge"``; ``strict``
        refuses on probe-hash drift.  Returns
        ``{"leaf_id": "<id>", "rev": <int>, "guards": [...]}``.

        Guards (model mismatch, system-prompt mismatch, probe drift) are
        also stamped on the imported branch's root node as ``notes`` so
        the surfaces can show a banner there.  Returning them in the body
        too saves the client one fetch.
        """
        from drowse.loom.transcript import (
            Transcript,
            TranscriptError,
            TranscriptFormatError,
        )

        mode = req.mode
        try:
            transcript = Transcript.from_yaml(req.yaml)
        except TranscriptFormatError as e:
            raise HTTPException(400, f"invalid transcript: {e}") from e
        import warnings

        captured: list[str] = []

        def _on_warning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            captured.append(str(message))

        async with session_lock(app):
            with warnings.catch_warnings():
                warnings.showwarning = _on_warning
                try:
                    leaf_id = await run_in_thread(
                        transcript.import_into,
                        session,
                        mode=mode,
                        strict=req.strict,
                    )
                except TranscriptError as e:
                    raise HTTPException(400, str(e)) from e
        return {
            "leaf_id": leaf_id,
            "rev": session.tree.rev,
            "guards": captured,
        }

    @router.get("/tree/edge-label")
    def tree_edge_label(
        parent_id: str, child_id: str,
    ) -> EdgeLabelResponse:
        """Steering-delta label for the parent → child edge.

        Returns ``{"label": "<text>"}`` — empty string when the two
        expressions are identical.  Both nodes must exist; the label is
        computed from each node's :attr:`LoomNode.steering_expr` (``None``
        for an authored turn, so the delta reads "from-nothing").
        """
        from drowse.loom.diff import steering_delta

        parent = session.tree.get(parent_id)
        child = session.tree.get(child_id)
        return {"label": steering_delta(parent.steering_expr, child.steering_expr)}

    @router.get("/tree/filter")
    def tree_filter(expr: str = "") -> FilterMatchesJSON:
        """Apply a filter-grammar expression and return matching node ids.

        Grammar in :mod:`drowse.loom.filter` — comma-AND'd
        ``agg:|any:|last:<probe> <op> <threshold>`` clauses.  ``agg:``
        reads a node's aggregate readings; ``any:`` / ``last:`` read its
        persisted per-token measurement rows.  Empty ``expr`` returns
        every node id (clears the filter).  Bad expressions land as 400
        via :class:`FilterParseError`.
        """
        from drowse.loom.filter import FilterParseError, filter_tree

        text = (expr or "").strip()
        if not text:
            return {"expr": "", "matching_node_ids": []}
        try:
            matches = filter_tree(session.tree, text)
        except FilterParseError as e:
            raise HTTPException(400, str(e)) from e
        return {"expr": text, "matching_node_ids": sorted(matches)}

    @router.post("/tree/diff")
    def tree_diff(req: TreeDiffRequest) -> NodeDiffJSON:
        """Cross-branch diff between two generated nodes.

        Returns a JSON view of :class:`NodeDiff` (text spans + readings
        deltas) augmented with the steering deltas against the shared
        parent and a byte-aligned per-token span table (present when both
        nodes carry token rows; ``reading_deltas`` stays empty).
        """
        from drowse.loom.diff import per_token_diff, steering_delta

        diff = session.diff_nodes(req.a_id, req.b_id)
        a_node = session.tree.get(req.a_id)
        b_node = session.tree.get(req.b_id)

        # Steering-delta against the shared parent's expression — only
        # meaningful for sibling diffs (parent_id present).
        parent_expr: str | None = None
        if diff.parent_id is not None:
            parent = session.tree.nodes.get(diff.parent_id)
            if parent is not None:
                parent_expr = parent.steering_expr

        a_expr = a_node.steering_expr
        b_expr = b_node.steering_expr

        # Per-token diff: only when both nodes carry token sequences.
        # Tokens may be absent on serialized-only nodes (loaded transcripts).
        a_tok_strs: list[str] = []
        if a_node.tokens:
            a_tok_strs = [t.get("text", "") for t in a_node.tokens]
        b_tok_strs: list[str] = []
        if b_node.tokens:
            b_tok_strs = [t.get("text", "") for t in b_node.tokens]
        per_token_spans: list[dict[str, Any]] = []
        if a_tok_strs and b_tok_strs:
            spans = per_token_diff(a_tok_strs, b_tok_strs)
            per_token_spans.extend(
                {
                    "a_index": sp.a_index,
                    "b_index": sp.b_index,
                    "a_text": sp.a_text,
                    "b_text": sp.b_text,
                    "aligned": sp.aligned,
                    "reading_deltas": [
                        {
                            "name": rd.name,
                            "delta": round(float(rd.delta), 6),
                            "a_value": round(float(rd.a_value), 6),
                            "b_value": round(float(rd.b_value), 6),
                        }
                        for rd in sp.reading_deltas
                    ],
                }
                for sp in spans
            )

        return cast(NodeDiffJSON, {
            "a_id": diff.a_id,
            "b_id": diff.b_id,
            "parent_id": diff.parent_id,
            "a_text": a_node.text,
            "b_text": b_node.text,
            "a_applied_steering": a_expr,
            "b_applied_steering": b_expr,
            "parent_applied_steering": parent_expr,
            "steering_delta": steering_delta(a_expr, b_expr),
            "parent_to_a_delta": (
                steering_delta(parent_expr, a_expr)
                if parent_expr is not None or a_expr is not None
                else ""
            ),
            "parent_to_b_delta": (
                steering_delta(parent_expr, b_expr)
                if parent_expr is not None or b_expr is not None
                else ""
            ),
            "text": [
                {"state": sp.state, "text": sp.text}
                for sp in diff.text
            ],
            "readings": [
                {
                    "name": rd.name,
                    "delta": round(float(rd.delta), 6),
                    "a_value": round(float(rd.a_value), 6),
                    "b_value": round(float(rd.b_value), 6),
                }
                for rd in diff.readings
            ],
            "per_token": per_token_spans,
        })

    @router.post("/tree/joint-logprobs")
    async def tree_joint_logprobs(
        req: JointLogprobsRequest,
    ) -> JointLogprobsJSON:
        """Cross-evaluation between two sibling generated nodes.

        Force-replays each branch under the node's stamped recipe, steering hooks, probe
        gates, penalties, logit bias, and sampler transform, then returns
        per-aligned-position records carrying both branches' chosen-token
        logprobs *and* the cross-branch evaluation (what each side would
        have given the other's chosen token at the same byte-aligned
        position).

        Cache shape:
        * Stored on ``session.joint_logprob_cache: dict[tuple[str,
          str], JointLogprobs]`` keyed by sorted ``(a_id, b_id)`` so
          the symmetric pair shares an entry.
        * Invalidated by tree edits/deletes/finalize events in
          ``DrowseSession``; navigate/star/note leave it intact.

        Held under the session lock because the forward passes compete
        for the same model with any concurrent generation; the request
        queues FIFO at the lock rather than 409ing, and a lock timeout is
        a 503 — the forwards never run unlocked.
        """
        from drowse.generation.joint_logprobs import (
            compute_joint_logprobs,
            _cache_key,
            reorient_for_request,
        )

        if req.a_id == req.b_id:
            raise HTTPException(400, "a_id and b_id must differ")
        if req.a_id not in session.tree.nodes:
            raise HTTPException(404, f"unknown node id: {req.a_id}")
        if req.b_id not in session.tree.nodes:
            raise HTTPException(404, f"unknown node id: {req.b_id}")

        cache = session.joint_logprob_cache

        key = _cache_key(req.a_id, req.b_id)
        hit = _cached_joint_logprobs(cache, key)
        if hit is None:
            async with session_lock(app):
                # Double-check under lock — another request may have
                # populated the cache while we waited.
                hit = _cached_joint_logprobs(cache, key)
                if hit is None:
                    hit = await run_in_thread(
                        compute_joint_logprobs, session, req.a_id, req.b_id,
                    )
                    _remember_joint_logprobs(cache, key, hit)
        return cast(
            JointLogprobsJSON,
            reorient_for_request(hit, req.a_id, req.b_id).to_dict(),
        )

    app.include_router(router)
