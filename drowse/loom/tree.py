"""LoomTree — engine-side tree of conversation nodes.

Nodes are conversation turns and children are alternative continuations.
The active path is what the model sees as context for the next
generation; the rest of the tree is preserved as dead branches for
navigation.

This module owns the data model only.  Generation integration, the
gen-lock, and HTTP/WS event delivery live in the session and server
layers.  Mutators that can conflict with an in-flight generation consult
the conflict checker the session installs (:meth:`LoomTree.set_conflict_check`)
and raise :class:`MutationDuringGenerationError` on a conflict, because the
gen-lock state lives on the session, not the tree.

The primitives (edit, branch, navigate, delete_subtree, graft, plus
``regenerate`` exposed via ``session.generate(parent_node_id=..., n=...)``)
are all that exist at the engine level.  Surface-level verbs (slash
commands, keyboard shortcuts, context menus) compose from these.

Per-node token rows (``tokens`` and ``thinking_tokens``) are held in memory
during streaming.  ``LoomTree.to_dict`` omits them by default;
``LoomTree.save`` writes them to a compressed sidecar next to the main tree
JSON so explicit save/load keeps the analysis surface intact.  The tree is
otherwise in-memory only — there is no automatic cross-session persistence.
Recipes and cast members live in :mod:`drowse.loom.recipe`, the error types
in :mod:`drowse.loom.errors`.
"""

from __future__ import annotations

import gzip
import json
import os
import secrets
import tempfile
import zlib
from contextlib import suppress
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, BinaryIO, Callable, Iterator, Literal, Mapping, Sequence, cast

from drowse.events import EventBus, LoomMutated
from drowse.loom.errors import (
    InvalidNodeOperationError,
    LoomTreeError,
    UnknownNodeError,
    is_number,
    require_fields,
)
from drowse.loom.recipe import CastMember, Recipe
from drowse.model.role_templates import InvalidRoleError, validate_role

# ---------------------------------------------------------------------------
# ulid — tiny inline implementation
# ---------------------------------------------------------------------------
# Crockford base32; sortable by timestamp prefix.  No external dep — ~30
# lines is cheaper than pulling python-ulid for the only thing we need
# from it.  Format: 10 chars (48-bit ms timestamp) + 16 chars (80-bit
# randomness) = 26 chars total.  Lexicographic order matches chronological
# order on the timestamp prefix; ties broken by the random tail.

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    """Return a fresh 26-char ULID."""
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = secrets.randbits(80)
    n = (ts_ms << 80) | rand
    out = []
    for _ in range(26):
        out.append(_ULID_ALPHABET[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def _read_json_bounded(stream: BinaryIO | gzip.GzipFile, max_bytes: int, label: str) -> Any:
    raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise LoomTreeError(f"{label} exceeds the {max_bytes}-byte import limit")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise LoomTreeError(f"{label} is invalid JSON") from error


_NODE_FIELDS = frozenset({
    "id", "parent_id", "role", "text", "role_label", "thinking_text",
    "recipe", "aggregate_readings", "applied_steering", "finish_reason",
    "starred", "notes", "created_at", "edited_at", "edit_count",
    "mean_logprob", "mean_surprise",
})
_TOKEN_FIELDS = frozenset({"tokens", "thinking_tokens", "raw_token_ids"})
_NODE_FIELDS_WITH_TOKENS = _NODE_FIELDS | _TOKEN_FIELDS
_TREE_FIELDS = frozenset({
    "tree_format", "drowse_version", "model_id", "session_id", "name", "rev",
    "root_id", "active_node_id", "nodes", "children_of", "cast",
})

#: One persisted token row — the token tap's ``{token_id, text, logprob,
#: raw_index, …, measurements}`` dict.  Rows are never mutated after
#: creation, so a shallow copy of a token list is a stable snapshot.
TokenRow = dict[str, Any]

Role = Literal["user", "assistant", "system"]


def _copy_list(value: list[Any] | None) -> list[Any] | None:
    return list(value) if value is not None else None


@dataclass
class LoomNode:
    """A single node in the loom tree.

    Token lists (``tokens``, ``thinking_tokens``) are owned here for live
    streaming but are persisted in side files by the session store; the
    tree's ``to_dict()`` omits them so the main tree file stays small.

    ``edit_count`` and ``edited_at`` flag assistant nodes whose text has
    been mutated in place since the model emitted it — downstream
    consumers (transcript replay, comparison views) use this to know
    that the text isn't pristine from the model.
    """

    id: str
    parent_id: str | None
    role: Role
    text: str = ""
    # Per-turn role-substitution label (roleplay scaffold).  The custom
    # label this turn was *sent* with — ``"captain"`` on a user node,
    # ``"pirate"`` on an assistant node — or ``None`` for the family's
    # standard label.  Stamped at send time (immutable afterward) so the
    # chat-render walks each turn with its own role and the transcript /
    # glyph display matches.  Distinct from the steering-driven
    # ``_active_role`` (transient, never persisted here).
    role_label: str | None = None
    # Verbatim thinking text for the turn — a committed thinking block
    # the author typed (any role), or the decoded thinking channel of a
    # generated node (stamped at finalize).  Rendered through the family
    # think delimiters by the scene stitcher, which applies the family's
    # history policy (strip families render it only while the turn is
    # last).  Plain text-scale string, so it lives in the main tree
    # JSON, not the token sidecar.
    thinking_text: str | None = None
    tokens: list[TokenRow] | None = None
    thinking_tokens: list[TokenRow] | None = None
    recipe: Recipe | None = None
    aggregate_readings: dict[str, float] = field(default_factory=dict)
    applied_steering: str | None = None
    finish_reason: str | None = None
    starred: bool = False
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    edited_at: float | None = None
    edit_count: int = 0
    # Mean chosen-token logprob over the non-thinking response span,
    # computed in :meth:`DrowseSession._generate_core` and stamped at
    # :meth:`LoomTree.finalize_assistant` time. ``None`` for legacy
    # nodes replayed from pre-logit-pass transcripts. ``mean_surprise``
    # caches ``-mean_logprob`` so the sidebar's "sort by surprise" mode
    # is a keyboard sort, not a recompute. Both fields surface through
    # ``to_dict`` / ``from_dict`` for tree persistence + WS bridge.
    mean_logprob: float | None = None
    mean_surprise: float | None = None
    # Raw decode-step token ids for an assistant node — the exact
    # sequence the engine sampled, *including* the structural delimiters
    # (`<think>` / `</think>` / channel markers) that ``tokens`` /
    # ``thinking_tokens`` suppress, and with partial-UTF-8 bytes
    # unmerged.  Stamped at :meth:`LoomTree.finalize_assistant` from the
    # engine's ``generated_ids``.  This is the forceable prefix a logit
    # fork replays; ``None`` for transcript-loaded nodes, where
    # the fork affordance falls back to disabled.  Persisted in the token
    # sidecar alongside ``tokens`` (bulky, per-decode-step granularity).
    # ``None`` remains a legitimate current value for transcript-imported
    # turns, and is persisted explicitly rather than inferred from absence.
    raw_token_ids: list[int] | None = None

    def to_dict(self, *, include_tokens: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "parent_id": self.parent_id,
            "role": self.role,
            "text": self.text,
            "role_label": self.role_label,
            "thinking_text": self.thinking_text,
            "aggregate_readings": dict(self.aggregate_readings),
            "applied_steering": self.applied_steering,
            "finish_reason": self.finish_reason,
            "starred": self.starred,
            "notes": self.notes,
            "created_at": self.created_at,
            "edited_at": self.edited_at,
            "edit_count": self.edit_count,
            "mean_logprob": self.mean_logprob,
            "mean_surprise": self.mean_surprise,
            "recipe": self.recipe.to_dict() if self.recipe is not None else None,
        }
        if include_tokens:
            # Copy the lists: a streaming node's lists keep growing after
            # this snapshot is taken (rows themselves are never mutated).
            out["tokens"] = _copy_list(self.tokens)
            out["thinking_tokens"] = _copy_list(self.thinking_tokens)
            out["raw_token_ids"] = _copy_list(self.raw_token_ids)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoomNode":
        required = _NODE_FIELDS_WITH_TOKENS if "tokens" in data else _NODE_FIELDS
        require_fields(data, required, "loom node")
        recipe = None
        if data["recipe"] is not None:
            if not isinstance(data["recipe"], dict):
                raise LoomTreeError("loom node field 'recipe' must be an object or null")
            recipe = Recipe.from_dict(data["recipe"])
        if data["role"] not in ("user", "assistant", "system"):
            raise LoomTreeError(f"invalid loom node role {data['role']!r}")
        if not isinstance(data["id"], str) or not data["id"]:
            raise LoomTreeError("loom node field 'id' must be a non-empty string")
        if data["parent_id"] is not None and not isinstance(data["parent_id"], str):
            raise LoomTreeError("loom node field 'parent_id' must be a string or null")
        for field_name in ("text", "notes"):
            if not isinstance(data[field_name], str):
                raise LoomTreeError(f"loom node field {field_name!r} must be a string")
        for field_name in ("role_label", "thinking_text", "applied_steering", "finish_reason"):
            value = data[field_name]
            if value is not None and not isinstance(value, str):
                raise LoomTreeError(
                    f"loom node field {field_name!r} must be a string or null"
                )
        if not isinstance(data["aggregate_readings"], dict) or any(
            not isinstance(name, str) or not is_number(value)
            for name, value in data["aggregate_readings"].items()
        ):
            raise LoomTreeError(
                "loom node field 'aggregate_readings' must map strings to numbers"
            )
        if not isinstance(data["starred"], bool):
            raise LoomTreeError("loom node field 'starred' must be a boolean")
        if not is_number(data["created_at"]):
            raise LoomTreeError("loom node field 'created_at' must be a number")
        if data["edited_at"] is not None and not is_number(data["edited_at"]):
            raise LoomTreeError("loom node field 'edited_at' must be a number or null")
        if (not isinstance(data["edit_count"], int) or isinstance(data["edit_count"], bool)
                or data["edit_count"] < 0):
            raise LoomTreeError("loom node field 'edit_count' must be a non-negative integer")
        for field_name in ("mean_logprob", "mean_surprise"):
            value = data[field_name]
            if value is not None and not is_number(value):
                raise LoomTreeError(
                    f"loom node field {field_name!r} must be a number or null"
                )
        for field_name in _TOKEN_FIELDS & data.keys():
            value = data[field_name]
            if value is not None and not isinstance(value, list):
                raise LoomTreeError(
                    f"loom node field {field_name!r} must be a list or null"
                )
        raw_token_ids = data.get("raw_token_ids")
        if raw_token_ids is not None and any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in raw_token_ids
        ):
            raise LoomTreeError("loom node raw_token_ids must contain integers")
        return cls(
            id=data["id"],
            parent_id=data["parent_id"],
            role=data["role"],
            text=data["text"],
            role_label=data["role_label"],
            thinking_text=data["thinking_text"],
            tokens=data.get("tokens"),
            thinking_tokens=data.get("thinking_tokens"),
            recipe=recipe,
            aggregate_readings=dict(data["aggregate_readings"]),
            applied_steering=data["applied_steering"],
            finish_reason=data["finish_reason"],
            starred=data["starred"],
            notes=data["notes"],
            created_at=float(data["created_at"]),
            edited_at=data["edited_at"],
            edit_count=data["edit_count"],
            mean_logprob=data["mean_logprob"],
            mean_surprise=data["mean_surprise"],
            raw_token_ids=data.get("raw_token_ids"),
        )


# ---------------------------------------------------------------------------
# LoomTree
# ---------------------------------------------------------------------------


# Exact format versions for ``tree.json`` and its optional token sidecar.
# Current-only loaders require these versions and every field written by the
# corresponding schema; there is no implicit v1 or additive-field migration.
TREE_FORMAT_VERSION = 2
_TREE_MAX_LOAD_BYTES = 64 * 1024 * 1024
_TOKEN_SIDECAR_MAX_LOAD_BYTES = 256 * 1024 * 1024
TOKEN_SIDECAR_FORMAT_VERSION = 2


# Hook signature used by the session to enforce gen-lock conflicts.  The
# tree calls this before any mutator; the session implementation maps
# the (node_id, op) pair to either a no-op (free op) or a raise
# (MutationDuringGenerationError).  Default is a no-op for unit-test
# usage of the tree without a session.
ConflictChecker = Callable[[str, str], None]


def _noop_conflict_check(node_id: str, op: str) -> None:
    del node_id, op  # signature is the protocol; no work to do here
    return


class LoomTree:
    """Mutation-safe tree of conversation nodes.

    Owned by :class:`drowse.session.session.DrowseSession`; callers go through
    session methods so locking + event emission happen in one place.  The
    tree's own methods are thread-safe under an internal ``RLock`` so the
    session can call them from arbitrary threads (generation workers and server
    handlers).

    Operations that mutate bump :attr:`rev` and emit a :class:`LoomMutated`
    event through the session's :class:`EventBus`.  Surfaces track the
    last-seen ``rev`` and full-refetch the tree if they detect a gap.
    """

    def __init__(
        self,
        *,
        events: EventBus | None = None,
        model_id: str | None = None,
        session_id: str | None = None,
        name: str | None = None,
        conflict_check: ConflictChecker | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._events: EventBus | None = events  # emits LoomMutated when set
        self.model_id: str | None = model_id
        self.session_id: str | None = session_id
        self.name: str | None = name
        self._conflict_check: ConflictChecker = conflict_check or _noop_conflict_check

        self.nodes: dict[str, LoomNode] = {}
        self.children_of: dict[str, list[str]] = {}
        self.rev: int = 0
        # Cast roster (phase 3): label → CastMember.  Tree-scoped, rides
        # ``to_dict``/``save`` so a saved conversation carries its cast.
        self.cast: dict[str, CastMember] = {}

        # Synthetic root: role="system", text empty, no parent.  First user
        # turn is its child.  This keeps the tree structure uniform (every
        # real node has a parent) without burning a special-case branch
        # in the active-path walker.
        root = LoomNode(id=_ulid(), parent_id=None, role="system")
        self.nodes[root.id] = root
        self.children_of[root.id] = []
        self.root_id: str = root.id
        self.active_node_id: str = root.id

    # ------------------------------------------------------------------
    # Conflict-check wiring
    # ------------------------------------------------------------------

    def set_conflict_check(self, fn: ConflictChecker | None) -> None:
        """Install a conflict-check hook.

        The session sets this in its ``__init__`` so all mutator paths
        consult ``session._loom_conflict_check`` before proceeding.
        """
        self._conflict_check = fn or _noop_conflict_check

    def attach_events(self, events: EventBus | None) -> None:
        """Wire (or unwire) the EventBus that mutations emit on."""
        self._events = events

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, node_id: str) -> LoomNode:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise UnknownNodeError(node_id) from None

    def has(self, node_id: str) -> bool:
        return node_id in self.nodes

    def node_to_dict(
        self, node_id: str, *, include_tokens: bool = False,
    ) -> dict[str, Any]:
        """Serialize one node under the tree lock — a stable snapshot even
        while a generation streams rows into it."""
        with self._lock:
            return self.get(node_id).to_dict(include_tokens=include_tokens)

    def children(self, node_id: str) -> list[LoomNode]:
        with self._lock:
            return [self.nodes[c] for c in self.children_of.get(node_id, [])]

    def descendants(self, node_id: str) -> Iterator[LoomNode]:
        """Depth-first iteration over the descendants of ``node_id``."""
        with self._lock:
            stack = list(self.children_of.get(node_id, []))
        # We snapshot the child list under lock, then walk it lock-free.
        # Concurrent mutations of the subtree mid-iteration aren't supported
        # — callers wanting that need to snapshot first.
        while stack:
            cur = stack.pop()
            node = self.nodes.get(cur)
            if node is None:
                continue
            yield node
            stack.extend(self.children_of.get(cur, []))

    def path_to(self, node_id: str) -> list[LoomNode]:
        """Return the path from the root to ``node_id``, inclusive."""
        with self._lock:
            if node_id not in self.nodes:
                raise UnknownNodeError(node_id)
            chain: list[LoomNode] = []
            cur: str | None = node_id
            while cur is not None:
                chain.append(self.nodes[cur])
                cur = self.nodes[cur].parent_id
            chain.reverse()
            return chain

    def active_path(self) -> list[LoomNode]:
        """Path from the root to :attr:`active_node_id`."""
        return self.path_to(self.active_node_id)

    def cast_roster(self) -> dict[str, CastMember]:
        """Return the configured cast plus every role observed in the tree.

        ``user`` and ``assistant`` are structural defaults and therefore always
        exist.  Per-turn labels join the roster automatically across the whole
        loom (not merely the active path), so navigation cannot make speakers
        flicker in and out.  Explicit members win because they may carry a
        standing recipe or notes.
        """
        with self._lock:
            roster: dict[str, CastMember] = {
                "user": CastMember(),
                "assistant": CastMember(),
            }
            for node in self.nodes.values():
                if node.role == "system":
                    continue
                label = node.role_label or node.role
                if label:
                    roster.setdefault(label, CastMember())
            roster.update(self.cast)
            return roster

    def messages_for(
        self,
        leaf_id: str | None = None,
        *,
        include_system: bool = False,
        with_labels: bool = False,
    ) -> list[dict[str, str]]:
        """Return the active-path (or path to ``leaf_id``) as chat messages.

        Returns the v2 ``[{"role": ..., "content": ...}, ...]`` shape that
        the rest of the engine + servers consume.  Skips the synthetic
        root by default (its empty text isn't a real system prompt) —
        callers wanting an explicit system prompt prepend it themselves.

        ``with_labels`` adds each node's per-turn ``role_label`` under a
        ``"label"`` key (the roleplay scaffold) so the chat-render path can
        splice each turn with its own role, plus — when present — the
        node's ``thinking_text`` under ``"thinking"`` (the scene stitcher
        renders it through the family think delimiters, applying the
        family's history policy).  Off by default to keep the canonical
        ``{role, content}`` shape transcript / server / OpenAI consumers
        expect.
        """
        target = leaf_id if leaf_id is not None else self.active_node_id
        path = self.path_to(target)
        out: list[dict[str, str]] = []
        for node in path:
            if node.id == self.root_id and not include_system:
                continue
            msg: dict[str, str] = {"role": node.role, "content": node.text}
            if with_labels:
                msg["label"] = node.role_label  # pyright: ignore[reportArgumentType]  # role_label is str | None; with_labels callers expect nullable label
                if node.thinking_text is not None:
                    msg["thinking"] = node.thinking_text
            out.append(msg)
        return out

    def flat_text(self, leaf_id: str | None = None) -> str:
        """Return the active-path (or path-to-``leaf_id``) text, concatenated.

        The base-model / flat-completion analogue of :meth:`messages_for`:
        no roles, no separators, no chat template — every node's text
        joined in root-to-leaf order, with the synthetic system root
        skipped.  The raw-mode generation path feeds this verbatim to the
        tokenizer so a completion model sees one continuous buffer rather
        than a chat-templated transcript.  Returns ``""`` when the tree is
        empty (target resolves to ``None`` or the bare root).
        """
        target = leaf_id if leaf_id is not None else self.active_node_id
        if target is None:
            return ""
        parts: list[str] = []
        for node in self.path_to(target):
            if node.id == self.root_id:
                continue
            parts.append(node.text)
        return "".join(parts)

    def ancestors_of(self, node_id: str) -> Iterator[str]:
        """Yield ``node_id``'s ancestor ids (parent first, root last)."""
        cur = self.nodes[node_id].parent_id
        while cur is not None:
            yield cur
            cur = self.nodes[cur].parent_id

    def is_ancestor_of(self, ancestor_id: str, descendant_id: str) -> bool:
        return any(a == ancestor_id for a in self.ancestors_of(descendant_id))

    def anchor_recipe(self, node_id: str | None) -> Recipe | None:
        """The recipe a regen anchored at ``node_id`` reproduces.

        The node's own recipe when it is generated (a recipe marks a
        generated node whatever its seat), else the nearest generated
        ancestor's, else ``None``.
        """
        if node_id is None:
            return None
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                return None
            if node.recipe is not None:
                return node.recipe
            for ancestor_id in self.ancestors_of(node_id):
                recipe = self.nodes[ancestor_id].recipe
                if recipe is not None:
                    return recipe
            return None

    # ------------------------------------------------------------------
    # Internal mutator scaffolding
    # ------------------------------------------------------------------

    def _emit(self, event: LoomMutated) -> None:
        if self._events is not None:
            self._events.emit(event)

    def _add_child(self, parent_id: str, node: LoomNode) -> None:
        self.nodes[node.id] = node
        self.children_of.setdefault(node.id, [])
        self.children_of.setdefault(parent_id, []).append(node.id)

    # ------------------------------------------------------------------
    # Streaming/generation entry points
    # ------------------------------------------------------------------

    def add_user_turn(
        self,
        text: str,
        parent_id: str | None = None,
        *,
        dedup_existing: bool = True,
        role_label: str | None = None,
        thinking_text: str | None = None,
    ) -> str:
        """Add a user turn under ``parent_id`` (default: the active node).

        If ``dedup_existing`` is set and ``parent_id`` already has a user-turn
        child that renders identically — same text, same ``role_label``, same
        ``thinking_text`` — returns that existing child's id without growing
        the tree.  This spares users a redundant tree level for the regen
        workflow where the user re-sends the same prompt; the same text sent
        under another cast label or with another thinking block is a
        different turn and gets its own node.

        ``role_label`` stamps the per-turn role-substitution label (the
        roleplay scaffold).  ``thinking_text`` is an optional committed
        thinking block (rendered through the family think delimiters by the
        scene stitcher).
        """
        with self._lock:
            parent = parent_id if parent_id is not None else self.active_node_id
            if parent not in self.nodes:
                raise UnknownNodeError(parent)
            self._conflict_check(parent, "add_user_turn")
            if dedup_existing:
                for cid in self.children_of.get(parent, []):
                    sib = self.nodes[cid]
                    if (
                        sib.role == "user"
                        and sib.text == text
                        and sib.role_label == role_label
                        and sib.thinking_text == thinking_text
                    ):
                        self.active_node_id = sib.id
                        self.rev += 1
                        self._emit(LoomMutated(
                            op="navigate", rev=self.rev,
                            active_node_id=self.active_node_id,
                        ))
                        return sib.id
            node = LoomNode(
                id=_ulid(), parent_id=parent, role="user", text=text,
                role_label=role_label, thinking_text=thinking_text,
            )
            self._add_child(parent, node)
            self.active_node_id = node.id
            self.rev += 1
            self._emit(LoomMutated(
                op="add_user", rev=self.rev,
                added=(node.id,), active_node_id=node.id,
            ))
            return node.id

    def begin_assistant(
        self,
        parent_id: str,
        recipe: Recipe | None = None,
        *,
        role_label: str | None = None,
        seat: str = "assistant",
    ) -> str:
        """Create an empty generated node under ``parent_id``.

        Returns the new node id.  The session calls this in the gen
        preamble; subsequent ``append_token`` / ``finalize_assistant``
        calls populate the node as the generation streams.

        ``role_label`` stamps the per-turn role-substitution label the
        model is generating under (the roleplay scaffold).  ``seat`` is
        the structural role the node occupies (the cast model: the model
        can generate into the user seat too — "generated" is provenance,
        carried by the recipe, not a role).
        """
        if seat not in ("user", "assistant"):
            raise InvalidNodeOperationError(
                f"begin_assistant: seat must be 'user' or 'assistant', "
                f"got {seat!r}"
            )
        with self._lock:
            if parent_id not in self.nodes:
                raise UnknownNodeError(parent_id)
            self._conflict_check(parent_id, "begin_assistant")
            node = LoomNode(
                id=_ulid(),
                parent_id=parent_id,
                role=cast(Role, seat),
                recipe=recipe,
                role_label=role_label,
                tokens=[],
                thinking_tokens=[],
            )
            self._add_child(parent_id, node)
            self.active_node_id = node.id
            self.rev += 1
            self._emit(LoomMutated(
                op="begin_assistant", rev=self.rev,
                added=(node.id,), active_node_id=node.id,
            ))
            return node.id

    def begin_continuation(
        self,
        node_id: str,
        recipe: Recipe,
    ) -> str:
        """Reuse ``node_id`` as the target of a same-role continuation.

        The caller has already saved the node's current text as a forced
        decode prefix.  Clear the visible/result payload in place so the
        ordinary token stream and finalizer can rebuild the complete message
        (prefix plus newly sampled tail) without adding a redundant same-role
        child to the tree.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if node_id == self.root_id:
                raise InvalidNodeOperationError(
                    "cannot continue the synthetic root node"
                )
            self._conflict_check(node_id, "begin_assistant")
            node.text = ""
            node.thinking_text = None
            node.tokens = []
            node.thinking_tokens = []
            node.recipe = recipe
            node.aggregate_readings = {}
            node.applied_steering = None
            node.finish_reason = None
            node.mean_logprob = None
            node.mean_surprise = None
            node.raw_token_ids = None
            self.active_node_id = node_id
            self.rev += 1
            self._emit(LoomMutated(
                op="begin_assistant", rev=self.rev,
                updated=(node_id,), active_node_id=node_id,
            ))
            return node_id

    def append_token(
        self,
        node_id: str,
        score: TokenRow,
        *,
        thinking: bool = False,
    ) -> None:
        """Append a streaming token-score blob to ``node_id``.

        Doesn't bump ``rev`` or emit events — token streaming runs in the
        hot loop and we don't want to flood the WS with per-token tree-
        mutated events.  Token deltas ride the existing ``token`` stream with
        a ``node_id`` tag.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if thinking:
                if node.thinking_tokens is None:
                    node.thinking_tokens = []
                node.thinking_tokens.append(score)
            else:
                if node.tokens is None:
                    node.tokens = []
                node.tokens.append(score)

    def set_authored_token_scores(
        self,
        node_id: str,
        scores: list[TokenRow],
        *,
        thinking: bool = False,
    ) -> None:
        """Attach captured prompt-token measurements to an authored node.

        Unlike :meth:`append_token`, this is a one-shot tree mutation: prompt
        rows arrive together after prefill, so the revision advances once and
        clients receive the updated node through the ordinary loom delta.
        Existing rows are never merged here; the session calls this only for a
        previously uncaptured authored channel so its original capture remains
        immutable across later generations.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if node.recipe is not None:
                raise InvalidNodeOperationError(
                    "set_authored_token_scores requires an authored node"
                )
            if thinking:
                node.thinking_tokens = list(scores)
            else:
                node.tokens = list(scores)
            self.rev += 1
            self._emit(LoomMutated(
                op="capture_authored",
                rev=self.rev,
                updated=(node_id,),
                active_node_id=self.active_node_id,
            ))

    def finalize_assistant(
        self,
        node_id: str,
        *,
        text: str,
        aggregate_readings: dict[str, float] | None = None,
        applied_steering: str | None = None,
        finish_reason: str | None = None,
        mean_logprob: float | None = None,
        mean_surprise: float | None = None,
        raw_token_ids: list[int] | None = None,
        thinking_text: str | None = None,
    ) -> None:
        """Mark an in-flight assistant node as complete.

        ``thinking_text`` is the decoded thinking-channel text of a
        generated node (or the authored block on a committed one);
        ``None`` leaves the node's prior value untouched.

        ``mean_logprob`` / ``mean_surprise`` are the per-turn rollups
        computed in :meth:`DrowseSession._generate_core` from the engine's
        chosen-token logprob stream (response span only — thinking tokens
        are excluded by construction).  ``None`` when logprob capture
        wasn't live (no on_token consumer + no logprobs request), which
        also covers replay-from-legacy-transcripts.

        ``raw_token_ids`` is the engine's exact ``generated_ids`` decode
        sequence (delimiters included) — the forceable prefix a logit
        fork replays.  ``None`` leaves the node's prior value untouched.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            node.text = text
            if aggregate_readings is not None:
                node.aggregate_readings = dict(aggregate_readings)
            node.applied_steering = applied_steering
            node.finish_reason = finish_reason
            node.mean_logprob = mean_logprob
            node.mean_surprise = mean_surprise
            if raw_token_ids is not None:
                node.raw_token_ids = list(raw_token_ids)
            if thinking_text is not None:
                node.thinking_text = thinking_text
            self.rev += 1
            self._emit(LoomMutated(
                op="finalize_assistant", rev=self.rev,
                updated=(node.id,),
            ))

    # ------------------------------------------------------------------
    # Core primitives — edit, branch, navigate, delete_subtree
    # ------------------------------------------------------------------

    def edit(self, node_id: str, text: str) -> None:
        """In-place text replacement.

        No new node, no tree-shape change.  Bumps ``edit_count`` and
        sets ``edited_at`` so downstream consumers can flag the node
        as no-longer-pristine.  Refused when ``node_id`` is in the
        reservation of an in-flight generation.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if node_id == self.root_id:
                raise InvalidNodeOperationError("cannot edit the root node")
            self._conflict_check(node_id, "edit")
            node.text = text
            node.edit_count += 1
            node.edited_at = time.time()
            self.rev += 1
            self._emit(LoomMutated(
                op="edit", rev=self.rev, updated=(node_id,),
            ))

    def append_authored(
        self,
        node_id: str,
        text: str,
        *,
        thinking_text: str | None = None,
        raw_token_ids: list[int] | None = None,
    ) -> str:
        """Append authored text to an existing same-role message.

        A node-level generation receipt cannot faithfully describe a message
        after an authored tail is added, so generated payloads are
        cleared and the node becomes an authored message.  The tree shape is
        unchanged and the existing node stays active.
        """
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if node_id == self.root_id:
                raise InvalidNodeOperationError(
                    "cannot append to the synthetic root node"
                )
            self._conflict_check(node_id, "edit")
            node.text += text
            if thinking_text is not None:
                node.thinking_text = (node.thinking_text or "") + thinking_text
            node.tokens = None
            node.thinking_tokens = None
            node.recipe = None
            node.aggregate_readings = {}
            node.applied_steering = None
            node.finish_reason = "stop"
            node.mean_logprob = None
            node.mean_surprise = None
            node.raw_token_ids = (
                list(raw_token_ids) if raw_token_ids is not None else None
            )
            node.edit_count += 1
            node.edited_at = time.time()
            self.active_node_id = node_id
            self.rev += 1
            self._emit(LoomMutated(
                op="edit", rev=self.rev, updated=(node_id,),
                active_node_id=node_id,
            ))
            return node_id

    def branch(
        self,
        node_id: str,
        text: str,
        *,
        role: Role | None = None,
        make_active: bool = True,
    ) -> str:
        """Create a new sibling of ``node_id`` with the given text.

        Always-sibling: the original is preserved.  Empty text is the
        "branch from blank" UI flavor; pre-fill text is "fork-and-edit".
        Defaults to the same role as the sibling; pass ``role=`` to
        override.  Returns the new node id.  Allowed during in-flight
        generation (creating a sibling doesn't disturb the streaming
        target).
        """
        with self._lock:
            sibling = self.nodes.get(node_id)
            if sibling is None:
                raise UnknownNodeError(node_id)
            if sibling.parent_id is None:
                raise InvalidNodeOperationError(
                    "cannot branch from the root — branch off its children instead"
                )
            new = LoomNode(
                id=_ulid(),
                parent_id=sibling.parent_id,
                role=role if role is not None else sibling.role,
                text=text,
                role_label=sibling.role_label,
                thinking_text=(
                    sibling.thinking_text if text == sibling.text else None
                ),
            )
            self._add_child(sibling.parent_id, new)
            if make_active:
                self.active_node_id = new.id
            self.rev += 1
            self._emit(LoomMutated(
                op="branch", rev=self.rev,
                added=(new.id,),
                active_node_id=self.active_node_id if make_active else None,
            ))
            return new.id

    def graft(
        self,
        parent_id: str,
        chain: Sequence[LoomNode],
        *,
        cast: Mapping[str, CastMember] | None = None,
        note: str | None = None,
    ) -> str:
        """Attach a prepared chain of turns under ``parent_id`` in one mutation.

        ``chain`` lists the new turns root-to-leaf as :class:`LoomNode`
        templates; their ``id`` and ``parent_id`` are ignored — the graft
        assigns fresh ids and links each node under the previous one.
        ``cast`` members land for labels the roster does not already hold.
        ``note`` is stamped on the first grafted node (or on ``parent_id``
        when the chain is empty).

        Everything is validated before the first write, so a bad turn leaves
        the tree untouched; the whole graft is one ``rev`` and one ``import``
        event (``added`` = the chain, ``active_node_id`` = its leaf).  Like
        :meth:`branch`, grafting new children never disturbs an in-flight
        generation.  Returns the chain's leaf id (``parent_id`` when empty).
        """

        for node in chain:
            if node.role not in ("user", "assistant"):
                raise InvalidNodeOperationError(
                    f"graft: turns occupy the user or assistant seat, got {node.role!r}"
                )
        try:
            for label in [n.role_label for n in chain if n.role_label is not None]:
                validate_role(label)
            for label in cast or {}:
                validate_role(label)
        except InvalidRoleError as error:
            raise InvalidNodeOperationError(f"graft: {error}") from error
        with self._lock:
            if parent_id not in self.nodes:
                raise UnknownNodeError(parent_id)
            added: list[str] = []
            current = parent_id
            for template in chain:
                node = replace(template, id=_ulid(), parent_id=current)
                self._add_child(current, node)
                added.append(node.id)
                current = node.id
            for label, member in (cast or {}).items():
                self.cast.setdefault(label, member)
            updated: tuple[str, ...] = ()
            if note:
                target = added[0] if added else parent_id
                self.nodes[target].notes = note
                if not added:
                    updated = (parent_id,)
            if added:
                self.active_node_id = current
            self.rev += 1
            self._emit(LoomMutated(
                op="import", rev=self.rev,
                added=tuple(added), updated=updated,
                active_node_id=current if added else None,
            ))
            return current

    def navigate(self, node_id: str) -> None:
        """Re-point :attr:`active_node_id` to ``node_id``.

        Always free relative to in-flight generation; the gen continues
        attached to its original target invisibly.  Emits a ``navigate``
        mutation event so surfaces re-render.
        """
        with self._lock:
            if node_id not in self.nodes:
                raise UnknownNodeError(node_id)
            if node_id == self.active_node_id:
                return
            self.active_node_id = node_id
            self.rev += 1
            self._emit(LoomMutated(
                op="navigate", rev=self.rev,
                active_node_id=node_id,
            ))

    def delete_subtree(self, node_id: str) -> int:
        """Drop the subtree rooted at ``node_id``.

        When the active node sits inside the deleted subtree (including
        the case ``node_id == active_node_id``), the active pointer is
        re-seated on the surviving parent — the nearest ancestor that
        isn't being removed.  If that parent is the root, the user
        lands on the empty root, which is the natural "fresh start"
        state.  Refuses the root delete outright; refuses (via the
        conflict checker) when the subtree intersects an in-flight
        generation's reservation.  Returns the count of nodes removed.
        """
        with self._lock:
            if node_id == self.root_id:
                raise InvalidNodeOperationError("cannot delete the root")
            if node_id not in self.nodes:
                raise UnknownNodeError(node_id)
            self._conflict_check(node_id, "delete_subtree")

            # Collect every node in the subtree (DFS) plus the root.
            to_remove: list[str] = [node_id]
            stack = list(self.children_of.get(node_id, []))
            while stack:
                cur = stack.pop()
                to_remove.append(cur)
                stack.extend(self.children_of.get(cur, []))

            parent_id = self.nodes[node_id].parent_id
            # The subtree root is non-root, so it has a parent — that
            # parent is the survivor the active pointer falls back to
            # when active is inside the doomed set.
            removed_set = set(to_remove)
            active_moved_to: str | None = None
            if self.active_node_id in removed_set:
                # parent_id is non-None because node_id != root_id.
                self.active_node_id = parent_id  # pyright: ignore[reportAttributeAccessIssue]  # non-None: node_id != root_id guard above ensures it
                active_moved_to = parent_id

            if parent_id is not None:
                self.children_of[parent_id] = [
                    c for c in self.children_of.get(parent_id, []) if c != node_id
                ]
            for nid in to_remove:
                self.nodes.pop(nid, None)
                self.children_of.pop(nid, None)

            self.rev += 1
            self._emit(LoomMutated(
                op="delete", rev=self.rev,
                removed=tuple(to_remove),
                active_node_id=active_moved_to,
            ))
            return len(to_remove)

    # ------------------------------------------------------------------
    # Decoration ops — always free
    # ------------------------------------------------------------------

    def star(self, node_id: str, on: bool = True) -> None:
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            if node.starred == on:
                return
            node.starred = on
            self.rev += 1
            self._emit(LoomMutated(
                op="star", rev=self.rev, updated=(node_id,),
            ))

    def annotate(self, node_id: str, notes: str) -> None:
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                raise UnknownNodeError(node_id)
            node.notes = notes
            self.rev += 1
            self._emit(LoomMutated(
                op="note", rev=self.rev, updated=(node_id,),
            ))

    # ------------------------------------------------------------------
    # Cast roster (phase 3)
    # ------------------------------------------------------------------

    def set_cast_member(self, label: str, member: CastMember) -> None:
        """Create or replace the cast member under ``label``.

        Roster ops are decoration-tier — always free under in-flight
        generation (they touch no node).  The label must be a legal
        role slug (it is rendered into turn headers); validation mirrors
        the per-turn ``role_label`` rules.
        """

        validate_role(label)
        with self._lock:
            if self.cast.get(label) == member:
                return
            self.cast[label] = member
            self.rev += 1
            self._emit(LoomMutated(op="cast", rev=self.rev))

    def remove_cast_member(self, label: str) -> None:
        """Drop the cast member under ``label`` (no-op when absent)."""
        with self._lock:
            if label not in self.cast:
                return
            del self.cast[label]
            self.rev += 1
            self._emit(LoomMutated(op="cast", rev=self.rev))

    # ------------------------------------------------------------------
    # Engine-level workflows: clear (reset), rewind
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop the entire tree.  Refused under in-flight generation."""
        with self._lock:
            self._conflict_check(self.root_id, "reset")
            # Every current id is removed — including the *old* root, which the
            # fresh ``_ulid()`` root below replaces.  Reporting the old root in
            # ``removed`` keeps the delta self-consistent for id-tracking
            # surfaces (the WS dashboard's ``applyTreeDelta``), which otherwise
            # retain a stale orphan root.
            removed = tuple(self.nodes)
            # Wipe everything; rebuild a fresh root.
            self.nodes.clear()
            self.children_of.clear()
            root = LoomNode(id=_ulid(), parent_id=None, role="system")
            self.nodes[root.id] = root
            self.children_of[root.id] = []
            self.root_id = root.id
            self.active_node_id = root.id
            self.rev += 1
            self._emit(LoomMutated(
                op="reset", rev=self.rev,
                removed=removed,
                added=(root.id,),
                active_node_id=root.id,
            ))

    def rewind(self) -> None:
        """Navigate the active node one user→assistant pair up the path.

        Non-destructive: the rewound pair stays in the tree as a dead
        branch, navigable back to.  No-op when the active node is the
        root or its direct child.
        """
        with self._lock:
            path = self.active_path()
            if len(path) < 2:
                return  # nothing meaningful to rewind from
            # Walk back one whole submission when the active node carries a
            # generation receipt, otherwise one committed turn. Structural
            # role is independent of provenance: swapped assistant→user
            # submissions rewind exactly like user→assistant ones.
            anchor = path[-1]
            steps = 2 if anchor.recipe is not None else 1
            target_idx = max(0, len(path) - 1 - steps)
            target = path[target_idx]
            if target.id == self.active_node_id:
                return
            self.active_node_id = target.id
            self.rev += 1
            self._emit(LoomMutated(
                op="navigate", rev=self.rev,
                active_node_id=target.id,
            ))

    # ------------------------------------------------------------------
    # Predicate ops
    # ------------------------------------------------------------------

    def filter(self, pred: Callable[[LoomNode], bool]) -> set[str]:
        """Return the set of node ids whose nodes satisfy ``pred``."""
        with self._lock:
            return {nid for nid, node in self.nodes.items() if pred(node)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self, *, include_tokens: bool = False) -> dict[str, Any]:
        with self._lock:
            # ``drowse_version`` rides alongside ``tree_format`` so future
            # migrations can branch on the originating build even when the
            # schema number hasn't moved — same pattern packs use.  Imported
            # lazily so a circular at module-load time stays impossible.
            from drowse import __version__ as _drowse_version
            out: dict[str, Any] = {
                "tree_format": TREE_FORMAT_VERSION,
                "drowse_version": _drowse_version,
                "model_id": self.model_id,
                "session_id": self.session_id,
                "name": self.name,
                "rev": self.rev,
                "root_id": self.root_id,
                "active_node_id": self.active_node_id,
                "nodes": [
                    self.nodes[nid].to_dict(include_tokens=include_tokens)
                    for nid in self.nodes
                ],
                "children_of": {k: list(v) for k, v in self.children_of.items()},
            }
            out["cast"] = {
                label: member.to_dict() for label, member in self.cast.items()
            }
            return out

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        events: EventBus | None = None,
        conflict_check: ConflictChecker | None = None,
    ) -> "LoomTree":
        require_fields(
            data, _TREE_FIELDS, "loom tree", optional=frozenset({"token_sidecar"})
        )
        version = data["tree_format"]
        if version != TREE_FORMAT_VERSION:
            raise LoomTreeError(
                f"unsupported tree_format {version!r} "
                f"(this build supports {TREE_FORMAT_VERSION})"
            )
        tree = cls.__new__(cls)
        tree._lock = threading.RLock()
        tree._events = events
        tree._conflict_check = conflict_check or _noop_conflict_check
        if not isinstance(data["drowse_version"], str):
            raise LoomTreeError("loom tree field 'drowse_version' must be a string")
        for field_name in ("model_id", "session_id", "name"):
            value = data[field_name]
            if value is not None and not isinstance(value, str):
                raise LoomTreeError(
                    f"loom tree field {field_name!r} must be a string or null"
                )
        tree.model_id = data["model_id"]
        tree.session_id = data["session_id"]
        tree.name = data["name"]
        if (not isinstance(data["rev"], int) or isinstance(data["rev"], bool)
                or data["rev"] < 0):
            raise LoomTreeError("loom tree field 'rev' must be non-negative")
        tree.rev = data["rev"]
        raw_nodes = data["nodes"]
        raw_children = data["children_of"]
        raw_cast = data["cast"]
        if not isinstance(raw_nodes, list):
            raise LoomTreeError("loom tree field 'nodes' must be a list")
        if not isinstance(raw_children, dict):
            raise LoomTreeError("loom tree field 'children_of' must be an object")
        if not isinstance(raw_cast, dict):
            raise LoomTreeError("loom tree field 'cast' must be an object")
        tree.nodes = {}
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                raise LoomTreeError("loom tree nodes must be objects")
            node = LoomNode.from_dict(raw)
            if node.id in tree.nodes:
                raise LoomTreeError(f"duplicate loom node id {node.id!r}")
            tree.nodes[node.id] = node
        if any(
            not isinstance(k, str)
            or not isinstance(v, list)
            or any(not isinstance(child, str) for child in v)
            for k, v in raw_children.items()
        ):
            raise LoomTreeError("children_of must map node ids to lists")
        tree.children_of = {k: list(v) for k, v in raw_children.items()}
        if not isinstance(data["root_id"], str) or not isinstance(data["active_node_id"], str):
            raise LoomTreeError("root_id and active_node_id must be strings")
        tree.root_id = data["root_id"]
        tree.active_node_id = data["active_node_id"]
        tree.cast = {}
        for label, raw in raw_cast.items():
            if not isinstance(label, str) or not isinstance(raw, dict):
                raise LoomTreeError("cast must map string labels to member objects")
            origin = raw.get("origin")
            if origin is not None and origin not in (
                "structural", "observed", "configured",
            ):
                raise LoomTreeError(f"invalid cast member origin {origin!r}")
            if origin in ("structural", "observed"):
                continue
            validate_role(label)
            tree.cast[label] = CastMember.from_dict(raw)
        tree._validate_structure()
        return tree

    def _validate_structure(self) -> None:
        node_ids = set(self.nodes)
        if set(self.children_of) != node_ids:
            raise LoomTreeError("children_of keys must exactly match loom node ids")
        if self.root_id not in node_ids:
            raise LoomTreeError("root_id does not identify a loom node")
        if self.active_node_id not in node_ids:
            raise LoomTreeError("active_node_id does not identify a loom node")
        root = self.nodes[self.root_id]
        if root.parent_id is not None or root.role != "system":
            raise LoomTreeError("loom root must be a parentless system node")

        seen_children: set[str] = set()
        for parent_id, children in self.children_of.items():
            if len(children) != len(set(children)):
                raise LoomTreeError(f"children_of[{parent_id!r}] contains duplicates")
            for child_id in children:
                if child_id not in node_ids:
                    raise LoomTreeError(f"children_of references unknown node {child_id!r}")
                if child_id in seen_children:
                    raise LoomTreeError(f"loom node {child_id!r} has multiple parents")
                if self.nodes[child_id].parent_id != parent_id:
                    raise LoomTreeError(f"loom node {child_id!r} disagrees with children_of")
                seen_children.add(child_id)
        if seen_children != node_ids - {self.root_id}:
            raise LoomTreeError("every non-root loom node must appear under its parent")
        reachable: set[str] = set()
        stack = [self.root_id]
        while stack:
            node_id = stack.pop()
            if node_id in reachable:
                raise LoomTreeError("loom tree contains a cycle")
            reachable.add(node_id)
            stack.extend(self.children_of[node_id])
        if reachable != node_ids:
            raise LoomTreeError("all loom nodes must be reachable from root_id")

    def save(self, path: Any) -> None:
        """Atomic write of the tree to ``path`` as JSON.

        The main tree file omits token blobs; response/thinking token rows
        are written to a gzip sidecar named ``<stem>.tokens.json.gz``.  This
        keeps routine tree payloads small while explicit save/load preserves
        logprob/top-alt/probe drilldown data.
        """
        from pathlib import Path
        from drowse.base.atomic import write_json_atomic

        out_path = Path(path)
        sidecar = out_path.with_name(f"{out_path.stem}.tokens.json.gz")
        with self._lock:
            data = self.to_dict(include_tokens=False)
            token_nodes: dict[str, dict[str, Any]] = {}
            for node in self.nodes.values():
                if node.tokens or node.thinking_tokens or node.raw_token_ids is not None:
                    token_nodes[node.id] = {
                        "tokens": node.tokens,
                        "thinking_tokens": node.thinking_tokens,
                        "raw_token_ids": node.raw_token_ids,
                    }
            token_payload = None
            if token_nodes:
                data["token_sidecar"] = sidecar.name
                token_payload = {
                    "token_sidecar_format": TOKEN_SIDECAR_FORMAT_VERSION,
                    "nodes": token_nodes,
                }

        if token_payload is None:
            write_json_atomic(out_path, data)
            with suppress(FileNotFoundError):
                sidecar.unlink()
            return

        sidecar.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=str(sidecar.parent),
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(token_payload, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, sidecar)
        finally:
            with suppress(FileNotFoundError):
                tmp.unlink()
        write_json_atomic(out_path, data)

    @classmethod
    def load(
        cls, path: Any, *, events: EventBus | None = None,
        max_bytes: int | None = None,
    ) -> "LoomTree":
        """Load a saved tree with bounded JSON and gzip expansion.

        Defaults to 64 MiB for the tree and 256 MiB for its expanded token
        sidecar. ``max_bytes`` overrides both limits for trusted large exports.
        """
        from pathlib import Path

        if max_bytes is not None and (
            type(max_bytes) is not int or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        in_path = Path(path)
        with open(in_path, "rb") as f:
            data = _read_json_bounded(
                f, _TREE_MAX_LOAD_BYTES if max_bytes is None else max_bytes, "loom tree",
            )
        if not isinstance(data, dict):
            raise LoomTreeError("loom tree file must contain an object")
        require_fields(
            data, _TREE_FIELDS, "loom tree", optional=frozenset({"token_sidecar"})
        )
        if data["tree_format"] != TREE_FORMAT_VERSION:
            raise LoomTreeError(
                f"unsupported tree_format {data['tree_format']!r} "
                f"(this build supports {TREE_FORMAT_VERSION})"
            )
        sidecar_name = data.get("token_sidecar")
        if sidecar_name is not None:
            expected_sidecar_name = f"{in_path.stem}.tokens.json.gz"
            if sidecar_name != expected_sidecar_name:
                raise LoomTreeError(
                    f"token_sidecar must be {expected_sidecar_name!r}, "
                    f"got {sidecar_name!r}"
                )
            sidecar = in_path.parent / sidecar_name
            try:
                with gzip.GzipFile(sidecar, "rb") as f:
                    token_payload = _read_json_bounded(
                        f, _TOKEN_SIDECAR_MAX_LOAD_BYTES if max_bytes is None else max_bytes,
                        "token sidecar",
                    )
            except FileNotFoundError as e:
                raise LoomTreeError(
                    f"token sidecar declared but missing: {sidecar}"
                ) from e
            except (gzip.BadGzipFile, EOFError, zlib.error) as error:
                raise LoomTreeError("token sidecar is not a complete gzip file") from error
            if not isinstance(token_payload, dict):
                raise LoomTreeError("token sidecar must contain an object")
            require_fields(token_payload, frozenset({"token_sidecar_format", "nodes"}), "token sidecar")
            if token_payload["token_sidecar_format"] != TOKEN_SIDECAR_FORMAT_VERSION:
                raise LoomTreeError(
                    f"unsupported token_sidecar_format "
                    f"{token_payload['token_sidecar_format']!r}"
                )
            by_id = token_payload["nodes"]
            if not isinstance(by_id, dict):
                raise LoomTreeError("token sidecar field 'nodes' must be an object")
            main_by_id = {raw["id"]: raw for raw in data["nodes"]}
            for node_id, node_tokens in by_id.items():
                if node_id not in main_by_id:
                    raise LoomTreeError(f"token sidecar references unknown node {node_id!r}")
                if not isinstance(node_tokens, dict):
                    raise LoomTreeError(f"token sidecar node {node_id!r} must be an object")
                require_fields(node_tokens, _TOKEN_FIELDS, f"token sidecar node {node_id!r}")
                for field_name in ("tokens", "thinking_tokens", "raw_token_ids"):
                    value = node_tokens[field_name]
                    if value is not None and not isinstance(value, list):
                        raise LoomTreeError(
                            f"token sidecar node {node_id!r} field {field_name!r} "
                            "must be a list or null"
                        )
                    main_by_id[node_id][field_name] = value
        return cls.from_dict(data, events=events)


__all__ = [
    "LoomNode",
    "LoomTree",
    "TREE_FORMAT_VERSION",
]
