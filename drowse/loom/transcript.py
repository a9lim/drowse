"""Transcript export / import / replay for loom sessions.

A **transcript** is a saved path through the loom tree: system prompt
+ every turn + each generated turn's :class:`Recipe` (steering, sampling
with its seed, thinking, probe set, per-probe content hash) + final
aggregate readings.  Serializes to YAML; round-trips through
:meth:`Transcript.to_yaml` / :meth:`Transcript.from_yaml`.

The per-node thing remains :class:`Recipe`; the file/export concept is
:class:`Transcript` so docs and CLI do not overload the two artifacts.

Schema (v2 — the cast model)::

    drowse_transcript: 2
    model_id: <hf-id>            # optional
    system_prompt: <str>         # optional
    cast:                        # optional; the tree's cast roster
      <label>: {recipe: {...}, notes: <str>}
    probes:                      # optional
      - name: <probe>
        sha256: <hex>
    turns:
      - role: user|assistant|system   # the seat (system turns are skipped on import)
        speaker: <label>         # optional; the turn's cast label
        text: <str>
        thinking: <str>          # optional; the turn's thinking block
        recipe: {...}            # generated turns (either seat); provenance
        readings: {...}          # generated turns

The reader is strict like the tree's: a missing ``role``/``text``, an unknown
key, a speaker or cast label that is not a role slug, or a reading that is
not a finite number raises :class:`TranscriptFormatError`.

On import, a turn with a ``recipe`` re-attaches as a *generated* node in its
recorded seat (provenance = recipe presence, the cast model's invariant); a
recipe-less turn is a committed one.  Cast entries merge into the session
tree's roster; a label the live roster already holds with a *different*
member is left alone and flagged in the guard notes.  The branch and the
cast additions land in one :meth:`LoomTree.graft` — one revision, one
``import`` event, and nothing written when any part is invalid.

Three import modes:

- **default** — attaches as a new top-level branch off the tree root.
- **here** — attaches as a child of the active node.
- **merge** — walks the active path from root, finds the deepest
  matching **user-turn** prefix between the active path and the
  transcript, attaches the non-matching tail there.  Falls back to
  root-attach when no user-turn prefix matches.

Guards:

- model mismatch → warn, refuse ``--merge``
  (:class:`TranscriptModelMismatch`), allow other modes with a banner
  on the imported root.
- system-prompt mismatch → warn, proceed, banner on imported root.
- missing probes → warn, readings recorded as-imported (display-only).
- probe hash drift → warn; ``--strict`` raises
  :class:`TranscriptProbeDriftError` on any hash mismatch.

:meth:`Transcript.replay` imports the transcript and regenerates each
generated turn under its recipe, as a sibling under its original parent
(so the model sees the transcript's full history), then diffs each
regeneration against the recorded turn.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from drowse.errors import DrowseError
from drowse.loom.diff import NodeDiff, node_diff
from drowse.loom.errors import LoomTreeError
from drowse.loom.recipe import CastMember, Recipe
from drowse.loom.tree import LoomNode, LoomTree
from drowse.model.role_templates import InvalidRoleError, validate_role


DROWSE_TRANSCRIPT_VERSION = 2

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TranscriptError(DrowseError):
    """Base for transcript-IO errors."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class TranscriptFormatError(TranscriptError):
    """Raised when a transcript YAML can't be parsed / lacks required fields."""


class TranscriptModelMismatch(TranscriptError):
    """Raised when a transcript's model differs from the session and ``--merge`` is requested."""


class TranscriptProbeDriftError(TranscriptError):
    """Raised under ``strict=True`` when any probe sha256 differs from the session."""


ImportMode = Literal["default", "here", "merge"]

_ROOT_REQUIRED = frozenset({"drowse_transcript", "turns"})
_ROOT_OPTIONAL = frozenset({"model_id", "system_prompt", "cast", "probes"})
_TURN_REQUIRED = frozenset({"role", "text"})
_TURN_OPTIONAL = frozenset({"speaker", "thinking", "recipe", "readings"})
_PROBE_FIELDS = frozenset({"name", "sha256"})


def _check_keys(
    data: Mapping[Any, Any], required: frozenset[str], optional: frozenset[str], label: str,
) -> None:
    missing = sorted(required - data.keys())
    if missing:
        raise TranscriptFormatError(f"{label} is missing: {', '.join(missing)}")
    unknown = sorted(str(k) for k in data.keys() - required - optional)
    if unknown:
        raise TranscriptFormatError(f"{label} has unknown keys: {', '.join(unknown)}")


def _optional_str(data: Mapping[str, Any], key: str, label: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise TranscriptFormatError(f"{label}.{key} must be a string")
    return value


def _role_slug(label: str, where: str) -> str:
    try:
        validate_role(label)
    except InvalidRoleError as error:
        raise TranscriptFormatError(f"{where}: {error}") from error
    return label


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeRef:
    """Probe entry in the transcript header — name + hash for drift detection."""

    name: str
    sha256: str


@dataclass
class Turn:
    """One conversation turn captured for replay.

    ``role`` is the *seat*; ``speaker`` the cast label the turn was
    rendered with (``None`` = the family's standard label).  Provenance
    is recipe presence — a generated turn carries its :class:`Recipe`
    whatever its seat.
    """

    role: Literal["user", "assistant", "system"]
    text: str
    speaker: str | None = None
    thinking: str | None = None
    recipe: Recipe | None = None
    readings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "text": self.text}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        if self.thinking is not None:
            out["thinking"] = self.thinking
        if self.recipe is not None:
            out["recipe"] = self.recipe.to_dict()
        if self.readings:
            out["readings"] = dict(self.readings)
        return out

    @classmethod
    def from_dict(cls, data: Any, label: str = "turn") -> "Turn":
        if not isinstance(data, Mapping):
            raise TranscriptFormatError(f"{label} must be a mapping")
        _check_keys(data, _TURN_REQUIRED, _TURN_OPTIONAL, label)
        role = data["role"]
        if role not in ("user", "assistant", "system"):
            raise TranscriptFormatError(f"{label}.role must be user, assistant or system")
        text = data["text"]
        if not isinstance(text, str):
            raise TranscriptFormatError(f"{label}.text must be a string")
        speaker = _optional_str(data, "speaker", label)
        if speaker is not None:
            _role_slug(speaker, f"{label}.speaker")
        recipe = None
        if data.get("recipe") is not None:
            if not isinstance(data["recipe"], Mapping):
                raise TranscriptFormatError(f"{label}.recipe must be a mapping")
            try:
                recipe = Recipe.from_dict(dict(data["recipe"]))
            except LoomTreeError as error:
                raise TranscriptFormatError(f"{label}.recipe: {error}") from error
        raw_readings = data.get("readings") or {}
        if not isinstance(raw_readings, Mapping):
            raise TranscriptFormatError(f"{label}.readings must be a mapping")
        readings: dict[str, float] = {}
        for name, value in raw_readings.items():
            if (
                not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise TranscriptFormatError(
                    f"{label}.readings must map probe names to finite numbers"
                )
            readings[name] = float(value)
        return cls(
            role=role,
            text=text,
            speaker=speaker,
            thinking=_optional_str(data, "thinking", label),
            recipe=recipe,
            readings=readings,
        )


@dataclass(frozen=True)
class ReplayStep:
    """One regenerated turn: the recorded node, its replay, and their diff.

    ``diff`` is ``recorded → replayed`` (:func:`drowse.loom.diff.node_diff`
    with ``a`` the recorded node), so a reading delta is ``replay − recorded``.
    """

    recorded_id: str
    replay_id: str
    diff: NodeDiff


@dataclass(frozen=True)
class TranscriptReplay:
    """The imported branch's leaf plus one :class:`ReplayStep` per generated turn."""

    leaf_id: str
    steps: list[ReplayStep]


@dataclass
class Transcript:
    """A saved path through the tree.

    Build from a tree node via :meth:`from_path`; round-trip through
    :meth:`to_yaml` / :meth:`from_yaml`; import into a live session
    via :meth:`import_into`; regenerate it via :meth:`replay`.
    """

    model_id: str | None
    system_prompt: str | None
    probes: list[ProbeRef]
    turns: list[Turn]
    # Cast roster (v2): label → member, mirrored from the tree at export.
    cast: dict[str, CastMember] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction from a session path
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, node_id: str | None, session: Any) -> "Transcript":
        """Build a transcript from the path ending at ``node_id``.

        ``node_id=None`` uses the session's active node.  Skips the
        synthetic root; carries the session's system prompt as the
        top-level ``system_prompt`` field rather than a ``role: system``
        turn.
        """
        target = node_id if node_id is not None else session.tree.active_node_id
        turns = [
            Turn(
                role=node.role,
                text=node.text or "",
                speaker=node.role_label,
                thinking=node.thinking_text,
                recipe=node.recipe,
                readings=dict(node.aggregate_readings or {}),
            )
            for node in session.tree.path_to(target)
            if node.id != session.tree.root_id
        ]
        probes = [
            ProbeRef(name=name, sha256=digest)
            for name, digest in session.probe_hashes().items()
        ]
        return cls(
            model_id=getattr(session, "model_id", None)
                or getattr(session.tree, "model_id", None),
            system_prompt=getattr(session.config, "system_prompt", None),
            probes=probes,
            turns=turns,
            cast=dict(getattr(session.tree, "cast", {}) or {}),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "drowse_transcript": DROWSE_TRANSCRIPT_VERSION,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
        }
        if self.cast:
            out["cast"] = {
                label: m.to_dict() for label, m in self.cast.items()
            }
        out["probes"] = [
            {"name": p.name, "sha256": p.sha256} for p in self.probes
        ]
        out["turns"] = [t.to_dict() for t in self.turns]
        return out

    def to_yaml(self) -> str:
        """Render to YAML (pyyaml is a hard drowse dependency)."""
        import yaml

        return yaml.safe_dump(
            self.to_dict(), sort_keys=False, default_flow_style=False,
        )

    @classmethod
    def from_dict(cls, data: Any) -> "Transcript":
        """Read the transcript document — strict, see the module docstring."""
        if not isinstance(data, Mapping):
            raise TranscriptFormatError(
                f"transcript root must be a mapping, got {type(data).__name__}"
            )
        version = data.get("drowse_transcript")
        if version != DROWSE_TRANSCRIPT_VERSION:
            raise TranscriptFormatError(
                f"unsupported drowse_transcript version {version!r} "
                f"(this build requires {DROWSE_TRANSCRIPT_VERSION})"
            )
        _check_keys(data, _ROOT_REQUIRED, _ROOT_OPTIONAL, "transcript")

        raw_probes = data.get("probes") or []
        if not isinstance(raw_probes, list):
            raise TranscriptFormatError("transcript probes must be a list")
        probes: list[ProbeRef] = []
        for index, raw in enumerate(raw_probes):
            label = f"probes[{index}]"
            if not isinstance(raw, Mapping):
                raise TranscriptFormatError(f"{label} must be a mapping")
            _check_keys(raw, _PROBE_FIELDS, frozenset(), label)
            if not isinstance(raw["name"], str) or not isinstance(raw["sha256"], str):
                raise TranscriptFormatError(f"{label} name and sha256 must be strings")
            probes.append(ProbeRef(name=raw["name"], sha256=raw["sha256"]))

        raw_turns = data["turns"]
        if not isinstance(raw_turns, list):
            raise TranscriptFormatError("transcript turns must be a list")
        turns = [
            Turn.from_dict(raw, f"turns[{index}]")
            for index, raw in enumerate(raw_turns)
        ]

        raw_cast = data.get("cast") or {}
        if not isinstance(raw_cast, Mapping):
            raise TranscriptFormatError("transcript cast must be a mapping")
        cast: dict[str, CastMember] = {}
        for label, raw in raw_cast.items():
            if not isinstance(label, str):
                raise TranscriptFormatError("transcript cast labels must be strings")
            _role_slug(label, f"cast.{label}")
            if not isinstance(raw, Mapping):
                raise TranscriptFormatError(f"cast.{label} must be a mapping")
            if "origin" in raw:
                raise TranscriptFormatError(f"cast.{label} has unknown keys: origin")
            try:
                cast[label] = CastMember.from_dict(dict(raw))
            except LoomTreeError as error:
                raise TranscriptFormatError(f"cast.{label}: {error}") from error

        return cls(
            model_id=_optional_str(data, "model_id", "transcript"),
            system_prompt=_optional_str(data, "system_prompt", "transcript"),
            probes=probes,
            turns=turns,
            cast=cast,
        )

    @classmethod
    def from_yaml(cls, text: str) -> "Transcript":
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise TranscriptFormatError(f"yaml parse error: {e}") from e
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Atomic-write the transcript to ``path``."""
        from drowse.base.atomic import write_bytes_atomic
        write_bytes_atomic(Path(path), self.to_yaml().encode("utf-8"))

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        with open(Path(path), "r", encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    # ------------------------------------------------------------------
    # Import into a live session
    # ------------------------------------------------------------------

    def import_into(
        self,
        session: Any,
        *,
        mode: ImportMode = "default",
        strict: bool = False,
    ) -> str:
        """Attach this transcript to ``session.tree`` under ``mode``.

        Returns the imported branch's leaf node id.  The branch's first
        node carries any guard notes (model / system-prompt / probe / cast
        mismatches) on its ``notes`` field so the surfaces can display a
        banner.  The import is one :meth:`LoomTree.graft`: either the whole
        branch and its cast additions land, or nothing does.

        Raises :class:`TranscriptModelMismatch` on model mismatch under
        ``mode="merge"`` and :class:`TranscriptProbeDriftError` on any
        probe hash difference when ``strict=True``.
        """
        return self._import(session, mode=mode, strict=strict)[1]

    def _import(
        self, session: Any, *, mode: ImportMode, strict: bool,
    ) -> tuple[str, str]:
        """Import and return ``(attach_parent, leaf_id)``."""
        tree: LoomTree = session.tree
        guard_notes = self._collect_guard_notes(session, mode=mode, strict=strict)

        # The session's standing roster wins a label conflict; absent labels
        # land with the branch.
        cast_conflicts = sorted(
            label for label, member in self.cast.items()
            if label in tree.cast and tree.cast[label] != member
        )
        if cast_conflicts:
            msg = (
                f"transcript cast differs from the session roster for: "
                f"{', '.join(cast_conflicts)}; keeping the session's members"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            guard_notes.append(f"cast_conflict: {cast_conflicts}")
        additions = {
            label: member for label, member in self.cast.items()
            if label not in tree.cast
        }

        attach_parent = self._resolve_attach_parent(session, mode=mode)
        chain = self._branch_chain(tree, attach_parent)
        leaf_id = tree.graft(
            attach_parent, chain, cast=additions,
            note="\n".join(guard_notes) or None,
        )
        return attach_parent, leaf_id

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        session: Any,
        *,
        mode: ImportMode = "default",
        strict: bool = False,
    ) -> TranscriptReplay:
        """Import, then regenerate every generated turn under its recipe.

        Each generated node of the imported branch is regenerated as a
        sibling under its own parent — the model sees the transcript's
        full history up to that turn, not the turn alone — with the
        recorded steering (``""`` when the turn was unsteered, so no cast
        default creeps in), sampling (seed included), thinking mode and
        system prompt, in the recorded seat.  Each step diffs the recorded
        node against its regeneration (text and aggregate readings).
        """
        attach_parent, leaf_id = self._import(session, mode=mode, strict=strict)
        tree: LoomTree = session.tree
        path = tree.path_to(leaf_id)
        start = next(i for i, node in enumerate(path) if node.id == attach_parent) + 1
        steps: list[ReplayStep] = []
        for node in path[start:]:
            recipe = node.recipe
            if recipe is None or node.parent_id is None:
                continue
            run = session.generate(
                None,
                steering=recipe.steering if recipe.steering is not None else "",
                sampling=recipe.sampling,
                thinking=recipe.thinking,
                parent_node_id=node.parent_id,
                gen_seat=node.role,
                append_same_role=False,
                system_prompt=recipe.system_prompt,
            )
            replay_id = run.node_ids[0]
            if replay_id is None:  # pragma: no cover - stateful generate always lands a node
                raise TranscriptError(f"replay of {node.id} produced no loom node")
            steps.append(ReplayStep(
                recorded_id=node.id,
                replay_id=replay_id,
                diff=node_diff(tree, node.id, replay_id),
            ))
        return TranscriptReplay(leaf_id=leaf_id, steps=steps)

    # ------------------------------------------------------------------
    # Guard checks
    # ------------------------------------------------------------------

    def _collect_guard_notes(
        self,
        session: Any,
        *,
        mode: ImportMode,
        strict: bool,
    ) -> list[str]:
        notes: list[str] = []

        session_model = (
            getattr(session, "model_id", None)
            or getattr(session.tree, "model_id", None)
        )
        if self.model_id and session_model and self.model_id != session_model:
            msg = (
                f"transcript model {self.model_id!r} differs from session "
                f"model {session_model!r}"
            )
            warnings.warn(msg, UserWarning, stacklevel=4)
            if mode == "merge":
                raise TranscriptModelMismatch(
                    msg + " — refusing to merge under semantic mismatch"
                )
            notes.append(f"model_mismatch: {msg}")

        session_sys = getattr(session.config, "system_prompt", None)
        if (
            self.system_prompt is not None
            and session_sys is not None
            and self.system_prompt != session_sys
        ):
            msg = "transcript system prompt differs from current session"
            warnings.warn(msg, UserWarning, stacklevel=4)
            notes.append(
                f"system_prompt_mismatch: original was {self.system_prompt!r}"
            )

        session_hashes = session.probe_hashes()
        drift: list[str] = []
        missing: list[str] = []
        for ref in self.probes:
            current = session_hashes.get(ref.name)
            if current is None:
                missing.append(ref.name)
                continue
            if ref.sha256 and current != ref.sha256:
                drift.append(ref.name)
        if missing:
            msg = (
                f"transcript references probes the session doesn't carry: "
                f"{', '.join(sorted(missing))}; readings recorded as-imported"
            )
            warnings.warn(msg, UserWarning, stacklevel=4)
            notes.append(f"probes_missing: {sorted(missing)}")
        if drift:
            msg = (
                f"probe content drift between transcript and session: "
                f"{', '.join(sorted(drift))}"
            )
            warnings.warn(msg, UserWarning, stacklevel=4)
            if strict:
                raise TranscriptProbeDriftError(msg)
            notes.append(f"probe_drift: {sorted(drift)}")
        return notes

    # ------------------------------------------------------------------
    # Mode resolution
    # ------------------------------------------------------------------

    def _resolve_attach_parent(
        self, session: Any, *, mode: ImportMode,
    ) -> str:
        if mode == "default":
            return session.tree.root_id
        if mode == "here":
            return session.tree.active_node_id
        # mode == "merge" — ``ImportMode`` is a Literal of the three
        # values above, so type-narrowing covers the universe.
        return self._find_merge_anchor(session)

    def _find_merge_anchor(self, session: Any) -> str:
        """Walk the active path; return the deepest user-turn match.

        Match = same user-turn text at the same position.  Assistant
        outputs are advisory and may differ (seed × steering × model
        × probe-state — byte-equal is rare).  Falls back to the tree
        root when no user-turn prefix matches.
        """
        active_users: list[tuple[str, str]] = [
            (node.id, node.text or "")
            for node in session.tree.active_path()
            if node.role == "user"
        ]
        transcript_users = [t.text for t in self.turns if t.role == "user"]

        anchor_id: str = session.tree.root_id
        for (node_id, active_text), tr_text in zip(
            active_users, transcript_users, strict=False,
        ):
            if active_text != tr_text:
                break
            anchor_id = node_id
        return anchor_id

    # ------------------------------------------------------------------
    # Branch planning
    # ------------------------------------------------------------------

    def _branch_chain(self, tree: LoomTree, attach_parent: str) -> list[LoomNode]:
        """The node templates :meth:`LoomTree.graft` attaches under ``attach_parent``.

        When ``attach_parent`` is a merge anchor, the matching user-turn
        prefix is already on the tree path above it: those user turns and
        the turns interleaved with them are skipped, and only the tail is
        planned.  Re-adding them under the anchor would land duplicates at
        the wrong depth.  System turns are skipped (the synthetic root
        carries the system slot).
        """
        skip_count = 0
        if attach_parent != tree.root_id:
            anchor_users = [n for n in tree.path_to(attach_parent) if n.role == "user"]
            transcript_users = [t for t in self.turns if t.role == "user"]
            for path_node, t_user in zip(anchor_users, transcript_users, strict=False):
                if path_node.text != t_user.text:
                    break
                skip_count += 1

        chain: list[LoomNode] = []
        users_seen = 0
        for turn in self.turns:
            if turn.role == "system":
                continue
            if turn.role == "user":
                users_seen += 1
                if users_seen <= skip_count:
                    continue
            elif users_seen < skip_count:
                continue
            # Provenance is recipe presence: a generated turn keeps its
            # recipe in whatever seat it was recorded; a recipe-less turn
            # is an authored one.
            chain.append(LoomNode(
                id="",
                parent_id=None,
                role=turn.role,
                text=turn.text,
                role_label=turn.speaker,
                thinking_text=turn.thinking,
                recipe=turn.recipe,
                aggregate_readings=dict(turn.readings),
                applied_steering=turn.recipe.steering if turn.recipe else None,
            ))
        return chain


__all__ = [
    "ImportMode",
    "ProbeRef",
    "DROWSE_TRANSCRIPT_VERSION",
    "ReplayStep",
    "Transcript",
    "TranscriptError",
    "TranscriptFormatError",
    "TranscriptModelMismatch",
    "TranscriptProbeDriftError",
    "TranscriptReplay",
    "Turn",
]
