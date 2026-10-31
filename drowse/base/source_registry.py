"""Per-model source families — the lifecycle the lens and SAE families share.

Both families are *source registries*: Drowse owns what it fits or trains and
stores it under ``DROWSE_HOME``, external publishers keep their payloads in
their own caches and Drowse stores only a pinned binding, and one selection
file names which of them the runtime uses.  The on-disk shape is the same in
both families::

    models/<safe_model_id>/<dirname>/
      active.json                    the runtime selection
      local/<name>/manifest.json     a Drowse-owned artifact
      bindings/<stem>.json           a commit-pinned provider binding

:class:`SourceFamily` owns that shape once — paths, the public source grammar,
existence, and the locked, validated selection — parameterized by the family's
name grammars, provider spelling, and selection slots.  It is also the ONE
parser of the source grammar: ``local:<name>`` names a local artifact;
anything else names a provider binding, spelled ``<provider_prefix><name>``
when the family has a prefix (SAE: ``saelens:<release>``) or as the bare
binding name when it does not (lens: ``workspace-r``).

What stays per-family: the payload *schemas* behind a manifest or binding,
their format versions, the loaders that turn a selection into a resident
artifact, and the listing rows each surface shows.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from drowse.base.atomic import artifact_lock, write_json_atomic
from drowse.base.paths import ensure_within, model_dir
from drowse.errors import SourceSpecError

LOCAL = "local"
LOCAL_PREFIX = "local:"


@dataclass(frozen=True)
class SourceRef:
    """One parsed source: ``kind`` is ``"local"`` or the provider kind."""

    kind: str
    name: str

    @property
    def local(self) -> bool:
        return self.kind == LOCAL


class SourceFamily:
    """Paths, grammar, and active selection for one per-model family.

    ``local_name`` is the grammar of a Drowse-owned artifact name;
    ``provider_name_ok`` accepts a provider binding name (provider-shaped —
    an SAE release id may carry slashes and mixed case).  ``binding_stem``
    maps a provider name to its binding filename stem.

    ``slots=None`` keeps one selection per model, stored flat as
    ``{format_version, model_id, kind, name}``.  A slotted family (the lens:
    one ``output`` and one ``input`` side) stores
    ``{format_version, model_id, <slot>: {kind, name} | null, ...}`` and
    routes each source to its slot with ``slot_of(model_id, ref)``.
    """

    def __init__(
        self,
        *,
        label: str,
        dirname: str,
        format_version: int,
        provider_kind: str,
        provider_prefix: str | None,
        local_name: re.Pattern[str],
        provider_name_ok: Callable[[str], bool],
        binding_stem: Callable[[str], str] = lambda name: name,
        slots: tuple[str, ...] | None = None,
        slot_of: Callable[[str, SourceRef], str] | None = None,
    ) -> None:
        if (slots is None) != (slot_of is None):
            raise ValueError("a slotted family needs both slots and slot_of")
        self.label = label
        self.dirname = dirname
        self.format_version = format_version
        self.provider_kind = provider_kind
        self.provider_prefix = provider_prefix
        self.local_name = local_name
        self._provider_name_ok = provider_name_ok
        self._binding_stem = binding_stem
        self.slots = slots
        self._slot_of = slot_of

    # -- paths -------------------------------------------------------------

    def root(self, model_id: str) -> Path:
        return model_dir(model_id) / self.dirname

    def local_dir(self, model_id: str, name: str) -> Path:
        self._validate(SourceRef(LOCAL, name))
        return ensure_within(self.root(model_id) / "local", name)

    def bindings_dir(self, model_id: str) -> Path:
        return self.root(model_id) / "bindings"

    def binding_path(self, model_id: str, name: str) -> Path:
        self._validate(SourceRef(self.provider_kind, name))
        return ensure_within(
            self.bindings_dir(model_id), f"{self._binding_stem(name)}.json",
        )

    def active_path(self, model_id: str) -> Path:
        return self.root(model_id) / "active.json"

    def path_of(self, model_id: str, ref: SourceRef) -> Path:
        """A local artifact's manifest, or a provider source's binding."""
        if ref.local:
            return self.local_dir(model_id, ref.name) / "manifest.json"
        return self.binding_path(model_id, ref.name)

    # -- grammar -----------------------------------------------------------

    def _validate(self, ref: SourceRef) -> None:
        if ref.kind == LOCAL:
            if self.local_name.fullmatch(ref.name) is None:
                raise SourceSpecError(
                    f"invalid local {self.label} name {ref.name!r} "
                    f"(must match {self.local_name.pattern})"
                )
        elif ref.kind == self.provider_kind:
            if not self._provider_name_ok(ref.name):
                raise SourceSpecError(
                    f"invalid {self.label} {self.provider_kind} source "
                    f"name {ref.name!r}"
                )
        else:
            raise SourceSpecError(f"unknown {self.label} source kind {ref.kind!r}")

    def _grammar(self) -> str:
        provider = (
            f"{self.provider_prefix}NAME" if self.provider_prefix
            else "a fetched binding name"
        )
        return f"local:NAME or {provider}"

    def parse(self, source: str) -> SourceRef:
        """Parse the public source grammar; raises :class:`~drowse.errors.SourceSpecError`."""
        text = source.strip()
        if text.startswith(LOCAL_PREFIX):
            ref = SourceRef(LOCAL, text[len(LOCAL_PREFIX):])
        elif self.provider_prefix is not None:
            if not text.startswith(self.provider_prefix):
                raise SourceSpecError(
                    f"{self.label} source must be {self._grammar()}, "
                    f"got {source!r}"
                )
            ref = SourceRef(self.provider_kind, text[len(self.provider_prefix):])
        else:
            ref = SourceRef(self.provider_kind, text)
        try:
            self._validate(ref)
        except ValueError as exc:
            raise SourceSpecError(
                f"{self.label} source must be {self._grammar()}: {exc}"
            ) from None
        return ref

    def format(self, ref: SourceRef) -> str:
        """Render a ref back into the public grammar (inverse of parse)."""
        if ref.local:
            return f"{LOCAL_PREFIX}{ref.name}"
        return f"{self.provider_prefix or ''}{ref.name}"

    # -- presence ----------------------------------------------------------

    def exists(self, model_id: str, ref: SourceRef) -> bool:
        """Whether the source is on disk, so a selection can never dangle."""
        try:
            return self.path_of(model_id, ref).exists()
        except ValueError:
            return False

    def local_names(self, model_id: str) -> list[str]:
        """Grammatical local artifact names that carry a manifest."""
        return sorted(
            path.parent.name
            for path in (self.root(model_id) / "local").glob("*/manifest.json")
            if self.local_name.fullmatch(path.parent.name) is not None
        )

    # -- active selection --------------------------------------------------

    def _ref_from(self, value: Any) -> SourceRef | None:
        if (
            not isinstance(value, dict)
            or set(value) != {"kind", "name"}
            or not isinstance(value["kind"], str)
            or not isinstance(value["name"], str)
        ):
            return None
        ref = SourceRef(value["kind"], value["name"])
        try:
            self._validate(ref)
        except ValueError:
            return None
        return ref

    def _read(self, model_id: str) -> dict[str, SourceRef | None] | None:
        """The validated per-slot selection, or ``None``.

        Any deviation — unreadable file, non-current format version, foreign
        model id, unknown kind, ungrammatical name — reads as "no selection"
        rather than raising: a stale or hand-edited selection must not keep a
        model from starting.
        """
        path = self.active_path(model_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if (
            not isinstance(data, dict)
            or data.get("format_version") != self.format_version
            or data.get("model_id") != model_id
        ):
            return None
        if self.slots is None:
            if set(data) != {"format_version", "model_id", "kind", "name"}:
                return None
            ref = self._ref_from({"kind": data["kind"], "name": data["name"]})
            return None if ref is None else {"": ref}
        if set(data) != {"format_version", "model_id", *self.slots}:
            return None
        out: dict[str, SourceRef | None] = {}
        for slot in self.slots:
            if data[slot] is None:
                out[slot] = None
                continue
            ref = self._ref_from(data[slot])
            if ref is None:
                return None
            out[slot] = ref
        return out

    def _payload(
        self, model_id: str, selection: dict[str, SourceRef | None],
    ) -> dict[str, Any]:
        head: dict[str, Any] = {
            "format_version": self.format_version, "model_id": model_id,
        }
        if self.slots is None:
            ref = selection[""]
            assert ref is not None
            return {**head, "kind": ref.kind, "name": ref.name}
        return {
            **head,
            **{
                slot: (
                    None if selection.get(slot) is None
                    else {"kind": selection[slot].kind, "name": selection[slot].name}  # type: ignore[union-attr]
                )
                for slot in self.slots
            },
        }

    def _slot(self, model_id: str, ref: SourceRef) -> str:
        if self._slot_of is None:
            return ""
        slot = self._slot_of(model_id, ref)
        assert self.slots is not None
        if slot not in self.slots:
            raise ValueError(f"{self.label} source routes to unknown slot {slot!r}")
        return slot

    def active(self, model_id: str, slot: str | None = None) -> SourceRef | None:
        """The selected source (in ``slot`` for a slotted family)."""
        selection = self._read(model_id)
        if selection is None:
            return None
        if self.slots is None:
            return selection[""]
        return selection.get(slot or self.slots[0])

    def select(self, model_id: str, ref: SourceRef) -> Path:
        """Select a source, validating its grammar and on-disk existence."""
        self._validate(ref)
        if not self.exists(model_id, ref):
            raise FileNotFoundError(
                f"{self.label} source {self.format(ref)} is not available"
            )
        slot = self._slot(model_id, ref)
        path = self.active_path(model_id)
        with artifact_lock(path):
            selection: dict[str, SourceRef | None] = (
                {"": ref} if self.slots is None
                else {**(self._read(model_id) or {}), slot: ref}
            )
            write_json_atomic(path, self._payload(model_id, selection))
        return path

    def use(self, model_id: str, source: str) -> SourceRef:
        """Select a source from the public grammar."""
        ref = self.parse(source)
        self.select(model_id, ref)
        return ref

    def clear_if_active(self, model_id: str, ref: SourceRef) -> bool:
        """Unpublish the selection wherever it names exactly this source.

        Called while a lifecycle operation removes a source, so the selection
        can never outlive what it points at — and never clears a different
        source.  Returns whether anything was cleared.
        """
        path = self.active_path(model_id)
        with artifact_lock(path):
            selection = self._read(model_id)
            if selection is None:
                return False
            hits = [slot for slot, current in selection.items() if current == ref]
            if not hits:
                return False
            remaining = {
                slot: (None if slot in hits else current)
                for slot, current in selection.items()
            }
            if all(current is None for current in remaining.values()):
                path.unlink(missing_ok=True)
            else:
                write_json_atomic(path, self._payload(model_id, remaining))
            return True
