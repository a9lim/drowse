"""Real ``DrowseSession`` construction on the tiny CPU checkpoints.

:class:`TinySessionFactory` loads a tiny checkpoint through
``DrowseSession.from_pretrained`` (the real loader, ~10 ms on CPU) and, by
default, warms the neutral baseline so ``session.whitener`` and
``session.layer_means`` are populated before the test starts.  The neutral
activations are computed once per process (~0.3 s) and copied into whatever
``DROWSE_HOME`` a later session is built under, so per-test homes hit the
cache instead of recomputing.

Prefer these over ``DrowseSession.__new__`` skeletons and hand-set private
attributes: a test written against a real session survives refactors of the
session's internals.
"""

from __future__ import annotations

import os
import shutil
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.support.tiny_models import (
    FAMILIES,
    ChatFamily,
    ReplySource,
    ScriptedReplies,
    write_tiny_checkpoint,
)

__all__ = [
    "TinyCheckpoint",
    "TinySessionFactory",
    "baseline_corpus",
    "drowse_home",
    "forward_logits",
    "register_vectors",
]


@contextmanager
def drowse_home(path: Path) -> Iterator[Path]:
    """Point ``DROWSE_HOME`` at ``path`` for the block, resetting selector caches."""
    from drowse.store import selectors

    previous = os.environ.get("DROWSE_HOME")
    os.environ["DROWSE_HOME"] = str(path)
    selectors.invalidate()
    try:
        yield path
    finally:
        if previous is None:
            os.environ.pop("DROWSE_HOME", None)
        else:
            os.environ["DROWSE_HOME"] = previous
        selectors.invalidate()


def baseline_corpus(response: str) -> list[str]:
    """One pole corpus: ``response`` answering every shared baseline prompt."""
    from drowse.fit.capture import _load_baseline_prompts

    return [response] * len(_load_baseline_prompts())


def register_vectors(session: Any, *names: str) -> dict[str, Any]:
    """Register a fixed random unit direction per layer under each name.

    Directions are seeded by the name, so the same name gives the same vector
    in every session.  Returns ``{name: Profile}``.
    """
    import torch

    from drowse.geometry.profile import Profile

    hidden = int(session.model_info["hidden_dim"])
    profiles: dict[str, Any] = {}
    for name in names:
        generator = torch.Generator().manual_seed(zlib.crc32(name.encode("utf-8")))
        tensors = {}
        for layer in range(len(session.layers)):
            direction = torch.randn(hidden, generator=generator)
            tensors[layer] = direction / direction.norm()
        profiles[name] = Profile(tensors)
        session.registry.put_profile(name, profiles[name])
    return profiles


def forward_logits(session: Any, text: str = "probe text") -> Any:
    """Last-position logits of one raw forward through the session's model.

    The forward runs whatever steering hooks are currently attached, so it
    shows whether an open ``session.steering(...)`` scope is acting.
    """
    import torch

    ids = session.tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        return session.model(input_ids=ids).logits[0, -1].clone()


@dataclass(frozen=True)
class TinyCheckpoint:
    """A tiny model + tokenizer saved as a local HF checkpoint."""

    path: Path
    family: ChatFamily

    @property
    def model_id(self) -> str:
        return str(self.path)

    @classmethod
    def write(cls, root: Path, family: str = "llama") -> "TinyCheckpoint":
        spec = FAMILIES[family]
        return cls(write_tiny_checkpoint(root / f"tiny-{spec.name}", spec), spec)


_NEUTRAL_GLOBS = ("neutral_activations*.json", "neutral_activations*.safetensors")


class TinySessionFactory:
    """Build real sessions on one tiny checkpoint; close them all at teardown.

    ``factory()`` returns a ``DrowseSession`` loaded on CPU with ``probes=[]``.

    - ``warm=True`` (default) builds the whitener before returning, so
      steering scopes, ``~``/``|`` projections and probes work immediately.
      Pass ``warm=False`` to test the lazy (cold-session) paths.
    - Remaining keyword arguments go to ``DrowseSession.from_pretrained``.

    ``factory.scripted(replies, ...)`` returns ``(session, script)`` with
    :class:`ScriptedReplies` installed on the session's model; ``script.prompts``
    records every rendered input the model saw.
    """

    def __init__(self, checkpoint: TinyCheckpoint, seed_dir: Path) -> None:
        self.checkpoint = checkpoint
        self.seed_dir = seed_dir
        self._sessions: list[Any] = []
        self._scripts: list[ScriptedReplies] = []

    def __call__(self, *, warm: bool = True, **kwargs: Any) -> Any:
        from drowse.session.session import DrowseSession

        kwargs.setdefault("device", "cpu")
        kwargs.setdefault("probes", [])
        if warm:
            self._seed_neutral_cache()
        session = DrowseSession.from_pretrained(self.checkpoint.model_id, **kwargs)
        self._sessions.append(session)
        if warm:
            assert session.whitener is not None, "tiny session whitener failed to build"
            self._remember_neutral_cache()
        return session

    def scripted(
        self, replies: ReplySource, **kwargs: Any,
    ) -> tuple[Any, ScriptedReplies]:
        session = self(**kwargs)
        script = ScriptedReplies(
            session.model, session.tokenizer, replies, family=self.checkpoint.family,
        ).install()
        self._scripts.append(script)
        return session, script

    def close(self) -> None:
        while self._scripts:
            self._scripts.pop().remove()
        while self._sessions:
            self._sessions.pop().close()

    # -- neutral baseline seeding --------------------------------------------

    def _model_dir(self) -> Path:
        from drowse.base.paths import model_dir

        return model_dir(self.checkpoint.model_id)

    def _seed_neutral_cache(self) -> None:
        """Copy the process-wide neutral cache into the current home if absent."""
        target = self._model_dir()
        if any(target.glob(_NEUTRAL_GLOBS[0])):
            return
        seeded = [path for pattern in _NEUTRAL_GLOBS for path in self.seed_dir.glob(pattern)]
        if not seeded:
            return
        target.mkdir(parents=True, exist_ok=True)
        for path in seeded:
            shutil.copy2(path, target / path.name)

    def _remember_neutral_cache(self) -> None:
        """Snapshot the current home's neutral cache as the process-wide seed."""
        if any(self.seed_dir.glob(_NEUTRAL_GLOBS[0])):
            return
        source = self._model_dir()
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        for pattern in _NEUTRAL_GLOBS:
            for path in source.glob(pattern):
                shutil.copy2(path, self.seed_dir / path.name)
