"""Shared artifact primitives: name grammar, digests, and integrity checks.

The cross-cutting infrastructure every persisted artifact family shares:

- ``NAME_REGEX`` — the artifact-name grammar.  Manifolds, selectors, templates,
  and the local lens/SAE source names all validate against this one pattern;
- ``LABEL_REGEX`` — the stricter manifold node-label grammar (no ``.``);
- ``is_sha256_digest`` / ``canonical_json`` — the one digest grammar and the
  one canonical-JSON encoding every content hash is taken over;
- ``hash_file`` / ``verify_integrity`` — the sha256 integrity helpers (the
  neutral/alignment caches, the lens shard sidecars, and the manifold format's
  own integrity manifest build on these);
- ``PROFILE_FORMAT_VERSION`` — the current profile sidecar version written by
  :func:`drowse.geometry.profile.save_profile`.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from drowse.base.paths import ensure_within


NAME_REGEX = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")

# Manifold node *labels* are stricter than ``NAME_REGEX``: they drop ``.``
# because the dot is the bipolar-pole separator (``deer.wolf``) and the
# steering grammar addresses a label via ``%label`` — a dotted label could
# neither be typed unambiguously nor resolved through the bare-name tier.  A
# templated manifold's node labels are its template's slugged values, so the
# template format shares this grammar.
LABEL_REGEX = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def is_sha256_digest(value: Any) -> bool:
    """True for a canonical lowercase 64-hex sha256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def canonical_json(obj: Any) -> bytes:
    """Stable canonical-JSON encoding (sorted keys, compact) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

# Current profile sidecar format version.  v6 is the honest stamp for the
# five-key schema (``format_version`` / ``drowse_version`` / ``method`` /
# ``tensor_sha256`` / ``provenance``): the field set was cut from fourteen keys
# without a bump, so a v5 file already fails the exact-set validator and the
# invalidation was paid in practice before it was declared.  No cache rides this
# version — the neutral-activation and alignment caches carry their own sidecar
# schemas and format versions in :mod:`drowse.store.alignment` — so the only reader
# affected is a user-saved ``Profile``, which gets a clear regenerate error.
PROFILE_FORMAT_VERSION = 6


def hash_file(path: Path) -> str:
    """Return hex sha256 of a file's contents.

    Shared by every artifact integrity manifest.
    """
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# In-process fingerprint cache keyed by absolute file path.
# Entry: stable file identity + expected sha256 -> last verification matched.
# Short-circuits full hashing on warm loads. First load, or any stat change,
# still runs the full sha256 before the entry is (re-)populated.
_FINGERPRINT_CACHE_MAX = 4096
_FINGERPRINT_CACHE: dict[str, tuple[int, int, int, int, int, str]] = {}


def verify_integrity(folder: Path, files: dict[str, str]) -> tuple[bool, list[str]]:
    """Compare every file in `files` (path -> expected sha256) against disk.

    Returns (all_ok, list_of_bad_paths). A missing file counts as bad.

    Uses an in-process stat-identity fingerprint cache to avoid re-hashing
    on warm loads. On first load and after any stat change, the full sha256
    still runs — the cache is purely an optimization and does not weaken the
    integrity contract.
    """
    bad: list[str] = []
    for rel, expected in files.items():
        # A manifest entry that resolves outside ``folder`` (a ``..`` or
        # absolute ``rel`` in a downloaded manifest) is treated as a failed
        # file rather than read off-tree. ``ensure_within`` is the path-
        # traversal barrier.
        try:
            fp = ensure_within(folder, rel)
        except ValueError:
            bad.append(rel)
            continue
        if not fp.exists():
            bad.append(rel)
            continue
        key = str(fp.resolve())
        try:
            st = fp.stat()
        except OSError:
            bad.append(rel)
            continue
        fp_key = (
            st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino,
            expected,
        )
        cached = _FINGERPRINT_CACHE.pop(key, None)
        if cached is not None and cached == fp_key:
            _FINGERPRINT_CACHE[key] = cached
            continue
        if hash_file(fp) != expected:
            _FINGERPRINT_CACHE.pop(key, None)
            bad.append(rel)
            continue
        while len(_FINGERPRINT_CACHE) >= _FINGERPRINT_CACHE_MAX:
            _FINGERPRINT_CACHE.pop(next(iter(_FINGERPRINT_CACHE)))
        _FINGERPRINT_CACHE[key] = fp_key
    return (not bad, bad)
