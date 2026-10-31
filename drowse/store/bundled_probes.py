"""The bundled default probe roster.

A steering vector lives as a 2-node ``pca`` manifold, so the bundled probe set
is a set of fitted bundled manifolds: :func:`load_default_manifolds` reads the
tag → manifold roster, and :func:`bootstrap_probe_roster` fits-or-loads it for
the loaded model.  The probe-centering means come from the session's neutral
baseline, not from here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from drowse.errors import ManifoldNotRegisteredError

if TYPE_CHECKING:
    from drowse.geometry.manifold import Manifold

_log = logging.getLogger(__name__)


def load_default_manifolds() -> dict[str, list[str]]:
    """Return {tag: [manifold_name, ...]} for bundled default/ manifolds.

    A steering vector lives as a 2-node ``pca`` manifold, tagged
    (``manifold.json::tags``) for category-grouped probe bootstrap.  Triggers
    first-run materialization of the bundled artifacts, then reads the
    memoized selector index rather than re-walking the tree.
    """
    from drowse.store.bundled import materialize_bundled_artifacts
    from drowse.store.selectors import Selector, resolve

    materialize_bundled_artifacts()
    by_tag: dict[str, list[str]] = {}
    for concept in resolve(Selector(kind="namespace", value="default")):
        for tag in concept.tags:
            by_tag.setdefault(tag, []).append(concept.name)
    return by_tag


def bootstrap_probe_roster(
    categories: list[str],
    *,
    model_id: str,
    load: "Callable[[str], Manifold]",
    fit: "Callable[[Path, Callable[[str], None] | None], object]",
    include_fitted_defaults: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> "dict[str, Manifold]":
    """Source the default probe roster as fitted bundled manifolds.

    One pass, two tiers:

    - **Tagged concept axes** — for each bundled 2-node ``pca`` manifold
      tagged in a requested category, ``load`` the per-model fit (registry
      key ``default/<name>``), and on a miss ``fit`` its folder then load it
      (a forward pass per node, then disk-cached).  Returned under the
      **bare** name (``confident.uncertain``) so the gate grammar and trait
      panel key off it.
    - **Fitted multi-node defaults** (``include_fitted_defaults``, the default
      roster) — every bundled ``default/`` manifold that is *already fitted*
      for ``model_id`` and not already in the roster (``personas`` /
      ``emotions``).  Attach-only: fitting a 107-node manifold would block
      startup for minutes, so an unfitted one is skipped with a log line.
      Returned under the qualified ``default/<name>`` selector so a manual
      attach from the manifolds drawer matches.

    ``load`` raises :class:`~drowse.errors.ManifoldNotRegisteredError` on a
    miss.  A fit/load failure for one manifold is logged and skipped, never
    fatal.  ``on_progress`` narrates the roster — an upfront count, then each
    concept's own fit progress.
    """
    from drowse.base.paths import manifold_dir, safe_model_id

    defaults = load_default_manifolds()
    probes: dict[str, Manifold] = {}
    # ``dict.fromkeys`` dedupes (a concept can be tagged in two requested
    # categories) while preserving roster order, so the announced count is
    # the number of concepts actually walked.
    roster = list(dict.fromkeys(
        name for cat in categories for name in defaults.get(cat, [])
    ))
    if on_progress is not None and roster:
        on_progress(
            f"Fitting default probe roster ({len(roster)} concepts)..."
        )
    for name in roster:
        key = f"default/{name}"
        try:
            try:
                probes[name] = load(key)
            except ManifoldNotRegisteredError:
                if on_progress is not None:
                    on_progress(f"Fitting probe '{name}'...")
                fit(manifold_dir("default", name), on_progress)
                probes[name] = load(key)
        except Exception as e:
            _log.warning("manifold probe '%s' failed to fit/load: %s", name, e)

    if not include_fitted_defaults:
        return probes

    # Attach-only sweep: every fitted bundled multi-node manifold that the
    # tagged tier didn't already pick up (personas / emotions carry no
    # category tag, so they fall through to here).
    from drowse.store.bundled import bundled_manifold_names
    from drowse.store.manifold_folder import ManifoldFolder
    from drowse.store.manifold_schema import ManifoldFormatError

    stem = safe_model_id(model_id)
    for name in bundled_manifold_names():
        key = f"default/{name}"
        if name in probes or key in probes:
            continue
        try:
            # Attach preflight needs fitted stems only.  The selected
            # model pair is strictly verified by ``load``.
            folder = ManifoldFolder.load(
                manifold_dir("default", name), verify_manifest=False,
            )
        except (ManifoldFormatError, FileNotFoundError):
            continue
        if stem not in folder.tensor_models():
            _log.info(
                "manifold probe '%s' not fitted for %s — skipping "
                "(fit it to auto-attach next launch)",
                key, model_id,
            )
            continue
        try:
            probes[key] = load(key)
        except Exception as e:
            _log.warning("manifold probe '%s' failed to load: %s", key, e)
    return probes
