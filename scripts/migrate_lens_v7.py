#!/usr/bin/env python
"""Migrate a DROWSE_HOME's Jacobian lenses from format v6 to v7.

drowse 5.4 reads only lens format v7.  This rewrites every
``models/<id>/jlens/`` tree in place into ``models/<id>/lens/``:

- ``local/default`` → ``local/j`` and ``local/relp`` → ``local/r`` (any other
  local name → ``local/<kind>-<name>``).  Each manifest/checkpoint pointer is
  rewritten to the v7 schema (``kind`` + ``sites``), and each shard is renamed
  ``jlens[.partial].layer-<L>.gen-<id>`` → ``out[.partial].layer-<L>.gen-<id>``.
  Shard *bytes* are never touched, so every recorded sha256 carries over.
  A ``local/relp`` pointer written before 2026-10-28 (which recorded the
  standard policy for a RelP fit) migrates as the R-lens it is.
- Provider bindings are rewritten to the v2 schema that maps files to lens
  keys.  A v1 Neuronpedia binding recorded no checkpoint digest; it is hashed
  from the local Hugging Face cache when present, and otherwise dropped with
  a note to re-fetch (bindings are small; the payload never left the cache).
- ``active.json`` becomes the per-side selection (the old source lands on the
  output side).

Shards are hard-linked into a staging tree that replaces ``jlens/`` only once
it is complete, so an interrupted run leaves the v6 tree intact.  Run with
``--dry-run`` first.  Idempotent: a model already migrated is skipped.

    python scripts/migrate_lens_v7.py [--home ~/.drowse] [--dry-run]
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

_V6 = 6
_V7 = 7
_PRE_FIX_RELP_RULES = {
    "ln_rule": True,
    "identity_rule": True,
    "half_rule": True,
    "half_rule_beta": 0.5,
    "include_qk_norms": False,
}
_CARRIED_FIELDS = (
    "n_prompts", "d_model", "dtype", "corpus_spec", "corpus_sha256",
    "corpus_hash_kind", "seq_len", "dim_batch", "raw_corpus_sha256",
    "raw_prompt_count", "usable_prompt_count", "model_layer_count",
    "model_fingerprint", "model_source_fingerprint", "checkpoint",
    "consumed_prefix_sha256",
)


class MigrationError(RuntimeError):
    pass


def _kind_of(pointer: dict[str, Any], old_name: str) -> str:
    method = pointer.get("method")
    if method == "relp_cotangent_sum":
        return "R"
    if method != "jlens_cotangent_sum":
        raise MigrationError(f"unknown v6 lens method {method!r}")
    # Before 2026-10-28 the checkpoint writer dropped the rules, so a RelP fit
    # (and any final promoted from its checkpoint) recorded the standard
    # policy; only local/relp ever held one.
    return "R" if old_name == "relp" else "J"


def _new_local_name(kind: str, old_name: str) -> str:
    if old_name in ("default", "relp"):
        return kind.lower()
    name = f"{kind.lower()}-{old_name}"
    if len(name) > 63:
        raise MigrationError(f"local lens name {old_name!r} is too long to tag")
    return name


def _v7_policy(policy: dict[str, Any], kind: str) -> dict[str, Any]:
    rules = policy.get("backward_rules")
    if kind == "R" and rules is None:
        rules = dict(_PRE_FIX_RELP_RULES)
    return {
        "direction": "output",
        "rules": rules if kind == "R" else None,
        "skip_first_positions": policy["skip_first_positions"],
        "corpus_preprocess_version": policy["corpus_preprocess_version"],
        "doc_chars": policy["doc_chars"],
    }


def _new_shard_name(name: str) -> str:
    if not name.startswith("jlens."):
        raise MigrationError(f"unexpected v6 shard name {name!r}")
    return "out." + name[len("jlens."):]


def _migrate_pointer(
    pointer: dict[str, Any], kind: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The v7 pointer and the ``{old shard: new shard}`` renames it implies."""
    if pointer.get("format_version") != _V6:
        raise MigrationError(
            f"pointer has format_version {pointer.get('format_version')!r}, "
            f"not {_V6}"
        )
    if pointer.get("checkpoint") and pointer.get("base_n_prompts") not in (0, None):
        raise MigrationError("checkpoint is not self-contained (base_n_prompts > 0)")
    files: dict[str, str] = pointer["tensor_files"]
    renames = {old: _new_shard_name(old) for old in files.values()}
    v7 = {
        "format_version": _V7,
        "kind": kind,
        "sites": {
            "out": {
                "layers": [int(layer) for layer in pointer["source_layers"]],
                "tensor_files": {
                    layer: renames[name] for layer, name in files.items()
                },
                "tensor_sha256": dict(pointer["tensor_sha256"]),
            },
        },
        "estimator_policy": _v7_policy(pointer["estimator_policy"], kind),
        **{field: pointer[field] for field in _CARRIED_FIELDS},
    }
    return v7, renames


def _cached_digest(repo_id: str, filename: str, revision: str) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(repo_id, filename, revision=revision)
    if not isinstance(path, str):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _migrate_binding(
    payload: dict[str, Any], *,
    digest_of: Callable[[str, str, str], str | None] = _cached_digest,
) -> dict[str, Any] | None:
    """The v2 binding, or ``None`` when it cannot be completed offline."""
    if payload.get("format_version") != 1:
        raise MigrationError("binding is not format v1")
    provider = payload["provider"]
    common = {
        "format_version": 2,
        "kind": "huggingface",
        "provider": provider,
        "name": payload["name"],
        "model_id": payload["model_id"],
        "model_revision": payload["model_revision"],
        "repo_id": payload["repo_id"],
        "repo_revision": payload["repo_revision"],
        "d_model": payload["d_model"],
        "n_prompts": payload["n_prompts"],
        "corpus": payload["corpus"],
        "seq_len": payload["seq_len"],
    }
    layers = list(payload["source_layers"])
    if provider == "workspace-lenses":
        key = "R:out" if payload["estimator"] == "relp" else "J:out"
        return {
            **common,
            "files": {key: {
                "path": payload["checkpoint"],
                "sha256": payload["checkpoint_sha256"],
                "layers": layers,
            }},
            "skip_first": payload["skip_first"],
            "provenance": {
                "estimator": payload["estimator"],
                "rules_json": payload["rules_json"],
                "target_layer": payload["target_layer"],
            },
        }
    if provider == "neuronpedia":
        digest = digest_of(
            payload["repo_id"], payload["checkpoint"], payload["repo_revision"],
        )
        if digest is None:
            return None
        return {
            **common,
            "files": {"J:out": {
                "path": payload["checkpoint"], "sha256": digest, "layers": layers,
            }},
            "skip_first": 0,
            "provenance": {
                "config_file": payload["config_file"],
                "config_sha256": payload["config_sha256"],
                "dim_batch": payload["dim_batch"],
            },
        }
    raise MigrationError(f"unknown lens provider {provider!r}")


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def migrate_model(
    model_dir: Path, *, dry_run: bool = False,
    log: Callable[[str], None] = print,
    digest_of: Callable[[str, str, str], str | None] = _cached_digest,
) -> bool:
    """Migrate one ``models/<id>`` directory. Returns whether it changed."""
    old_root = model_dir / "jlens"
    new_root = model_dir / "lens"
    if not old_root.is_dir():
        return False
    if new_root.exists():
        raise MigrationError(
            f"{model_dir.name}: both jlens/ and lens/ exist — resolve by hand"
        )
    staging = model_dir / "lens.migrating"
    if staging.exists() and not dry_run:
        shutil.rmtree(staging)
    names: dict[str, str] = {}
    actions: list[str] = []

    for local in sorted((old_root / "local").glob("*")):
        if not local.is_dir():
            continue
        # An interrupted first fit has only a checkpoint; keep it resumable.
        pointers = [
            (filename, json.loads((local / filename).read_text()))
            for filename in ("manifest.json", "checkpoint.json")
            if (local / filename).exists()
        ]
        if not pointers:
            actions.append(f"skip local:{local.name} (no pointer)")
            continue
        try:
            kind = _kind_of(pointers[0][1], local.name)
            new_name = _new_local_name(kind, local.name)
        except MigrationError as exc:
            actions.append(f"drop local:{local.name}: {exc}")
            continue
        target = staging / "local" / new_name
        for filename, pointer in pointers:
            try:
                v7, renames = _migrate_pointer(pointer, kind)
            except MigrationError as exc:
                actions.append(f"drop {local.name}/{filename}: {exc}")
                continue
            if filename == "manifest.json":
                names[local.name] = new_name
            for old, new in renames.items():
                if not (local / old).exists():
                    raise MigrationError(f"{local}/{old} is missing")
                if not dry_run:
                    target.mkdir(parents=True, exist_ok=True)
                    if not (target / new).exists():
                        _link(local / old, target / new)
            if not dry_run:
                _write_json(target / filename, v7)
            actions.append(
                f"local:{local.name} {filename} → local:{new_name} "
                f"({kind}, {len(renames)} shards)"
            )

    for binding in sorted((old_root / "bindings").glob("*.json")):
        v2 = _migrate_binding(json.loads(binding.read_text()), digest_of=digest_of)
        if v2 is None:
            actions.append(
                f"drop binding {binding.stem}: its checkpoint is not in the "
                f"local Hugging Face cache — re-run `drowse lens fetch "
                f"<model> {binding.stem}`"
            )
            continue
        if not dry_run:
            _write_json(staging / "bindings" / binding.name, v2)
        actions.append(f"binding {binding.stem} → v2 ({', '.join(v2['files'])})")

    active_path = old_root / "active.json"
    if active_path.exists():
        active = json.loads(active_path.read_text())
        kind, name = active.get("kind"), active.get("name")
        output: dict[str, str] | None
        if kind == "local" and name in names:
            output = {"kind": "local", "name": names[name]}
        elif kind == "huggingface" and (
            (staging / "bindings" / f"{name}.json").exists() or dry_run
        ):
            output = {"kind": "huggingface", "name": name}
        else:
            output = None
        if not dry_run:
            if output is not None:
                _write_json(staging / "active.json", {
                    "format_version": 2,
                    "model_id": active["model_id"],
                    "output": output,
                    "input": None,
                })
        actions.append(f"active → output {output}")

    for action in actions:
        log(f"{model_dir.name}: {action}")
    if dry_run:
        return bool(actions)
    if staging.exists():
        os.replace(staging, new_root)
    shutil.rmtree(old_root)
    return True


def migrate_home(
    home: Path, *, dry_run: bool = False, log: Callable[[str], None] = print,
    digest_of: Callable[[str, str, str], str | None] = _cached_digest,
) -> int:
    """Migrate every model under ``home``; returns how many changed."""
    changed = 0
    for model_dir in sorted((home / "models").glob("*")):
        if model_dir.is_dir() and migrate_model(
            model_dir, dry_run=dry_run, log=log, digest_of=digest_of,
        ):
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--home", type=Path,
        default=Path(os.environ.get("DROWSE_HOME", "~/.drowse")).expanduser(),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        changed = migrate_home(args.home, dry_run=args.dry_run)
    except MigrationError as exc:
        print(f"migrate_lens_v7: {exc}", file=sys.stderr)
        return 1
    verb = "would migrate" if args.dry_run else "migrated"
    print(f"{verb} {changed} model(s) under {args.home}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
