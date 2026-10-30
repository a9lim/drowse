"""CLI wiring tests for the ``drowse manifold`` lifecycle subverbs.

Parser shape + runner dispatch for the parity verbs added alongside the
fit/discover/generate/ls/show block: install / search / merge / push /
rm / clear / refresh / transfer, plus the ``ls -v`` and ``show -j``
changes.  The io-layer backends are mocked the way ``test_cli_flags``
mocks the io lifecycle / HF layer — these tests exercise the CLI plumbing
(arg parsing, runner→backend call shape, output idioms), not the
backends themselves (those live in ``test_manifolds_io`` /
``test_hf``).
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

import pytest

from drowse.cli.main import main


def _materialize_bundles_for_cli_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    monkeypatch.setattr("drowse.store.manifolds._materialized_home", None)
    from drowse.store import selectors
    from drowse.store.manifolds import materialize_bundled_manifolds
    selectors.invalidate()
    materialize_bundled_manifolds()
    selectors.invalidate()


def test_run_manifold_from_template_writes_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """End-to-end: template create -> manifold from-template derives the folder."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    ctx_file = tmp_path / "ctx.json"
    ctx_file.write_text(_json.dumps([
        {"user": "what day is it?", "assistant": "today is [DAY]"},
        {"user": "which day?", "assistant": "it's [DAY]"},
    ]))
    main([
        "template", "create", "weekday",
        "--slot", "[DAY]",
        "--values", "Monday", "Tuesday", "Wednesday",
        "--contexts", str(ctx_file),
    ])
    main(["manifold", "from-template", "weekday", "--fit-mode", "auto"])
    out = capsys.readouterr().out
    assert "3 nodes x 2 contexts" in out

    from drowse.store.manifolds import ManifoldFolder
    from drowse.base.paths import manifold_dir
    mf = ManifoldFolder.load(manifold_dir("local", "weekday"))
    assert mf.fit_mode == "auto"
    assert mf.node_labels == ["monday", "tuesday", "wednesday"]
    assert mf.template_ref == "local/weekday"
    assert dict(mf.node_groups())["monday"] == ["today is Monday", "it's Monday"]


def test_run_manifold_from_template_missing_template_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(SystemExit):
        main(["manifold", "from-template", "nonexistent"])


# ---------------------------------------------------------------------------
# Runner dispatch — each runner calls its backend with the mapped args
# ---------------------------------------------------------------------------

def test_run_manifold_install_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[tuple[Any, ...]] = []

    def fake_install(
        target: str, as_: Any = None, *, force: bool = False,
        on_progress: Any = None,
    ) -> Path:
        calls.append((target, as_, force))
        return Path("/home/.drowse/manifolds/local/circumplex")

    monkeypatch.setattr("drowse.store.hub.install_manifold", fake_install)
    main(["pack", "install", "alice/circumplex", "-a", "local/mood", "-f"])
    assert calls == [("alice/circumplex", "local/mood", True)]
    captured = capsys.readouterr()
    # Narrated on stderr before the network call (hub progress bars vanish
    # off-TTY); the result line is stdout.
    assert captured.err.startswith("Installing alice/circumplex...\n")
    assert captured.out.startswith("Installed alice/circumplex")


def test_run_manifold_search_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    seen: list[Any] = []

    def fake_search(query: Any) -> list[dict[str, Any]]:
        seen.append(query)
        return [{
            "name": "circumplex", "namespace": "alice", "description": "moods",
            "tags": [], "node_count": 9, "domain_label": "box(2d)",
            "fit_mode": "authored", "tensor_models": ["gemma"],
        }]

    monkeypatch.setattr("drowse.store.hub.search_manifolds", fake_search)
    main(["pack", "search", "mood"])
    assert seen == ["mood"]
    captured = capsys.readouterr()
    assert captured.err.startswith("Searching Hugging Face for manifolds matching 'mood'...\n")
    assert "circumplex" in captured.out
    assert "box(2d)" in captured.out


def test_run_manifold_search_empty_query_passes_none(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    seen: list[Any] = []

    def fake_search(query: Any) -> list[dict[str, Any]]:
        seen.append(query)
        return []

    monkeypatch.setattr("drowse.store.hub.search_manifolds", fake_search)
    main(["pack", "search"])
    # Empty CLI default coerces to None for the backend (list by recency).
    assert seen == [None]
    out = capsys.readouterr().out
    assert "(no matches)" in out


def test_run_manifold_search_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    rows = [{
        "name": "circumplex", "namespace": "alice", "description": "moods",
        "tags": [], "node_count": 9, "domain_label": "box(2d)",
        "fit_mode": "authored", "tensor_models": [],
    }]
    monkeypatch.setattr("drowse.store.hub.search_manifolds", lambda q: rows)
    main(["pack", "search", "mood", "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert data[0]["name"] == "circumplex"
    assert data[0]["domain_label"] == "box(2d)"


def test_run_manifold_merge_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[dict[str, Any]] = []

    def fake_merge(target_ns: str, target_name: str, desc: str, *, sources: Any,
                   fit_mode: Any = None, force: bool = False) -> Path:
        calls.append({
            "ns": target_ns, "name": target_name, "desc": desc,
            "sources": sources, "fit_mode": fit_mode, "force": force,
        })
        return Path("/home/.drowse/manifolds/local/combined")

    monkeypatch.setattr("drowse.store.manifolds.merge_discover_manifolds", fake_merge)
    main([
        "manifold", "merge", "local/combined", "local/a", "alice/b",
        "--method", "spectral", "-f",
    ])
    assert len(calls) == 1
    c = calls[0]
    assert c["ns"] == "local" and c["name"] == "combined"
    assert c["sources"] == [("local", "a"), ("alice", "b")]
    assert c["fit_mode"] == "spectral"
    assert c["force"] is True
    captured = capsys.readouterr()
    assert "Merged manifold written to" in captured.out
    assert "fit local/combined" in captured.err


def test_merge_sources_resolve_existing_manifolds_cross_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression: bare merge *sources* were split as authoring names
    (bare → ``local/``), so ``merge mix emotions personas`` looked for
    ``local/emotions``.  Sources are existing manifolds and resolve
    cross-namespace, reaching bundled ``default/``."""
    _materialize_bundles_for_cli_test(monkeypatch, tmp_path)
    calls: list[Any] = []
    monkeypatch.setattr(
        "drowse.store.manifolds.merge_discover_manifolds",
        lambda *_a, sources, **_k: calls.append(sources) or tmp_path,
    )
    main(["manifold", "merge", "mix", "personas", "formal.casual"])
    assert calls == [[("default", "personas"), ("default", "formal.casual")]]
    with pytest.raises(SystemExit) as exc:
        main(["manifold", "merge", "mix", "personas", "not_installed_anywhere"])
    assert exc.value.code == 1


def test_run_manifold_merge_one_source_errors(monkeypatch: pytest.MonkeyPatch):
    # The parser allows >=1 positional source; the runner refuses <2.
    monkeypatch.setattr(
        "drowse.store.manifolds.merge_discover_manifolds",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        main(["manifold", "merge", "combined", "only-one"])
    assert ex.value.code == 2


def test_run_manifold_push_calls_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")

    calls: list[dict[str, Any]] = []

    def fake_push(folder_arg: Path, coord: str, *, private: bool = False,
                  model_scope: Any = None, variant: str = "all",
                  dry_run: bool = False) -> tuple[str, Any]:
        calls.append({
            "folder": folder_arg, "coord": coord, "private": private,
            "model_scope": model_scope, "variant": variant, "dry_run": dry_run,
        })
        return (f"https://huggingface.co/{coord}", "abcdef123456")

    monkeypatch.setattr("drowse.store.hub.push_manifold", fake_push)
    # -a override means resolve_target_coord doesn't need whoami().
    main([
        "pack", "push", "local/circumplex",
        "-a", "alice/circumplex", "-m", "google/gemma-3-4b-it",
        "--variant", "sae", "--private",
    ])
    assert len(calls) == 1
    c = calls[0]
    assert c["coord"] == "alice/circumplex"
    assert c["model_scope"] == "google/gemma-3-4b-it"
    assert c["variant"] == "sae"
    assert c["private"] is True
    assert c["dry_run"] is False
    captured = capsys.readouterr()
    assert captured.err.startswith("Pushing local/circumplex to alice/circumplex...\n")
    assert "Pushed alice/circumplex" in captured.out
    assert "abcdef123456" in captured.out


def test_push_resolves_a_bare_name_cross_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression: push addressed an existing manifold with the authoring
    split, so a bare bundled name looked under ``local/``."""
    _materialize_bundles_for_cli_test(monkeypatch, tmp_path)
    pushed: list[Path] = []
    monkeypatch.setattr(
        "drowse.store.hub.push_manifold",
        lambda folder, coord, **_k: pushed.append(folder) or (coord, None),
    )
    main(["pack", "push", "formal.casual", "-a", "alice/formal", "--dry-run"])
    assert [p.parent.name for p in pushed] == ["default"]


def test_run_manifold_push_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")

    monkeypatch.setattr(
        "drowse.store.hub.push_manifold",
        lambda f, coord, **k: (f"https://huggingface.co/{coord}", None),
    )
    main([
        "pack", "push", "local/circumplex",
        "-a", "alice/circumplex", "--dry-run",
    ])
    out = capsys.readouterr().out
    assert "Dry-run: would push alice/circumplex" in out


def test_run_manifold_push_missing_folder_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "drowse.store.hub.push_manifold",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        main(["pack", "push", "local/nope", "-a", "x/y"])
    assert ex.value.code == 1


def test_run_manifold_rm_calls_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[tuple[str, str]] = []

    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        calls.append((ns, name))
        return {"namespace": ns, "name": name, "source": "local",
                "removed": True, "rematerializes_on_restart": False}

    monkeypatch.setattr("drowse.store.manifolds.remove_manifold_folder", fake_rm)
    main(["pack", "rm", "local/mood", "-y"])
    assert calls == [("local", "mood")]
    out = capsys.readouterr().out
    assert "Removed local/mood" in out


def test_run_manifold_rm_local_refuses_without_yes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    """The irrecoverable case is guarded too — a ``local/`` manifold costs
    real extraction/fit time and never re-materializes."""
    monkeypatch.setattr(
        "drowse.store.manifolds.remove_manifold_folder",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        main(["pack", "rm", "local/mood"])
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "local/mood" in err
    assert "-y/--yes" in err


def test_run_manifold_rm_bundled_refuses_without_yes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "drowse.store.manifolds.remove_manifold_folder",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        main(["pack", "rm", "default/personas"])
    assert ex.value.code == 2


def test_run_manifold_rm_bundled_with_yes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        return {"namespace": ns, "name": name, "source": "bundled",
                "removed": True, "rematerializes_on_restart": True}

    monkeypatch.setattr("drowse.store.manifolds.remove_manifold_folder", fake_rm)
    main(["pack", "rm", "default/personas", "-y"])
    out = capsys.readouterr().out
    assert "Removed default/personas" in out
    assert "re-materializes" in out


def test_run_manifold_rm_missing_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_rm(ns: str, name: str) -> dict[str, Any]:
        raise FileNotFoundError("manifold local/nope not found")

    monkeypatch.setattr("drowse.store.manifolds.remove_manifold_folder", fake_rm)
    with pytest.raises(SystemExit) as ex:
        main(["pack", "rm", "local/nope", "-y"])
    assert ex.value.code == 1


def test_run_manifold_clear_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    calls: list[dict[str, Any]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope, "variant": variant})
        return 3

    monkeypatch.setattr("drowse.store.manifolds.clear_manifold_tensors", fake_clear)
    _materialize_bundles_for_cli_test(monkeypatch, tmp_path)
    # Bare ``personas`` resolves cross-namespace to the bundled ``default/personas``
    # (the only installed match) — the lifecycle verbs no longer hard-default
    # a bare name to ``local/``.
    main(["pack", "clear", "personas", "--variant", "raw"])
    assert calls == [
        {"ns": "default", "name": "personas", "model_scope": None, "variant": "raw"},
    ]
    out = capsys.readouterr().out
    assert "Deleted 3 files" in out


def test_run_manifold_clear_passes_model_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """``-m`` is now real — it threads through as ``model_scope`` (no warning)."""
    calls: list[dict[str, Any]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope, "variant": variant})
        return 1

    monkeypatch.setattr("drowse.store.manifolds.clear_manifold_tensors", fake_clear)
    _materialize_bundles_for_cli_test(monkeypatch, tmp_path)
    # Bare name resolves cross-namespace to bundled ``default/personas``.
    main(["pack", "clear", "personas", "-m", "foo/bar"])
    assert calls == [
        {"ns": "default", "name": "personas", "model_scope": "foo/bar", "variant": "all"},
    ]
    captured = capsys.readouterr()
    assert "Deleted 1 files" in captured.out
    assert "ignored" not in captured.err


def test_run_manifold_refresh_tiers(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    for tier, fragment in (("skipped", "nothing to refresh"),
                           ("bundled", "bundled"),
                           ("hf", "re-pulled from HF")):
        monkeypatch.setattr(
            "drowse.store.manifolds.refresh_manifold",
            lambda ns, name, *, model_scope=None, t=tier: t,
        )
        main(["pack", "refresh", "alice/circumplex"])
        out = capsys.readouterr().out
        assert fragment in out


def test_run_manifold_refresh_passes_model_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """``-m`` does a real scoped refresh now — threads ``model_scope``, no warning."""
    calls: list[dict[str, Any]] = []

    def fake_refresh(ns: str, name: str, *, model_scope: Any = None) -> str:
        calls.append({"ns": ns, "name": name, "model_scope": model_scope})
        return "scoped"

    monkeypatch.setattr("drowse.store.manifolds.refresh_manifold", fake_refresh)
    _materialize_bundles_for_cli_test(monkeypatch, tmp_path)
    # Bare name resolves cross-namespace to bundled ``default/personas``.
    main(["pack", "refresh", "personas", "-m", "foo/bar"])
    assert calls == [{"ns": "default", "name": "personas", "model_scope": "foo/bar"}]
    captured = capsys.readouterr()
    assert "foo/bar" in captured.out and "re-fits on next use" in captured.out
    assert "no effect" not in captured.err


def test_run_manifold_refresh_missing_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_refresh(ns: str, name: str, *, model_scope: Any = None) -> str:
        raise FileNotFoundError("not installed")

    monkeypatch.setattr("drowse.store.manifolds.refresh_manifold", fake_refresh)
    with pytest.raises(SystemExit) as ex:
        main(["pack", "refresh", "local/nope"])
    assert ex.value.code == 1


def test_run_manifold_transfer_calls_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir, safe_model_id
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    src_model = "google/gemma-3-4b-it"
    (folder / f"{safe_model_id(src_model)}.safetensors").write_bytes(b"x")
    from drowse.store.manifolds import TransferSourceProof

    source_proof = TransferSourceProof(
        tensor_name="source.safetensors",
        tensor_sha256="a" * 64,
        sidecar_sha256="b" * 64,
        layers=(14, 15),
    )
    monkeypatch.setattr(
        "drowse.store.manifolds.preflight_transfer_manifold",
        lambda *_args, **_kwargs: source_proof,
    )

    import torch
    from drowse.store.alignment import LayerAlignment

    fake_M = {
        layer: LayerAlignment(torch.eye(4), torch.eye(4), torch.zeros(4))
        for layer in (14, 15)
    }
    target_whitener = object()
    target_layer_means = {14: torch.zeros(4), 15: torch.zeros(4)}
    alignment_calls: list[dict[str, Any]] = []

    def fake_alignment(*_args: Any, **kwargs: Any):
        alignment_calls.append(kwargs)
        return (
            fake_M, {14: 0.9, 15: 0.8}, tmp_path / "alignment.safetensors",
            {"model_fingerprint": "src-fp"},
            {"model_fingerprint": "tgt-fp"},
            target_whitener,
            target_layer_means,
        )

    monkeypatch.setattr(
        "drowse.store.alignment.load_or_fit_transfer_alignment", fake_alignment,
    )

    calls: list[dict[str, Any]] = []

    def fake_transfer(folder_arg: Path, *, from_model: str, to_model: str,
                      alignment: Any, transfer_quality_estimate: Any = None,
                      source_model_fingerprint: Any = None,
                      target_model_fingerprint: Any = None,
                      whitener: Any = None, target_layer_means: Any = None,
                      force: bool = False,
                      expected_source_proof: Any = None) -> Path:
        calls.append({
            "folder": folder_arg, "from": from_model, "to": to_model,
            "layers": sorted(alignment.keys()), "quality": transfer_quality_estimate,
            "force": force, "whitener": whitener,
            "target_layer_means": target_layer_means,
            "source_proof": expected_source_proof,
        })
        return folder_arg / "Qwen__Qwen3-4B_from-google__gemma-3-4b-it.safetensors"

    monkeypatch.setattr("drowse.store.manifolds.transfer_manifold", fake_transfer)
    main([
        "manifold", "transfer", "local/circumplex",
        "--from", src_model, "--to", "Qwen/Qwen3-4B",
    ])
    assert len(calls) == 1
    c = calls[0]
    assert c["from"] == src_model
    assert c["to"] == "Qwen/Qwen3-4B"
    assert c["layers"] == [14, 15]
    assert c["whitener"] is target_whitener
    assert c["target_layer_means"] is target_layer_means
    assert alignment_calls[0]["requested_layers"] == [14, 15]
    assert c["source_proof"] is source_proof
    # Median of {0.9, 0.8} = 0.85.
    assert abs(c["quality"] - 0.85) < 1e-9
    out = capsys.readouterr().out
    assert "Transferred manifold local/circumplex" in out
    assert "2 shared" in out


def test_run_manifold_transfer_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir, safe_model_id
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    src_model = "src/model"
    (folder / f"{safe_model_id(src_model)}.safetensors").write_bytes(b"x")
    monkeypatch.setattr(
        "drowse.store.manifolds.preflight_transfer_manifold",
        lambda *_args, **_kwargs: None,
    )

    import torch
    target_whitener = object()
    monkeypatch.setattr(
        "drowse.store.alignment.load_or_fit_transfer_alignment",
        lambda *args, **kwargs: (
            {10: torch.eye(2)}, {10: 0.5}, tmp_path / "alignment.safetensors",
            {"model_fingerprint": "src-fp"},
            {"model_fingerprint": "tgt-fp"},
            target_whitener,
            {10: torch.zeros(2)},
        ),
    )
    monkeypatch.setattr(
        "drowse.store.manifolds.transfer_manifold",
        lambda f, **k: f / "out.safetensors",
    )
    main([
        "manifold", "transfer", "local/circumplex",
        "--from", src_model, "--to", "tgt/model", "-j",
    ])
    out = capsys.readouterr().out
    data = _json.loads(out)
    assert data["source_model"] == src_model
    assert data["target_model"] == "tgt/model"
    assert data["transferred_layers"] == [10]
    assert abs(data["median_transfer_quality"] - 0.5) < 1e-9


def test_run_manifold_transfer_retries_unproven_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A post-pair manifest failure must not make the CLI demand ``-f``."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir, safe_model_id, tensor_filename

    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    src_model = "src/model"
    tgt_model = "tgt/model"
    (folder / f"{safe_model_id(src_model)}.safetensors").write_bytes(b"source")
    monkeypatch.setattr(
        "drowse.store.manifolds.preflight_transfer_manifold",
        lambda *_args, **_kwargs: None,
    )

    import torch

    target_whitener = object()
    monkeypatch.setattr(
        "drowse.store.alignment.load_or_fit_transfer_alignment",
        lambda *args, **kwargs: (
            {0: torch.eye(2)}, {0: 0.5}, tmp_path / "alignment.safetensors",
            {"model_fingerprint": "src-fp"},
            {"model_fingerprint": "tgt-fp"},
            target_whitener,
            {0: torch.zeros(2)},
        ),
    )
    target = folder / tensor_filename(tgt_model, transferred_from=src_model)
    calls = 0

    def fail_once(folder_arg: Path, **_kwargs: Any) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Model the durable pair / missing manifest-proof crash window.
            target.write_bytes(b"unproven")
            target.with_suffix(".json").write_text("{}")
            raise RuntimeError("injected post-pair manifest failure")
        return target

    monkeypatch.setattr("drowse.store.manifolds.transfer_manifold", fail_once)
    argv = [
        "manifold", "transfer", "local/circumplex",
        "--from", src_model, "--to", tgt_model,
    ]
    with pytest.raises(RuntimeError, match="post-pair manifest"):
        main(argv)
    main(argv)

    assert calls == 2


def test_run_manifold_transfer_trusted_target_skips_alignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir

    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    monkeypatch.setattr(
        "drowse.store.manifolds.preflight_transfer_manifold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileExistsError("trusted transferred target exists")
        ),
    )
    monkeypatch.setattr(
        "drowse.store.alignment.load_or_fit_transfer_alignment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trusted target performed alignment work")
        ),
    )

    with pytest.raises(SystemExit) as exc:
        main([
            "manifold", "transfer", "local/circumplex",
            "--from", "src/model", "--to", "tgt/model",
        ])

    assert exc.value.code == 1


def test_run_manifold_transfer_missing_source_fit_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir
    folder = manifold_dir("local", "circumplex")
    folder.mkdir(parents=True)
    (folder / "manifold.json").write_text("{}")
    # No source safetensors → runner exits before any alignment work.
    with pytest.raises(SystemExit) as ex:
        main([
            "manifold", "transfer", "local/circumplex",
            "--from", "src/model", "--to", "tgt/model",
        ])
    assert ex.value.code == 1


# ---------------------------------------------------------------------------
# Cross-namespace bare-name lifecycle resolution (clear/refresh/rm/transfer)
# ---------------------------------------------------------------------------

def test_lifecycle_bare_name_not_found_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A bare name with no installed match exits 1, no backend call."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "drowse.store.manifolds.clear_manifold_tensors",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        main(["pack", "clear", "nope_not_installed"])
    assert ex.value.code == 1


def test_lifecycle_bare_name_resolves_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A bare name uniquely installed under ``local/`` resolves there."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)  # writes local/moodlite

    calls: list[tuple[str, str]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append((ns, name))
        return 0

    monkeypatch.setattr("drowse.store.manifolds.clear_manifold_tensors", fake_clear)
    main(["pack", "clear", "moodlite"])
    assert calls == [("local", "moodlite")]


def test_lifecycle_bare_name_ambiguous_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A bare name installed in two namespaces raises an ambiguity error (exit 2)."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.store.manifolds import create_manifold_folder
    domain_spec = {"type": "box", "axes": [{"min": -1.0, "max": 1.0, "periodic": False}]}
    nodes = [
        {"label": "low", "coords": [-1.0], "statements": ["s."]},
        {"label": "mid", "coords": [0.0], "statements": ["s."]},
        {"label": "high", "coords": [1.0], "statements": ["s."]},
    ]
    for ns in ("local", "alice"):
        create_manifold_folder(ns, "dup", "d", domain_spec, nodes)

    monkeypatch.setattr(
        "drowse.store.manifolds.clear_manifold_tensors",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call backend")),
    )
    with pytest.raises(SystemExit) as ex:
        main(["pack", "clear", "dup"])
    assert ex.value.code == 2


def test_lifecycle_explicit_ns_pins_without_walk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """An explicit ``ns/name`` pins verbatim — no existence pre-check, backend runs."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))  # nothing installed
    calls: list[tuple[str, str]] = []

    def fake_clear(ns: str, name: str, model_scope: Any = None, *, variant: str = "all") -> int:
        calls.append((ns, name))
        return 0

    monkeypatch.setattr("drowse.store.manifolds.clear_manifold_tensors", fake_clear)
    main(["pack", "clear", "alice/ghost"])
    assert calls == [("alice", "ghost")]


# ---------------------------------------------------------------------------
# ls -v / show -j changes
# ---------------------------------------------------------------------------

def _author_circumplex_lite(home: Path) -> Path:
    """Write a minimal authored 1-D box manifold folder under DROWSE_HOME.

    Two nodes on an open axis (min_nodes(1) = 3 isn't enforced until fit,
    and ``ManifoldFolder.load`` enforces it for authored folders — so use
    enough nodes to pass load).  Just enough to drive ls / show.
    """
    from drowse.store.manifolds import create_manifold_folder
    domain_spec = {"type": "box", "axes": [{"min": -1.0, "max": 1.0, "periodic": False}]}
    nodes = [
        {"label": "low", "coords": [-1.0], "statements": ["a statement here."]},
        {"label": "mid", "coords": [0.0], "statements": ["a statement here."]},
        {"label": "high", "coords": [1.0], "statements": ["a statement here."]},
    ]
    folder, _adv = create_manifold_folder(
        "local", "moodlite", "a tiny test mood manifold", domain_spec, nodes,
    )
    return folder


def test_run_manifold_ls_verbose_shows_description(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)
    from drowse.store import selectors
    selectors.invalidate()
    main(["pack", "ls", "--namespace", "local", "-v"])
    out = capsys.readouterr().out
    assert "local/moodlite" in out
    assert "a tiny test mood manifold" in out


def test_run_manifold_ls_non_verbose_hides_description(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)
    from drowse.store import selectors
    selectors.invalidate()
    main(["pack", "ls", "--namespace", "local"])
    out = capsys.readouterr().out
    assert "local/moodlite" in out
    assert "a tiny test mood manifold" not in out


def test_run_manifold_show_json_uses_summary_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    _author_circumplex_lite(tmp_path)
    from drowse.store import selectors
    selectors.invalidate()
    main(["pack", "show", "local/moodlite", "-j"])
    out = capsys.readouterr().out
    data = _json.loads(out)
    # manifold_summary's contract keys (shared with the server route).
    for key in ("namespace", "name", "description", "source", "fit_mode",
                "is_discover", "domain", "domain_label", "intrinsic_dim",
                "min_nodes", "node_count", "node_labels", "node_coords",
                "node_roles", "hyperparams", "fitted_models", "tensor_variants"):
        assert key in data, f"missing summary key {key!r}"
    assert data["namespace"] == "local"
    assert data["name"] == "moodlite"
    assert data["node_count"] == 3
    assert data["node_labels"] == ["low", "mid", "high"]


def test_run_manifold_show_json_matches_summary_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """CLI ``show -j`` output is byte-equivalent to ``manifold_summary``.

    ``show`` is an inspection surface, so it opts into the per-tensor
    ``fitted`` block; the light listing surfaces do not.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = _author_circumplex_lite(tmp_path)
    from drowse.store import selectors
    selectors.invalidate()
    from drowse.store.manifolds import manifold_summary
    expected = manifold_summary(folder, include_fits=True)
    assert "fitted" in expected
    main(["pack", "show", "local/moodlite", "-j"])
    out = capsys.readouterr().out
    assert _json.loads(out) == expected
