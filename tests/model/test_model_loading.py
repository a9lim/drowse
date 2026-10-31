"""Tests for the ``drowse.model`` loader: plan, fallbacks, extraction, identity.

These tests do not load real models — they intercept the AutoConfig and
AutoModelForCausalLM calls (and the Hub commit lookup) and inspect the
load_kwargs `load_model` passes.
"""
from __future__ import annotations

import json
import os

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from drowse.model import arch as arch_mod
from drowse.model import checkpoint as ckpt_mod
from drowse.model import compile as compile_mod
from drowse.model import fingerprint as fp_mod
from drowse.model import load as model_mod
_resolve_model_revision = fp_mod.resolve_model_revision
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _offline_hub_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The config stack is mocked here; keep the commit lookup offline too."""
    monkeypatch.setattr(fp_mod, "resolve_model_revision", lambda _model_id: None)


def test_resolve_model_revision_returns_the_commit_not_the_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub
    from huggingface_hub import ResolvedRevision

    calls: list[str] = []

    class _Api:
        def resolve_revision(self, repo_id: str, **_kwargs: Any) -> ResolvedRevision:
            calls.append(repo_id)
            return ResolvedRevision(resolved=_COMMIT, repo_id=repo_id)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    # ``ResolvedRevision`` is a ``str`` equal to the requested ref ("main");
    # only ``.resolved`` is the immutable commit.
    assert _resolve_model_revision("org/model") == _COMMIT
    assert calls == ["org/model"]


def test_resolve_model_revision_skips_the_hub_for_local_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    def _no_hub() -> None:
        raise AssertionError("a local checkpoint must not query the Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", _no_hub)
    assert _resolve_model_revision(str(tmp_path)) is None


def test_hub_source_fingerprint_pins_the_resolved_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preflight (no config) resolves the commit and reproduces the stamp."""
    cfg = _FakeConfig("qwen3")
    reads: list[dict[str, Any]] = []

    def _read(_model_id: str, **kwargs: Any) -> _FakeConfig:
        reads.append(kwargs)
        return cfg

    monkeypatch.setattr(fp_mod, "resolve_model_revision", lambda _m: "commit-a")
    with patch.object(fp_mod, "AutoConfig") as mock_cfg:
        mock_cfg.from_pretrained.side_effect = _read
        preflight = fp_mod.model_source_fingerprint(
            "org/model", device="cpu", parameter_dtype=torch.float32,
        )
    assert preflight is not None
    assert reads == [{"trust_remote_code": False, "revision": "commit-a"}]
    stamp = fp_mod.model_source_fingerprint(
        "org/model", device="cpu", config=cfg, revision="commit-a",
        parameter_dtype=torch.float32,
    )
    assert stamp == preflight
    assert fp_mod.model_source_fingerprint(
        "org/model", device="cpu", config=cfg, revision="commit-b",
        parameter_dtype=torch.float32,
    ) != preflight


@pytest.mark.parametrize("previous", [None, "0", "1"])
@pytest.mark.parametrize("fails", [False, True])
def test_mps_loading_is_synchronous_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch, previous: str | None, fails: bool,
):
    if previous is None:
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    else:
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", previous)
    plan = _plan(_FakeConfig("gpt_neox"), device="mps")
    loaded = _FakeModel("sdpa")

    def load(_model_id: str, **_kwargs: Any):
        assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"
        if fails:
            raise ValueError("bad weights")
        return loaded, {"missing_keys": []}

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_pretrained", load)
    if fails:
        with pytest.raises(ValueError, match="bad weights"):
            model_mod._load_with_fallbacks(plan)
    else:
        assert model_mod._load_with_fallbacks(plan) is loaded
    assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == previous


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_mps_loading_keeps_async_setting(monkeypatch: pytest.MonkeyPatch, device: str):
    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    with model_mod._model_load_environment(device):
        assert "HF_DEACTIVATE_ASYNC_LOAD" not in os.environ


def test_local_model_file_hash_cache_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp_mod._MODEL_FILE_HASH_CACHE.clear()
    monkeypatch.setattr(fp_mod, "_MODEL_FILE_HASH_CACHE_MAX", 2)
    for index in range(3):
        (tmp_path / f"file-{index}.json").write_text(str(index))

    rows = fp_mod._local_model_files(tmp_path)

    assert len(rows) == 3
    assert len(fp_mod._MODEL_FILE_HASH_CACHE) == 2
    assert str((tmp_path / "file-0.json").resolve()) not in (
        fp_mod._MODEL_FILE_HASH_CACHE
    )
    fp_mod._MODEL_FILE_HASH_CACHE.clear()


def test_empty_config_path_never_fingerprints_the_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanned: list[Path] = []
    monkeypatch.setattr(
        fp_mod, "_local_model_files",
        lambda path: scanned.append(path) or [],
    )

    assert fp_mod.model_source_fingerprint(
        "remote/model",
        device="cpu",
        config=_FakeConfig("qwen3"),
    ) is None
    assert scanned == []


class _FakeConfig(SimpleNamespace):
    """Minimal AutoConfig stand-in with knobs for model_type / text_config."""

    def __init__(self, model_type: str, text_model_type: str | None = None):
        super().__init__()
        self.model_type = model_type
        if text_model_type is not None:
            self.text_config = SimpleNamespace(
                model_type=text_model_type, _name_or_path=""
            )
        else:
            self.text_config = None


class _FakeModel(SimpleNamespace):
    """Stand-in for an HF model object — only what load_model touches."""

    def __init__(self, attn_impl: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=attn_impl)

    def requires_grad_(self, _flag: bool) -> _FakeModel:  # noqa: D401
        return self

    def train(self, _flag: bool) -> _FakeModel:
        return self

    def parameters(self):
        return iter([torch.zeros(1)])

    def named_parameters(self):
        return iter([])


def _captured_load_kwargs(device: str, model_type: str = "deepseek_v2"):
    """Run load_model with a mocked transformers stack; return the kwargs
    actually handed to AutoModelForCausalLM.from_pretrained."""
    cfg = _FakeConfig(model_type)
    captured: dict[str, Any] = {}

    def _fake_from_pretrained(model_id: str, **kwargs: Any):
        captured.update(kwargs)
        return _FakeModel(kwargs.get("attn_implementation", "sdpa")), {"missing_keys": []}

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.side_effect = _fake_from_pretrained
        model_mod.load_model("fake/repo", device=device)

    return captured


def test_mla_on_mps_forces_eager():
    """DeepSeek-V2/V3 on MPS must request eager — PyTorch MPS SDPA mishandles
    mismatched query/value head_dim, breaking o_proj at runtime."""
    for mt in ("deepseek_v2", "deepseek_v3"):
        kwargs = _captured_load_kwargs(device="mps", model_type=mt)
        assert kwargs["attn_implementation"] == "eager", (
            f"{mt} on MPS should force eager, got "
            f"{kwargs['attn_implementation']!r}"
        )


def test_mla_on_cpu_keeps_sdpa():
    """CPU SDPA correctly handles mismatched Eq/Ev — no need to downgrade."""
    kwargs = _captured_load_kwargs(device="cpu", model_type="deepseek_v2")
    assert kwargs["attn_implementation"] == "sdpa"


def test_non_mla_on_mps_keeps_sdpa():
    """The MLA carve-out is narrow: vanilla architectures still get sdpa."""
    kwargs = _captured_load_kwargs(device="mps", model_type="qwen3")
    assert kwargs["attn_implementation"] == "sdpa"


def test_mla_in_text_config_also_triggers():
    """A multimodal wrapper whose text_config is deepseek_v2 must also
    fall through to eager on MPS."""
    cfg = _FakeConfig(model_type="some_vlm", text_model_type="deepseek_v2")
    captured: dict[str, Any] = {}

    def _fake_from_pretrained(model_id: str, **kwargs: Any):
        captured.update(kwargs)
        return _FakeModel(kwargs.get("attn_implementation", "sdpa")), {"missing_keys": []}

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.side_effect = _fake_from_pretrained
        # The text_config carries deepseek_v2 — model_type is the wrapper.
        # extract_text_model takes the multimodal-wrapping branch whenever
        # the text_config's model_type is a causal-LM type, so load_model
        # would route through load_text_from_multimodal.  Stub it out — we
        # only care that attn_impl was already flipped to eager *before*
        # dispatch.
        with patch.object(model_mod, "load_text_from_multimodal") as mlm:
            mlm.return_value = _FakeModel("eager")
            model_mod.load_model("fake/repo", device="mps")

    # When the multimodal branch fires, AutoModelForCausalLM.from_pretrained
    # is never called — but the eager decision lives in load_kwargs which is
    # built either way.  Inspect the dtype-only call path: re-run with a
    # plain (non-multimodal) deepseek_v2 to confirm.
    kwargs = _captured_load_kwargs(device="mps", model_type="deepseek_v2")
    assert kwargs["attn_implementation"] == "eager"


def _captured_tokenizer_kwargs(model_id: str, model_type: str = "qwen3"):
    """Run load_model with a mocked transformers stack; return the kwargs
    actually handed to AutoTokenizer.from_pretrained."""
    cfg = _FakeConfig(model_type)
    captured: dict[str, Any] = {}

    def _fake_tok_from_pretrained(mid: str, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
    ):
        mock_tok.from_pretrained.side_effect = _fake_tok_from_pretrained
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.return_value = _FakeModel("sdpa"), {"missing_keys": []}
        model_mod.load_model(model_id, device="cpu")

    return captured


def test_mistral_regex_fix_passed_for_mistral():
    """HF-distributed Mistral checkpoints ship a buggy pre-tokenizer regex
    that mis-splits ~1% of tokens. ``load_model`` must pass
    ``fix_mistral_regex=True`` to AutoTokenizer for any mistral-family
    repo so the corrected regex from ``mistral_common`` is used.
    See https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84
    """
    for mid in (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai/Ministral-3-14B-Instruct-2512",
        "someorg/mistral-7b-finetune",
    ):
        kwargs = _captured_tokenizer_kwargs(mid)
        assert kwargs.get("fix_mistral_regex") is True, (
            f"{mid}: expected fix_mistral_regex=True in tokenizer kwargs, "
            f"got {kwargs!r}"
        )


def test_mistral_regex_fix_skipped_for_non_mistral():
    """The fix kwarg is mistral-tokenizer-specific. Don't pass it for
    other architectures — modern transformers may absorb unknown kwargs
    silently, but we shouldn't rely on that."""
    for mid in (
        "google/gemma-4-31b-it",
        "Qwen/Qwen3.6-27B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ):
        kwargs = _captured_tokenizer_kwargs(mid)
        assert "fix_mistral_regex" not in kwargs, (
            f"{mid}: fix_mistral_regex should NOT be set, got {kwargs!r}"
        )


# ---------------------------------------------------------------------------
# ``compile_with_probe`` — the session's one compile path.  Tests mock
# ``torch.compile`` so nothing actually compiles; they verify the wrapper is
# returned, the mode passes through, and a failing probe falls back to eager.
# ---------------------------------------------------------------------------


class _FakeCompiled:
    """Callable sentinel accepting the probe's prefill call shape."""

    _compiled_marker = True

    def __init__(self, model: Any) -> None:
        self._orig_mod = model

    def __call__(self, **fwd: Any) -> SimpleNamespace:
        n = fwd["input_ids"].shape[1]
        return SimpleNamespace(logits=torch.zeros(1, n, 1))


def _compile(*, device: str = "cpu", mode: str = "default", patch_precision: bool = True):
    invocations: list[dict[str, Any]] = []

    def _fake_compile(model: Any, **kwargs: Any) -> _FakeCompiled:
        invocations.append(kwargs)
        return _FakeCompiled(model)

    precision_context = (
        patch.object(torch, "set_float32_matmul_precision")
        if patch_precision else nullcontext()
    )
    with (
        patch.object(torch, "compile", side_effect=_fake_compile),
        patch.object(compile_mod, "run_compile_probes", return_value=None),
        precision_context,
    ):
        model = compile_mod.compile_with_probe(
            _FakeModel("sdpa"), SimpleNamespace(), device, mode=mode,
        )
    return invocations, model


def test_compile_with_probe_returns_the_compiled_wrapper():
    invocations, model = _compile(mode="reduce-overhead")
    assert [inv["mode"] for inv in invocations] == ["reduce-overhead"]
    assert getattr(model, "_compiled_marker", False) is True


def test_compile_probe_failure_falls_back_to_eager():
    """A compiled artifact that raises in the warmup probe (e.g. an inductor
    codegen bug) warns and returns the un-compiled model."""
    import warnings as _w

    class _BrokenCompiled(_FakeCompiled):
        def __call__(self, **fwd: Any) -> SimpleNamespace:
            raise RuntimeError("inductor codegen produced an invalid kernel")

    base = _FakeModel("sdpa")
    precision_before = torch.get_float32_matmul_precision()
    with (
        patch.object(torch, "compile", side_effect=lambda m, **k: _BrokenCompiled(m)),
        _w.catch_warnings(record=True) as caught,
    ):
        _w.simplefilter("always")
        ret = compile_mod.compile_with_probe(base, SimpleNamespace(), "cpu")

    assert ret is base
    assert any("torch.compile probe failed" in str(w.message) for w in caught)
    assert torch.get_float32_matmul_precision() == precision_before


def test_cuda_compile_enables_high_float32_matmul_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    precision: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precision.append)
    _compile(device="cuda", patch_precision=False)
    assert precision == ["high"]


def test_get_memory_gb_returns_zero_when_mps_backend_unavailable(monkeypatch: pytest.MonkeyPatch):
    """``device='mps'`` on a host without an MPS backend (e.g. Linux CI)
    must not crash.  The CUDA branch already gates on
    ``torch.cuda.is_available()``; the MPS branch must mirror that — the
    bare ``torch.mps.current_allocated_memory()`` call raises RuntimeError
    rather than AttributeError when the backend isn't actually present,
    so a plain ``except AttributeError`` is insufficient.
    """
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    # Make the underlying call raise to prove the gate fires before it.
    def _raise():
        raise RuntimeError("Cannot execute getCurrentAllocatedMemory() without MPS backend.")
    monkeypatch.setattr(torch.mps, "current_allocated_memory", _raise)
    assert model_mod.memory_gb("mps") == 0.0


# ---------------------------------------------------------------------------
# Multimodal text-model extraction.
#
# A multimodal checkpoint stores the text model's weights under a
# ``language_model.`` path segment and ships an unused vision tower.
# ``load_text_from_multimodal`` builds the text-only model on-device and
# streams just the text weights in, dropping the ``language_model.`` segment
# to recover the text model's own parameter names.  This both skips the
# vision tower and — critically on unified memory — avoids the ~2× peak of
# the standard CPU-stage-then-device-copy load (the >128 GB gemma-4-31B
# blowup).  Two real wire layouts exist; both must round-trip:
#
#   gemma-3 / gemma-4 : model.language_model.layers…   (segment nested)
#   mistral-3 / ministral : language_model.model.layers…  (leading prefix)
#
# These tests write a synthetic checkpoint with both text and fake vision
# keys, then assert the loader recovers exactly the text weights.
# ---------------------------------------------------------------------------


def _tiny_text_config(tie: bool):
    from transformers import AutoConfig as _AC
    return _AC.for_model(
        "llama", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        head_dim=8, max_position_embeddings=64, pad_token_id=0,
        bos_token_id=1, eos_token_id=2, tie_word_embeddings=tie,
    )


def _remap_to_layout(state_dict: dict[str, torch.Tensor], layout: str) -> dict[str, torch.Tensor]:
    """Re-key a text-only state dict into a multimodal wire layout."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if layout == "gemma":
            # model.X -> model.language_model.X ; lm_head.X -> language_model.lm_head.X
            nk = ("model.language_model." + k[len("model."):]
                  if k.startswith("model.") else "language_model." + k)
        elif layout == "mistral":
            nk = "language_model." + k  # leading prefix on everything
        else:  # pragma: no cover - guard
            raise ValueError(layout)
        out[nk] = v.clone()
    return out


def _write_synthetic_vlm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, layout: str, tie: bool):
    """Write a synthetic multimodal checkpoint (text + fake vision weights)
    and patch ``cached_file`` so the loader resolves it.  Returns the
    reference text-only model the checkpoint was derived from."""
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM as _AM
    from transformers.utils import SAFE_WEIGHTS_NAME

    text_cfg = _tiny_text_config(tie)
    torch.manual_seed(0)
    ref = _AM.from_config(text_cfg)
    remapped = _remap_to_layout(ref.state_dict(), layout)
    # Vision-tower / projector weights carry no ``language_model.`` segment
    # and must be skipped, not loaded into the text model.
    remapped["model.vision_tower.encoder.layers.0.weight"] = torch.randn(8, 8)
    remapped["model.multi_modal_projector.linear.weight"] = torch.randn(8, 8)

    weights_path = tmp_path / "model.safetensors"
    save_file(remapped, str(weights_path))

    def _fake_cached_file(model_id: str, filename: str, **kw: Any):
        # Single-file path: no index, weights resolve to our temp file.
        return str(weights_path) if filename == SAFE_WEIGHTS_NAME else None

    monkeypatch.setattr("transformers.utils.hub.cached_file", _fake_cached_file)
    return text_cfg, ref


@pytest.mark.parametrize(
    "layout,tie",
    [("gemma", True), ("mistral", False)],
    ids=["gemma-nested-tied", "mistral-prefix-untied"],
)
def test_load_text_from_multimodal_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, tie: bool):
    """The loaded text model's weights equal the reference's — for both
    wire layouts, and with the lm_head both tied (gemma) and separate
    (mistral, where ``language_model.lm_head.weight`` must be recovered)."""
    text_cfg, ref = _write_synthetic_vlm(
        tmp_path, monkeypatch, layout=layout, tie=tie,
    )
    loaded = ckpt_mod.load_text_from_multimodal(
        "fake/vlm", text_cfg, torch.float32, "cpu",
    )

    ref_params = dict(ref.named_parameters())
    got_params = dict(loaded.named_parameters())
    assert set(got_params) == set(ref_params), (
        f"param-name mismatch after {layout} extraction"
    )
    for name, ref_t in ref_params.items():
        assert torch.equal(got_params[name], ref_t), (
            f"{layout}: param {name!r} did not round-trip through the "
            f"language_model. rename"
        )
    # The vision tower never made it into the text model.
    assert not any("vision" in n for n in got_params), (
        "vision-tower weights leaked into the text model"
    )
    # And the extracted model is a usable causal LM.
    assert len(arch_mod.get_layers(loaded)) == text_cfg.num_hidden_layers


def test_load_text_from_multimodal_raises_when_no_text_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A checkpoint with no ``language_model.`` weights (everything skipped)
    must raise rather than silently return a randomly-initialized model."""
    from safetensors.torch import save_file
    from transformers.utils import SAFE_WEIGHTS_NAME

    text_cfg = _tiny_text_config(tie=True)
    weights_path = tmp_path / "model.safetensors"
    save_file(
        {"model.vision_tower.encoder.layers.0.weight": torch.randn(8, 8)},
        str(weights_path),
    )

    def _fake_cached_file(model_id: str, filename: str, **kw: Any):
        return str(weights_path) if filename == SAFE_WEIGHTS_NAME else None

    monkeypatch.setattr("transformers.utils.hub.cached_file", _fake_cached_file)

    with pytest.raises(ckpt_mod.NoTextWeightsExtracted):
        ckpt_mod.load_text_from_multimodal(
            "fake/vlm", text_cfg, torch.float32, "cpu",
        )


# ---------------------------------------------------------------------------
# Source-fingerprint parity: pre-load recompute == post-load stamp.
#
# ``_finalize_model`` stamps ``model_source_fingerprint`` from ``plan.config``
# *after* the weights loaded.  On the text-extraction path the load mutates
# that object in place (``_materialize_model`` fills
# ``text_config._name_or_path``; ``from_config`` writes the load dtype onto
# ``text_config``), so a naive pre-load recompute hashed a different config
# and every metadata-only fast path (manifold-fit preflight, lens-fit no-op,
# alignment ``_proven_sidecar``) failed closed on gemma-3/gemma-4/mistral-3.
# The fingerprint canonicalizes its config payload to the post-load view;
# these tests pin recompute == stamp end-to-end for both checkpoint shapes.
# ---------------------------------------------------------------------------


def _stub_tokenizer(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name_or_path=name, vocab_size=128, eos_token_id=2,
        added_tokens_encoder={},
    )


def _load_stamped(ckpt: Path, tok_name: str):
    """Real ``load_model`` on a local checkpoint; only the tokenizer is stubbed."""
    with patch.object(model_mod, "AutoTokenizer") as mock_tok:
        mock_tok.from_pretrained.return_value = _stub_tokenizer(tok_name)
        model, _ = model_mod.load_model(str(ckpt), device="cpu")
    return model


def test_source_fingerprint_recompute_matches_stamp_text_only(tmp_path: Path):
    from transformers import AutoModelForCausalLM as _AM

    text_cfg = _tiny_text_config(tie=True)
    torch.manual_seed(0)
    ckpt = tmp_path / "text-only"
    _AM.from_config(text_cfg).save_pretrained(ckpt)

    pre = fp_mod.model_source_fingerprint(str(ckpt), device="cpu")
    assert pre is not None
    model = _load_stamped(ckpt, "text-only-stub")
    assert getattr(model, "_drowse_source_fingerprint", None) == pre


def test_source_fingerprint_recompute_matches_stamp_multimodal(tmp_path: Path):
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM as _AM
    from transformers import LlavaConfig

    text_cfg = _tiny_text_config(tie=False)
    torch.manual_seed(0)
    ref = _AM.from_config(text_cfg)
    ckpt = tmp_path / "vlm"
    LlavaConfig(text_config=text_cfg.to_dict()).save_pretrained(ckpt)
    save_file(
        _remap_to_layout(ref.state_dict(), "mistral"),
        str(ckpt / "model.safetensors"),
    )

    pre = fp_mod.model_source_fingerprint(str(ckpt), device="cpu")
    assert pre is not None
    model = _load_stamped(ckpt, "vlm-stub")
    # The extraction path must actually have fired (and with it the in-place
    # plan-config mutations), else this asserts nothing.
    assert type(model).__name__ == "LlamaForCausalLM"
    assert getattr(model, "_drowse_source_fingerprint", None) == pre


# ---------------------------------------------------------------------------
# Routing: when does load_model prefer the text-only extraction path?
# ---------------------------------------------------------------------------


def _routing_calls(
    cfg: _FakeConfig, *, extractor_raises: bool = False,
    device: str = "cpu", quantize: str | None = None,
):
    """Run load_model with a mocked stack; return
    ``(text_extractor_called, from_pretrained_called)``."""
    fp_called = {"v": False}

    def _fake_from_pretrained(model_id: str, **kwargs: Any):
        fp_called["v"] = True
        return _FakeModel(kwargs.get("attn_implementation", "sdpa")), {"missing_keys": []}

    def _fake_extractor(*args: Any, **kwargs: Any) -> _FakeModel:
        if extractor_raises:
            raise ckpt_mod.NoTextWeightsExtracted("synthetic miss")
        return _FakeModel("sdpa")

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
        patch.object(model_mod, "load_text_from_multimodal",
                     side_effect=_fake_extractor) as mock_extract,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.side_effect = _fake_from_pretrained
        model_mod.load_model("fake/repo", device=device, quantize=quantize)

    return mock_extract.called, fp_called["v"]


def test_multimodal_routes_to_text_extraction_even_when_outer_accessible():
    """A gemma-style wrapper (outer ``gemma4`` *is* itself loadable,
    text ``gemma4_text`` also is) must still prefer text-only extraction —
    otherwise the standard load pulls the whole multimodal model into
    memory and double-buffers it on unified memory."""
    cfg = _FakeConfig(model_type="gemma4", text_model_type="gemma4_text")
    extractor_called, from_pretrained_called = _routing_calls(cfg)
    assert extractor_called, "gemma4 should route through text extraction"
    assert not from_pretrained_called, (
        "the full multimodal from_pretrained must not run when text "
        "extraction succeeds"
    )


def test_text_extraction_preserves_the_pinned_hub_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Official artifacts bind to the immutable base-model revision.

    The text-only multimodal loader constructs a new model config, so Drowse
    must carry the already-resolved revision onto that live config.
    """
    cfg = _FakeConfig(model_type="gemma3", text_model_type="gemma3_text")
    monkeypatch.setattr(
        fp_mod, "resolve_model_revision", lambda _m: "immutable-model-commit",
    )

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM"),
        patch.object(model_mod, "load_text_from_multimodal") as mock_extract,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_extract.return_value = _FakeModel("sdpa")
        model, _ = model_mod.load_model("google/gemma-3-4b-it", device="cpu")

    assert model.config._commit_hash == "immutable-model-commit"
    assert mock_extract.call_args.kwargs["revision"] == "immutable-model-commit"


def test_plain_text_model_uses_standard_load():
    """A non-multimodal config (no text_config) takes the standard load —
    there is nothing to extract."""
    cfg = _FakeConfig(model_type="qwen3")  # text_config is None
    extractor_called, from_pretrained_called = _routing_calls(cfg)
    assert not extractor_called
    assert from_pretrained_called


def test_quantized_multimodal_routes_to_standard_full_load():
    """Under bitsandbytes quantization the text-extraction path must be
    skipped — its shard streamer copies raw weights and cannot quantize, so
    routing through it would silently produce an unquantized model stamped
    as quantized.  The standard from_pretrained quantizes the full model."""
    cfg = _FakeConfig(model_type="gemma4", text_model_type="gemma4_text")
    extractor_called, from_pretrained_called = _routing_calls(
        cfg, device="cuda", quantize="4bit",
    )
    assert not extractor_called, (
        "quantized loads must not take the text-extraction path"
    )
    assert from_pretrained_called


def test_quantized_multimodal_without_standard_route_raises_actionable():
    """Ministral-as-Mistral3 + quantize: the composite is not registered with
    AutoModelForCausalLM, and the only working route — text extraction —
    cannot quantize.  HF's registry error must be rewrapped with the
    actionable cause."""
    cfg = _FakeConfig(model_type="mistral3", text_model_type="mistral")

    def _registry_error(model_id: str, **kwargs: Any):
        raise ValueError(
            "Unrecognized configuration class <class 'Mistral3Config'> for "
            "this kind of AutoModel: AutoModelForCausalLM."
        )

    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
        patch.object(model_mod, "load_text_from_multimodal") as mock_extract,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = cfg
        mock_model.from_pretrained.side_effect = _registry_error
        with pytest.raises(
            RuntimeError, match="does not support bitsandbytes",
        ):
            model_mod.load_model(
                "fake/ministral", device="cuda", quantize="4bit",
            )
    assert not mock_extract.called


def test_load_model_falls_back_to_full_load_on_no_text_weights():
    """If text extraction raises ``_NoTextWeightsExtracted`` (unexpected
    weight layout), load_model must fall back to the standard full load
    rather than propagate the error or ship random init."""
    cfg = _FakeConfig(model_type="some_vlm", text_model_type="qwen3")
    extractor_called, from_pretrained_called = _routing_calls(
        cfg, extractor_raises=True,
    )
    assert extractor_called, "extraction should be attempted first"
    assert from_pretrained_called, (
        "a failed extraction must fall back to the standard full load"
    )


# ---------------------------------------------------------------------------
# The load-plan seam: decisions without weight I/O
# ---------------------------------------------------------------------------


def _plan(cfg: _FakeConfig, **kwargs: Any):
    """Resolve a load plan against a mocked AutoConfig — reads no weights."""
    with patch.object(model_mod, "AutoConfig") as mock_cfg:
        mock_cfg.from_pretrained.return_value = cfg
        return model_mod._resolve_load_plan(
            "fake/repo",
            quantize=kwargs.pop("quantize", None),
            device=kwargs.pop("device", "cpu"),
            dtype=kwargs.pop("dtype", None),
        )


def test_plan_resolves_decisions_without_loading_weights():
    """Every ``load_model`` decision is inspectable before any weight read.

    ``AutoModelForCausalLM`` is deliberately not patched here: reaching it
    would fail the test, which is the point — the plan step must not load.
    """
    plan = _plan(_FakeConfig("qwen3"), device="cpu")
    assert plan.model_id == "fake/repo"
    assert plan.device == "cpu"
    assert plan.dtype is torch.float32
    assert plan.load_kwargs["attn_implementation"] == "sdpa"
    assert plan.load_kwargs["device_map"] == {"": "cpu"}
    assert plan.extract_text_model is False


def test_plan_flags_text_extraction_for_supported_submodel():
    plan = _plan(_FakeConfig("some_vlm", text_model_type="qwen3"), device="cpu")
    assert plan.extract_text_model is True
    assert plan.text_config is not None

    # A text_config drowse has no accessor for stays a full-model load.
    plan = _plan(_FakeConfig("some_vlm", text_model_type="not_a_family"))
    assert plan.extract_text_model is False


def test_plan_forces_eager_for_mla_on_mps():
    """The MLA/MPS carve-out is a plan decision, testable on its own."""
    for model_type in ("deepseek_v2", "deepseek_v3"):
        plan = _plan(_FakeConfig(model_type), device="mps")
        assert plan.load_kwargs["attn_implementation"] == "eager"
    # Narrow: a vanilla architecture on MPS keeps sdpa, and MLA on CPU too.
    assert _plan(
        _FakeConfig("qwen3"), device="mps",
    ).load_kwargs["attn_implementation"] == "sdpa"
    assert _plan(
        _FakeConfig("deepseek_v2"), device="cpu",
    ).load_kwargs["attn_implementation"] == "sdpa"


def test_plan_drops_quantization_off_cuda():
    with pytest.warns(UserWarning, match="requires CUDA"):
        plan = _plan(_FakeConfig("qwen3"), device="mps", quantize="4bit")
    assert plan.quantize is None
    assert "quantization_config" not in plan.load_kwargs
    assert plan.load_kwargs["dtype"] is plan.dtype


def test_plan_quantize_skips_text_extraction():
    """Quantization flips a text-extractable composite onto the standard
    quantized full-model path — the extraction streamer cannot quantize."""
    cfg = _FakeConfig("gemma3", text_model_type="gemma3_text")
    plan = _plan(cfg, device="cuda", quantize="4bit")
    assert plan.extract_text_model is False
    assert "quantization_config" in plan.load_kwargs

    # Unquantized on the same config still prefers extraction.
    plan = _plan(cfg, device="cuda")
    assert plan.extract_text_model is True


def test_plan_pins_every_read_to_the_resolved_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Transformers >=5.18 configs carry no ``_commit_hash``: the pin comes
    # from the Hub lookup alone, and the probe config read is pinned too.
    monkeypatch.setattr(fp_mod, "resolve_model_revision", lambda _m: "deadbeef")
    with patch.object(model_mod, "AutoConfig") as mock_cfg:
        mock_cfg.from_pretrained.return_value = _FakeConfig("qwen3")
        plan = model_mod._resolve_load_plan(
            "fake/repo", quantize=None, device="cpu", dtype=None,
        )
    assert plan.revision == "deadbeef"
    assert plan.pin_kwargs == {"revision": "deadbeef"}
    assert plan.load_kwargs["revision"] == "deadbeef"
    reads = mock_cfg.from_pretrained.call_args_list
    assert reads and all(read.kwargs.get("revision") == "deadbeef" for read in reads)


def test_plan_fails_open_when_the_commit_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unreachable(_model_id: str) -> str:
        raise RuntimeError("Hub unreachable and nothing cached")

    monkeypatch.setattr(fp_mod, "resolve_model_revision", _unreachable)
    plan = _plan(_FakeConfig("qwen3"))
    assert plan.revision is None
    assert plan.pin_kwargs == {}
    assert "revision" not in plan.load_kwargs


def test_standard_load_stamps_the_resolved_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live config's ``_commit_hash`` slot is Drowse's stamp, not HF's."""
    monkeypatch.setattr(fp_mod, "resolve_model_revision", lambda _m: _COMMIT)
    with (
        patch.object(model_mod, "AutoTokenizer") as mock_tok,
        patch.object(model_mod, "AutoConfig") as mock_cfg,
        patch.object(model_mod, "AutoModelForCausalLM") as mock_model,
    ):
        mock_tok.from_pretrained.return_value = SimpleNamespace()
        mock_cfg.from_pretrained.return_value = _FakeConfig("qwen3")
        mock_model.from_pretrained.return_value = _FakeModel("sdpa"), {"missing_keys": []}
        model, _ = model_mod.load_model("org/model", device="cpu")

    assert model.config._commit_hash == _COMMIT
    assert mock_model.from_pretrained.call_args.kwargs["revision"] == _COMMIT
    assert mock_tok.from_pretrained.call_args.kwargs["revision"] == _COMMIT


def test_plan_adds_the_mistral_tokenizer_fix_by_model_id():
    with patch.object(model_mod, "AutoConfig") as mock_cfg:
        mock_cfg.from_pretrained.return_value = _FakeConfig("mistral")
        mistral = model_mod._resolve_load_plan(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            quantize=None, device="cpu", dtype=None,
        )
        other = model_mod._resolve_load_plan(
            "Qwen/Qwen3-4B", quantize=None, device="cpu", dtype=None,
        )
    assert mistral.tokenizer_kwargs["fix_mistral_regex"] is True
    assert "fix_mistral_regex" not in other.tokenizer_kwargs


def test_load_with_fallbacks_narrows_attn_then_dtype_in_place():
    """The retry ladder rewrites the plan's kwargs so retries compose."""
    plan = _plan(_FakeConfig("qwen3"), device="cpu")
    calls: list[dict[str, Any]] = []

    def _first_attn_then_ok(model_id: str, **kwargs: Any):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise ValueError("model does not support an attention implementation")
        return _FakeModel(kwargs["attn_implementation"]), {"missing_keys": []}

    with patch.object(model_mod, "AutoModelForCausalLM") as mock_model:
        mock_model.from_pretrained.side_effect = _first_attn_then_ok
        model_mod._load_with_fallbacks(plan)

    assert calls[0]["attn_implementation"] == "sdpa"
    assert calls[1]["attn_implementation"] == "eager"
    assert plan.load_kwargs["attn_implementation"] == "eager"


# ---------------------------------------------------------------------------
# Checkpoint path safety and sharded remap loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shard", ["../outside.safetensors", "/tmp/outside.safetensors", "C:/weights.safetensors", "sub\\weights.safetensors", "./weights.safetensors", "sub//weights.safetensors", "weights.bin", "", None, 1])
def test_index_rejects_unsafe_shard_names_before_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shard: Any):
    index = tmp_path / SAFE_WEIGHTS_INDEX_NAME
    index.write_text(json.dumps({"weight_map": {"language_model.model.weight": shard}}))
    calls = []

    def cached_file(model: str, name: str, **kwargs: Any):
        calls.append(name)
        if name == SAFE_WEIGHTS_INDEX_NAME:
            return str(index)
        pytest.fail("unsafe index shard reached file resolution")

    monkeypatch.setattr("transformers.utils.hub.cached_file", cached_file)
    with pytest.raises(ValueError, match="shard"):
        ckpt_mod.load_text_from_multimodal("fixture/model", _tiny_text_config(False), torch.float32, "cpu")
    assert calls == [SAFE_WEIGHTS_INDEX_NAME]


def test_local_index_traversal_cannot_read_an_external_checkpoint(tmp_path: Path):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    save_file(_remap_to_layout(reference.state_dict(), "mistral"), str(tmp_path / "outside.safetensors"))
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps({
        "weight_map": {name: "../outside.safetensors" for name in reference.state_dict()},
    }))
    with pytest.raises(ValueError, match="shard"):
        ckpt_mod.load_text_from_multimodal(str(model_root), config, torch.float32, "cpu")


@pytest.mark.parametrize("kind", ["index", "shard"])
def test_checkpoint_rejects_fifo_without_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation unavailable")
    path = tmp_path / (SAFE_WEIGHTS_INDEX_NAME if kind == "index" else SAFE_WEIGHTS_NAME)
    os.mkfifo(path)
    def cached_file(model: str, name: str, **kwargs: Any):
        return str(path) if name == path.name else None
    monkeypatch.setattr("transformers.utils.hub.cached_file", cached_file)
    with pytest.raises(ValueError, match="regular file"):
        ckpt_mod.load_text_from_multimodal("fixture/model", _tiny_text_config(False), torch.float32, "cpu")


@pytest.mark.parametrize("damage", ["missing", "broadcast"])
def test_partial_or_broadcast_checkpoint_is_not_a_loaded_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    state = _remap_to_layout(reference.state_dict(), "mistral")
    key = "language_model.model.embed_tokens.weight"
    if damage == "missing":
        del state[key]
    else:
        state[key] = state[key][:1].clone()
    path = tmp_path / SAFE_WEIGHTS_NAME
    save_file(state, str(path))
    monkeypatch.setattr("transformers.utils.hub.cached_file", lambda model, name, **kwargs: str(path) if name == SAFE_WEIGHTS_NAME else None)
    with pytest.raises(ValueError, match="missing|shape"):
        ckpt_mod.load_text_from_multimodal("fixture/model", config, torch.float32, "cpu")


@pytest.mark.parametrize("header", [
    {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}},
    {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [-1, 7]}},
    {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 20]}},
    {"weight": {"dtype": "F32", "shape": [True, 2], "data_offsets": [0, 8]}},
    {"weight": {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}},
    {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
])
def test_manual_header_reader_rejects_malformed_layout(tmp_path: Path, header: dict[str, Any]):
    raw = json.dumps(header).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * 8)
    with pytest.raises(ValueError, match="safetensors"):
        ckpt_mod.read_safetensors_header(str(path))


def test_manual_header_reader_bounds_length_before_reading(tmp_path: Path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes((2**40).to_bytes(8, "little"))
    with pytest.raises(ValueError, match="header"):
        ckpt_mod.read_safetensors_header(str(path))


def test_tied_checkpoint_alias_and_nonpersistent_buffers_are_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _tiny_text_config(True)
    original_factory = model_mod.AutoModelForCausalLM.from_config
    reference = original_factory(config)
    state = _remap_to_layout(reference.state_dict(), "gemma")
    del state["language_model.lm_head.weight"]
    weights = tmp_path / SAFE_WEIGHTS_NAME
    save_file(state, str(weights))

    def with_runtime_buffer(*args: Any, **kwargs: Any):
        model = original_factory(*args, **kwargs)
        model.register_buffer("runtime_cache", torch.ones(3), persistent=False)
        return model

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_config", with_runtime_buffer)
    monkeypatch.setattr("transformers.utils.hub.cached_file", lambda model, name, **kwargs: str(weights) if name == SAFE_WEIGHTS_NAME else None)
    loaded = ckpt_mod.load_text_from_multimodal("fixture/model", config, torch.float32, "cpu")
    assert torch.equal(loaded.runtime_cache, torch.ones(3))
    for name, tensor in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight


def test_tensor_reader_rejects_post_header_truncation(tmp_path: Path):
    path = tmp_path / SAFE_WEIGHTS_NAME
    save_file({"weight": torch.arange(4)}, str(path))
    header, offset = ckpt_mod.read_safetensors_header(str(path))
    with path.open("r+b") as handle:
        handle.truncate(offset + 8)
    fd = ckpt_mod.open_uncached(str(path))
    try:
        with pytest.raises(ValueError, match="truncated"):
            ckpt_mod.read_st_tensor(fd, header, offset, "weight")
    finally:
        os.close(fd)


def test_scalar_and_empty_tensor_headers_remain_valid(tmp_path: Path):
    path = tmp_path / SAFE_WEIGHTS_NAME
    values = {"empty": torch.empty(0, 3), "scalar": torch.tensor(2.0)}
    save_file(values, str(path))
    header, offset = ckpt_mod.read_safetensors_header(str(path))
    fd = ckpt_mod.open_uncached(str(path))
    try:
        for name, expected in values.items():
            assert torch.equal(ckpt_mod.read_st_tensor(fd, header, offset, name), expected)
    finally:
        os.close(fd)


def test_valid_nested_shards_and_hf_blob_symlinks_preserve_exact_weights(tmp_path: Path):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    state = _remap_to_layout(reference.state_dict(), "mistral")
    model_root = tmp_path / "snapshot"
    shards = model_root / "shards"
    shards.mkdir(parents=True)
    blob = tmp_path / "blob"
    save_file(state, str(blob))
    (shards / "weights.safetensors").symlink_to(blob)
    (model_root / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps({
        "weight_map": {name: "shards/weights.safetensors" for name in state},
    }))
    loaded = ckpt_mod.load_text_from_multimodal(str(model_root), config, torch.float32, "cpu")
    for name, tensor in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64, torch.complex64, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz])
def test_ignored_vision_dtype_does_not_restrict_text_loading(tmp_path: Path, dtype: torch.dtype):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    state = _remap_to_layout(reference.state_dict(), "mistral")
    state["model.vision_tower.buffer"] = torch.zeros(2, dtype=dtype)
    save_file(state, str(tmp_path / SAFE_WEIGHTS_NAME))

    loaded = ckpt_mod.load_text_from_multimodal(str(tmp_path), config, torch.float32, "cpu")

    for name, tensor in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name


@pytest.mark.parametrize("dtype,bits", [
    ("F4", 4), ("F6_E2M3", 6), ("F6_E3M2", 6), ("F8_E8M0", 8),
    ("F8_E4M3FNUZ", 8), ("F8_E5M2FNUZ", 8), ("U16", 16),
    ("U32", 32), ("U64", 64), ("C64", 64),
])
def test_valid_unselected_dtypes_validate_but_do_not_silently_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: str, bits: int):
    path = tmp_path / SAFE_WEIGHTS_NAME
    raw = json.dumps({"vision": {"dtype": dtype, "shape": [8], "data_offsets": [0, bits]}}).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(bits))
    with safe_open(str(path), framework="pt") as archive:
        assert archive.keys() == ["vision"]
    header, offset = ckpt_mod.read_safetensors_header(str(path))

    def no_read(*args: Any):
        pytest.fail("unsupported selected dtype was read before rejection")

    monkeypatch.setattr(os, "pread", no_read)
    fd = ckpt_mod.open_uncached(str(path))
    try:
        with pytest.raises(ValueError, match="unsupported.*selected tensor"):
            ckpt_mod.read_st_tensor(fd, header, offset, "vision")
    finally:
        os.close(fd)


@pytest.mark.parametrize("shape,dtype", [([2**64, 0], "F32"), ([2**32] * 256, "F32"), ([1], "F4")])
def test_shape_validation_rejects_unbounded_or_misaligned_dimensions(tmp_path: Path, shape: list[int], dtype: str):
    header = {"vision": {"dtype": dtype, "shape": shape, "data_offsets": [0, 1]}}
    raw = json.dumps(header).encode()
    path = tmp_path / SAFE_WEIGHTS_NAME
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0")
    with pytest.raises(ValueError, match="safetensors"):
        ckpt_mod.read_safetensors_header(str(path))


def test_outer_head_alone_does_not_identify_a_supported_text_layout(tmp_path: Path):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    save_file(reference.state_dict(), str(tmp_path / SAFE_WEIGHTS_NAME))
    with pytest.raises(ckpt_mod.NoTextWeightsExtracted):
        ckpt_mod.load_text_from_multimodal(str(tmp_path), config, torch.float32, "cpu")


@pytest.mark.parametrize("architecture", ["mistral3", "gemma3"])
@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_actual_multimodal_wrapper_preserves_head_weights_and_logits(tmp_path: Path, architecture: str, tie: bool, sharded: bool):
    from transformers import AutoConfig, AutoModelForImageTextToText

    text_config = {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
        "vocab_size": 32, "tie_word_embeddings": tie, "pad_token_id": 0,
    }
    vision_config = {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "image_size": 16, "patch_size": 8,
    }
    extra = {"mm_tokens_per_image": 4} if architecture == "gemma3" else {}
    config = AutoConfig.for_model(
        architecture, text_config=text_config, vision_config=vision_config,
        tie_word_embeddings=tie, **extra,
    )
    reference = AutoModelForImageTextToText.from_config(config).eval()
    state = {name: value.clone() for name, value in reference.state_dict().items()}
    assert "lm_head.weight" in state
    if tie:
        del state["lm_head.weight"]
    if sharded:
        body = {name: value for name, value in state.items() if name != "lm_head.weight"}
        head = {name: value for name, value in state.items() if name == "lm_head.weight"}
        save_file(body, str(tmp_path / "body.safetensors"))
        save_file(head, str(tmp_path / "head.safetensors"))
        (tmp_path / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps({
            "weight_map": {name: "head.safetensors" if name in head else "body.safetensors" for name in state},
        }))
    else:
        save_file(state, str(tmp_path / SAFE_WEIGHTS_NAME))

    loaded = ckpt_mod.load_text_from_multimodal(str(tmp_path), config.text_config, torch.float32, "cpu").eval()

    reference_state = reference.state_dict()
    for name, value in loaded.state_dict().items():
        source = "model.language_model." + name[len("model."):] if name.startswith("model.") else name
        assert torch.equal(value, reference_state[source]), name
    assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tie
    with torch.no_grad():
        tokens = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(loaded(tokens).logits, reference(tokens).logits, rtol=0, atol=0)


def _generic_plan(folder: Path, config: Any) -> model_mod.LoadPlan:
    return model_mod.LoadPlan(
        model_id=str(folder), device="cpu", dtype=torch.float32, quantize=None,
        revision=None, pin_kwargs={}, tokenizer_kwargs={}, config=config,
        text_config=None, extract_text_model=False,
        load_kwargs={"dtype": torch.float32, "device_map": {"": "cpu"},
                     "attn_implementation": "eager", "trust_remote_code": False,
                     "local_files_only": True},
    )


@pytest.mark.parametrize("tie", [False, True])
def test_generic_load_preserves_exact_weights_logits_and_runtime_buffers(tmp_path: Path, tie: bool):
    config = _tiny_text_config(tie)
    reference = model_mod.AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    state = reference.state_dict()
    state["training_only.unused"] = torch.ones(1)
    reference.save_pretrained(str(tmp_path), state_dict=state, safe_serialization=True)

    loaded = model_mod._materialize_model(_generic_plan(tmp_path, config)).eval()

    for name, value in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value), name
    for name, value in reference.named_buffers():
        assert torch.equal(dict(loaded.named_buffers())[name], value), name
    assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tie
    with torch.no_grad():
        tokens = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(loaded(tokens).logits, reference(tokens).logits, rtol=0, atol=0)


@pytest.mark.parametrize("tie", [False, True])
def test_generic_load_rejects_missing_required_weights(tmp_path: Path, tie: bool):
    config = _tiny_text_config(tie)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    state = reference.state_dict()
    if tie:
        del state["model.embed_tokens.weight"]
    del state["lm_head.weight"]
    reference.save_pretrained(str(tmp_path), state_dict=state, safe_serialization=True)
    with pytest.raises(model_mod._IncompleteCheckpointError, match="missing model parameters"):
        model_mod._materialize_model(_generic_plan(tmp_path, config))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_missing_parameter_error_never_uses_backend_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str):
    model = torch.nn.Module()
    model.register_parameter("CONVERSION", torch.nn.Parameter(torch.ones(1)))
    plan = _generic_plan(tmp_path, _tiny_text_config(False))
    plan.device = device
    attempts = []

    def from_pretrained(*args: Any, **kwargs: Any):
        attempts.append(kwargs)
        return model, {"missing_keys": ["CONVERSION"]}

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    with pytest.raises(model_mod._IncompleteCheckpointError):
        model_mod._materialize_model(plan)
    assert len(attempts) == 1
    assert attempts[0]["output_loading_info"] is True


@pytest.mark.parametrize("kind", ["attention", "dtype", "device", "quantized"])
def test_checked_generic_load_preserves_backend_options_and_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    model = torch.nn.Linear(2, 2)
    plan = _generic_plan(tmp_path, _tiny_text_config(False))
    plan.load_kwargs.update(dtype=torch.bfloat16, attn_implementation="sdpa", revision="pinned")
    if kind == "device":
        plan.device = "mps"
        plan.load_kwargs["device_map"] = {"": "mps"}
    if kind == "quantized":
        plan.quantize = "4bit"
        plan.load_kwargs["quantization_config"] = object()
    attempts = []
    moves = []

    def from_pretrained(*args: Any, **kwargs: Any):
        attempts.append(dict(kwargs))
        if kind == "attention" and len(attempts) == 1:
            raise ValueError("model does not support an attention implementation")
        if kind == "dtype" and len(attempts) == 1:
            raise RuntimeError("unsupported backend dtype")
        if kind == "device" and len(attempts) <= 2:
            raise RuntimeError("CONVERSION backend failure")
        return model, {"missing_keys": ["optional_buffer"], "unexpected_keys": ["training_only"]}

    def to_device(device: str):
        moves.append(device)
        return model

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(model, "to", to_device)
    assert model_mod._materialize_model(plan) is model
    assert all(row["output_loading_info"] and row["revision"] == "pinned" for row in attempts)
    if kind == "attention":
        assert [row["attn_implementation"] for row in attempts] == ["sdpa", "eager"]
    elif kind == "dtype":
        assert [row["dtype"] for row in attempts] == [torch.bfloat16, torch.float32]
    elif kind == "device":
        assert [row["device_map"] for row in attempts] == [{"": "mps"}, {"": "mps"}, {"": "cpu"}]
        assert moves == ["mps"]
    else:
        assert len(attempts) == 1
        assert attempts[0]["quantization_config"] is plan.load_kwargs["quantization_config"]


# ---------------------------------------------------------------------------
# Remote-code trust
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trusted", [False, True])
def test_model_configuration_requires_explicit_code_trust(tmp_path: Path, trusted: bool) -> None:
    marker = tmp_path / "configuration-executed"
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "intermediate_size": 16,
        "auto_map": {"AutoConfig": "configuration_fixture.FixtureConfig"},
    }))
    (tmp_path / "configuration_fixture.py").write_text(
        "from pathlib import Path\n"
        "from transformers import LlamaConfig\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class FixtureConfig(LlamaConfig):\n    pass\n"
    )
    plan = model_mod._resolve_load_plan(str(tmp_path), quantize=None, device="cpu", dtype=None, trust_remote_code=trusted)
    assert plan.config.model_type == "llama"
    assert marker.exists() is trusted
    assert plan.tokenizer_kwargs["trust_remote_code"] is trusted

@pytest.mark.parametrize(("explicit", "environment", "trusted"), [(None, "", False), (None, "true", True), (False, "1", False), (True, "", True), (None, "false", False)])
def test_tokenizer_code_trust_is_explicit_and_overrides_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool | None, environment: str, trusted: bool) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    marker = tmp_path / "tokenizer-executed"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2}))
    Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")).save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",
        "auto_map": {"AutoTokenizer": [None, "tokenization_fixture.FixtureTokenizer"]},
    }))
    (tmp_path / "tokenization_fixture.py").write_text(
        "from pathlib import Path\n"
        "from transformers import PreTrainedTokenizerFast\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class FixtureTokenizer(PreTrainedTokenizerFast):\n    pass\n"
    )
    monkeypatch.setattr(model_mod, "_materialize_model", lambda plan: object())
    monkeypatch.setattr(model_mod, "_finalize_model", lambda model, *args, **kwargs: model)
    monkeypatch.setenv("DROWSE_TRUST_REMOTE_CODE", environment)
    _, tokenizer = model_mod.load_model(str(tmp_path), device="cpu", trust_remote_code=explicit)
    assert tokenizer.encode("hello") == [1]
    assert marker.exists() is trusted

@pytest.mark.parametrize("value", ["true", 1, [], {}])
def test_model_code_trust_rejects_ambiguous_non_booleans(value: Any) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        model_mod.load_model("unused/model", trust_remote_code=value)
