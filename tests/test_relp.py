"""CPU tests for the RelP (R-lens) backward rules and the dual-name fit.

``lens/relp.py`` keeps the estimator's forward values bit-identical —
custom autograd Functions return each module's own output untouched and
only substitute the backward factor — while reading the Jacobian through
the LRP-modified backward graph.  The sparse jlens toy (no RMSNorm /
gated-MLP children) exercises the structural rejection; a dense
llama-shaped toy defined here drives the detached backward factors, the
first-forward self-verification, the reversible instance patching, and the
session-level ``fit_jlens`` name/policy separation (``local/default``
beside ``local/relp``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from drowse.cli.runners import _try_lens_fit_noop_preflight
from drowse.errors import RelpUnsupportedError
from drowse.lens.jlens import JacobianLens
from drowse.lens.relp import (
    RELP_RULES,
    _act_backward_factor,
    _relp_norm_forward,
    relp_backward_rules,
)
from drowse.lens.store import (
    lens_checkpoint_paths,
    lens_estimator_policy,
    lens_paths,
    load_lens_checkpoint_sidecar,
    load_local_lens,
    load_local_lens_sidecar,
    remove_lens,
    save_lens,
    save_lens_checkpoint_accumulator,
)
from drowse.lens.sources import (
    LOCAL_SOURCE_PREFIX,
    list_lens_sources,
    load_active_lens_source,
    use_lens_source,
)
from tests._jlens_toys import TOY_VOCAB, ToyCausalLM
from tests.test_jlens_session import _PROMPTS, _StubSession

_DENSE_MODEL_ID = "toy/relp-dense-model"
DENSE_D = 6


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))


# ---------------------------------------------------------------------------
# Dense toy: llama-shaped blocks the RelP rules accept
# ---------------------------------------------------------------------------


class ToyRMSNorm(nn.Module):
    """Llama-style plain-weight RMSNorm; class name is the RelP norm marker."""

    def __init__(self, d: int, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(1.0 + 0.1 * torch.randn(d, generator=gen))
        self.variance_epsilon = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (
            x
            * torch.rsqrt(
                x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
            )
        )


class ToyGatedMLP(nn.Module):
    """Standard gated MLP shape: gate/up d->2d, down 2d->d, SiLU gate.

    The forward is exactly the canonical composition — the RelP rewrite
    self-verifies against ``type(self).forward`` bit-for-bit.
    """

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.gate_proj = nn.Linear(d, 2 * d, bias=False)
        self.up_proj = nn.Linear(d, 2 * d, bias=False)
        self.down_proj = nn.Linear(2 * d, d, bias=False)
        with torch.no_grad():
            for lin in (self.gate_proj, self.up_proj, self.down_proj):
                lin.weight.copy_(
                    torch.randn(lin.weight.shape, generator=gen) / d**0.5
                )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DenseToyBlock(nn.Module):
    """Residual block whose children mimic the llama decoder-layer shape."""

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        self.input_layernorm = ToyRMSNorm(d, seed=seed)
        self.post_attention_layernorm = ToyRMSNorm(d, seed=seed + 1000)
        self.mlp = ToyGatedMLP(d, seed=seed + 2000)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + 0.1 * self.input_layernorm(h)
        return h + self.mlp(self.post_attention_layernorm(h))


class _DenseDecoder(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(199)
        self.embed_tokens = nn.Embedding(TOY_VOCAB, DENSE_D)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.randn(TOY_VOCAB, DENSE_D, generator=gen)
            )
        self.layers = nn.ModuleList(
            DenseToyBlock(DENSE_D, seed=i) for i in range(n_layers)
        )
        self.norm = ToyRMSNorm(DENSE_D, seed=87)


class DenseToyCausalLM(nn.Module):
    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.model = _DenseDecoder(n_layers)
        self.lm_head = nn.Linear(DENSE_D, TOY_VOCAB, bias=False)
        with torch.no_grad():
            gen = torch.Generator().manual_seed(211)
            self.lm_head.weight.copy_(
                torch.randn(TOY_VOCAB, DENSE_D, generator=gen)
            )
        self.config = SimpleNamespace(model_type="llama")

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        del use_cache
        h = self.model.embed_tokens(input_ids)
        for block in self.model.layers:
            h = block(h)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(h)))


class _DenseStubSession(_StubSession):
    """The jlens stub session with the RelP-compatible dense toy swapped in."""

    def __init__(self, *, n_layers: int = 3) -> None:
        super().__init__(n_layers=n_layers)
        model = DenseToyCausalLM(n_layers)
        model.requires_grad_(False)  # replicate load_model's freeze
        model.eval()
        self._model = model
        self._layers = model.model.layers
        self.model_id = _DENSE_MODEL_ID


# ---------------------------------------------------------------------------
# The detached backward factors
# ---------------------------------------------------------------------------


def test_norm_ln_rule_matches_reference_detached_rsqrt() -> None:
    norm = ToyRMSNorm(DENSE_D, seed=3)
    gen = torch.Generator().manual_seed(7)
    base = torch.randn(5, DENSE_D, generator=gen)
    base[0, 3] = 0.0

    x = base.clone().requires_grad_(True)
    out = _relp_norm_forward(norm, x)
    assert torch.equal(out.detach(), norm(base))  # forward untouched
    (grad,) = torch.autograd.grad(out.sum(), x)

    # LN-rule reference: the rsqrt denominator treated as a constant.
    x_ref = base.clone().requires_grad_(True)
    ref = norm.weight * (
        x_ref
        * torch.rsqrt(
            x_ref.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
        ).detach()
    )
    (ref_grad,) = torch.autograd.grad(ref.sum(), x_ref)

    assert torch.allclose(grad, ref_grad, atol=1e-6)
    # Division-free factor: the x = 0 coordinate keeps weight/rms, not 0.
    assert grad[0, 3] != 0.0


def test_act_factor_is_detached_sigmoid() -> None:
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(64, generator=gen)
    x[3] = 0.0
    factor = _act_backward_factor(nn.SiLU(), x)
    assert torch.equal(factor, torch.sigmoid(x.float()))
    assert factor[3] == 0.5


# ---------------------------------------------------------------------------
# Structural rejection, self-verification, and reversible patching
# ---------------------------------------------------------------------------


def test_relp_rules_reject_toy_without_dense_shape() -> None:
    layers = list(ToyCausalLM().model.layers)
    with pytest.raises(RelpUnsupportedError):
        with relp_backward_rules(layers):
            pass  # pragma: no cover - __enter__ must raise


def test_relp_rules_forward_bitexact_and_gradients_differ() -> None:
    blocks = [DenseToyBlock(DENSE_D, seed=i) for i in range(3)]
    gen = torch.Generator().manual_seed(11)
    base = torch.randn(1, 8, DENSE_D, generator=gen)

    def _forward_and_grad() -> tuple[torch.Tensor, torch.Tensor]:
        x = base.clone().requires_grad_(True)
        h = x
        for block in blocks:
            h = block(h)
        (grad,) = torch.autograd.grad(h.sum(), x)
        return h.detach(), grad

    vanilla_out, vanilla_grad = _forward_and_grad()
    with relp_backward_rules(blocks):
        relp_out, relp_grad = _forward_and_grad()
    restored_out, restored_grad = _forward_and_grad()

    assert torch.equal(relp_out, vanilla_out)
    assert not torch.allclose(relp_grad, vanilla_grad)
    assert torch.equal(restored_out, vanilla_out)
    assert torch.equal(restored_grad, vanilla_grad)
    assert all(
        "forward" not in module.__dict__
        for block in blocks
        for module in block.modules()
    )


def test_deviant_gated_mlp_refused_on_first_forward() -> None:
    class _ScaledGatedMLP(ToyGatedMLP):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return 2.0 * super().forward(x)

    block = DenseToyBlock(DENSE_D, seed=0)
    block.mlp = _ScaledGatedMLP(DENSE_D, seed=5)
    gen = torch.Generator().manual_seed(13)
    x = torch.randn(1, 4, DENSE_D, generator=gen)
    with relp_backward_rules([block]):
        with pytest.raises(RelpUnsupportedError, match="canonical gated-MLP"):
            block(x)


def test_preexisting_instance_forward_refused_at_patch_time() -> None:
    block = DenseToyBlock(DENSE_D, seed=0)
    block.mlp.forward = lambda x: x  # instance-level override
    with pytest.raises(RelpUnsupportedError, match="instance-level forward"):
        with relp_backward_rules([block]):
            pass  # pragma: no cover - __enter__ must raise
    # The norms patched before the refusal were restored on the way out.
    assert "forward" not in block.input_layernorm.__dict__
    assert "forward" not in block.post_attention_layernorm.__dict__


# ---------------------------------------------------------------------------
# Session-level fits: local/relp lands beside local/default
# ---------------------------------------------------------------------------


def test_relp_fit_lands_beside_standard_and_switches() -> None:
    s = _DenseStubSession()
    standard = s.fit_jlens(_PROMPTS, corpus_spec="test")
    assert standard.n_prompts == len(_PROMPTS)
    relp = s.fit_jlens(_PROMPTS, corpus_spec="test", backward_rules="relp")
    assert relp.n_prompts == len(_PROMPTS)

    std_sidecar = load_local_lens_sidecar(_DENSE_MODEL_ID, "default")
    relp_sidecar = load_local_lens_sidecar(_DENSE_MODEL_ID, "relp")
    assert std_sidecar is not None
    assert relp_sidecar is not None
    assert std_sidecar["estimator_policy"] == lens_estimator_policy()
    assert relp_sidecar["estimator_policy"] == lens_estimator_policy(
        backward_rules="relp",
    )
    assert relp_sidecar["estimator_policy"]["backward_rules"] == RELP_RULES
    assert std_sidecar["method"] == "jlens_cotangent_sum"
    assert relp_sidecar["method"] == "relp_cotangent_sum"

    std_loaded = load_local_lens(_DENSE_MODEL_ID, "default")
    relp_loaded = load_local_lens(_DENSE_MODEL_ID, "relp")
    assert std_loaded is not None
    assert relp_loaded is not None
    std_lens, _ = std_loaded
    relp_lens, _ = relp_loaded
    assert not torch.allclose(std_lens.jacobians[0], relp_lens.jacobians[0])

    sources = {row["source"] for row in list_lens_sources(_DENSE_MODEL_ID)}
    assert {"local:default", "local:relp"} <= sources

    # save_lens activates the name it saved: the relp fit ran last.
    active = load_active_lens_source(_DENSE_MODEL_ID)
    assert active is not None
    assert (active["kind"], active["name"]) == ("local", "relp")

    use_lens_source(_DENSE_MODEL_ID, f"{LOCAL_SOURCE_PREFIX}default")
    active = load_active_lens_source(_DENSE_MODEL_ID)
    assert active is not None
    assert (active["kind"], active["name"]) == ("local", "default")

    with pytest.raises(ValueError, match="backward_rules"):
        s.fit_jlens(_PROMPTS, corpus_spec="test", backward_rules="nonsense")


def test_relp_policy_never_resumes_standard_artifact() -> None:
    s = _DenseStubSession()
    s.fit_jlens(_PROMPTS, corpus_spec="test")
    s.fit_jlens(_PROMPTS, corpus_spec="test", backward_rules="relp")
    std_before = load_local_lens_sidecar(_DENSE_MODEL_ID, "default")
    relp_before = load_local_lens_sidecar(_DENSE_MODEL_ID, "relp")
    assert std_before is not None
    assert relp_before is not None

    # Same prompts, standard policy again: an exact no-op against the
    # default artifact, never a resume of (or over) the relp one.
    again = s.fit_jlens(_PROMPTS, corpus_spec="test")
    assert again.n_prompts == len(_PROMPTS)

    std_after = load_local_lens_sidecar(_DENSE_MODEL_ID, "default")
    assert std_after is not None
    assert std_after["n_prompts"] == std_before["n_prompts"]
    assert std_after == std_before
    assert load_local_lens_sidecar(_DENSE_MODEL_ID, "relp") == relp_before


# ---------------------------------------------------------------------------
# The RelP policy survives checkpoint, resume, promotion and no-op
# ---------------------------------------------------------------------------


_RELP_POLICY = lens_estimator_policy(backward_rules="relp")


@pytest.fixture
def relabels(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record pre-fix relabels; a post-fix fit must never need one."""
    import drowse.lens.store as lens_io

    seen: list[str] = []
    monkeypatch.setattr(
        lens_io, "_persist_relabel",
        lambda sc_path, *_args, **_kwargs: seen.append(sc_path.name),
    )
    return seen


def _interrupt_relp_fit_after(n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a RelP fit that checkpoints every prompt and dies after ``n``."""
    import drowse.lens.jlens as jlens_mod

    real_fit = jlens_mod.fit_jacobian_lens

    def _interrupted(*args: Any, **kwargs: Any) -> Any:
        args = (*args[:2], list(args[2])[:n], *args[3:])
        kwargs["input_id_rows"] = kwargs["input_id_rows"][:n]
        kwargs["checkpoint_every"] = 1
        real_fit(*args, **kwargs)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", _interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _DenseStubSession().fit_jlens(
            _PROMPTS, corpus_spec="test", backward_rules="relp",
        )
    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", real_fit)


def _forbid_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    import drowse.lens.jlens as jlens_mod
    import drowse.lens.store as lens_io

    def _fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a finished RelP fit must not refit or rewrite")

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", _fail)
    monkeypatch.setattr(lens_io, "_save_fp32_square_safetensors_atomic", _fail)


def _write_pre_fix_labels(pointer: Path) -> None:
    """Label a pointer the way the pre-fix checkpoint writer did."""
    data = json.loads(pointer.read_text())
    data["method"] = "jlens_cotangent_sum"
    data["estimator_policy"] = lens_estimator_policy()
    pointer.write_text(json.dumps(data))


def test_relp_checkpoint_records_relp_policy(
    monkeypatch: pytest.MonkeyPatch, relabels: list[str],
) -> None:
    _interrupt_relp_fit_after(2, monkeypatch)

    # The pointer the writer produced, read before any loader touches it.
    _, checkpoint_path = lens_checkpoint_paths(_DENSE_MODEL_ID, "relp")
    written = json.loads(checkpoint_path.read_text())
    assert written["checkpoint"] is True
    assert written["n_prompts"] == 2
    assert written["method"] == "relp_cotangent_sum"
    assert written["estimator_policy"] == _RELP_POLICY
    assert load_lens_checkpoint_sidecar(_DENSE_MODEL_ID, "relp") == written
    assert relabels == []


def test_interrupted_relp_fit_resumes_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch, relabels: list[str],
) -> None:
    import drowse.lens.jlens as jlens_mod

    full = _DenseStubSession().fit_jlens(
        _PROMPTS, corpus_spec="test", backward_rules="relp",
    )
    remove_lens(_DENSE_MODEL_ID, "relp")
    _interrupt_relp_fit_after(2, monkeypatch)

    real_fit = jlens_mod.fit_jacobian_lens
    consumed: list[int] = []

    def _counting(*args: Any, **kwargs: Any) -> Any:
        consumed.append(len(list(args[2])))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(jlens_mod, "fit_jacobian_lens", _counting)
    messages: list[str] = []
    resumed = _DenseStubSession().fit_jlens(
        _PROMPTS, corpus_spec="test", backward_rules="relp",
        on_progress=messages.append,
    )
    assert any("checkpoint at 2 prompts" in m for m in messages)
    assert consumed == [len(_PROMPTS) - 2]
    assert resumed.n_prompts == len(_PROMPTS)
    for layer in full.source_layers:
        assert torch.allclose(
            resumed.jacobians[layer], full.jacobians[layer], atol=2e-3,
        )
    assert relabels == []


def test_promoted_relp_final_carries_relp_policy(
    monkeypatch: pytest.MonkeyPatch, relabels: list[str],
) -> None:
    import drowse.lens.store as lens_io

    promoted: list[bool] = []
    real_promote = lens_io.promote_lens_checkpoint

    def _spy(*args: Any, **kwargs: Any) -> bool:
        promoted.append(real_promote(*args, **kwargs))
        return promoted[-1]

    monkeypatch.setattr(lens_io, "promote_lens_checkpoint", _spy)
    _DenseStubSession().fit_jlens(
        _PROMPTS, corpus_spec="test", backward_rules="relp",
        checkpoint_every=len(_PROMPTS),
    )
    assert promoted == [True]
    _, manifest = lens_paths(_DENSE_MODEL_ID, "relp")
    final = json.loads(manifest.read_text())
    assert final["checkpoint"] is False
    assert final["method"] == "relp_cotangent_sum"
    assert final["estimator_policy"] == _RELP_POLICY
    assert load_lens_checkpoint_sidecar(_DENSE_MODEL_ID, "relp") is None
    assert relabels == []


def test_finished_relp_fit_rerun_is_noop(
    monkeypatch: pytest.MonkeyPatch, relabels: list[str],
) -> None:
    _DenseStubSession().fit_jlens(
        _PROMPTS, corpus_spec="test", backward_rules="relp",
        checkpoint_every=len(_PROMPTS),
    )
    before = load_local_lens_sidecar(_DENSE_MODEL_ID, "relp")
    assert before is not None

    _forbid_estimator(monkeypatch)
    messages: list[str] = []
    again = _DenseStubSession().fit_jlens(
        _PROMPTS, corpus_spec="test", backward_rules="relp",
        on_progress=messages.append,
    )
    assert any("nothing to do" in m for m in messages)
    assert again.n_prompts == len(_PROMPTS)
    assert load_local_lens_sidecar(_DENSE_MODEL_ID, "relp") == before
    assert relabels == []


def test_cli_preflight_reports_finished_relp_fit(
    monkeypatch: pytest.MonkeyPatch, relabels: list[str],
) -> None:
    from drowse.lens.store import HfLensCorpus

    corpus = HfLensCorpus("repo", "config", "a" * 40)
    spec = corpus.spec
    session = _DenseStubSession()
    setattr(session._model, "_drowse_source_fingerprint", "source-a")
    session.fit_jlens(
        _PROMPTS, corpus_spec=spec, backward_rules="relp",
        checkpoint_every=len(_PROMPTS),
    )
    monkeypatch.setattr(
        "drowse.model.loading.model_source_fingerprint",
        lambda *_args, **_kwargs: "source-a",
    )
    args = argparse.Namespace(
        force=False, model=_DENSE_MODEL_ID, quantize=None, device="cpu",
        seq_len=None, corpus=None, prompts=len(_PROMPTS), relp=True,
    )
    assert _try_lens_fit_noop_preflight(args, None, hf_corpus=corpus)
    args.relp = False  # the standard fit has its own, absent, artifact
    assert not _try_lens_fit_noop_preflight(args, None, hf_corpus=corpus)
    assert relabels == []


def test_pre_fix_relp_final_is_relabelled_and_rerun_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _DenseStubSession().fit_jlens(
        _PROMPTS, corpus_spec="test", backward_rules="relp",
    )
    _, manifest = lens_paths(_DENSE_MODEL_ID, "relp")
    _write_pre_fix_labels(manifest)

    sidecar = load_local_lens_sidecar(_DENSE_MODEL_ID, "relp")
    assert sidecar is not None
    assert sidecar["method"] == "relp_cotangent_sum"
    assert sidecar["estimator_policy"] == _RELP_POLICY
    assert json.loads(manifest.read_text()) == sidecar  # persisted once

    _forbid_estimator(monkeypatch)
    messages: list[str] = []
    _DenseStubSession().fit_jlens(
        _PROMPTS, corpus_spec="test", backward_rules="relp",
        on_progress=messages.append,
    )
    assert any("nothing to do" in m for m in messages)


def test_pre_fix_relp_checkpoint_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    _interrupt_relp_fit_after(2, monkeypatch)
    _, checkpoint_path = lens_checkpoint_paths(_DENSE_MODEL_ID, "relp")
    _write_pre_fix_labels(checkpoint_path)

    messages: list[str] = []
    resumed = _DenseStubSession().fit_jlens(
        _PROMPTS, corpus_spec="test", backward_rules="relp",
        on_progress=messages.append,
    )
    assert any("checkpoint at 2 prompts" in m for m in messages)
    assert resumed.n_prompts == len(_PROMPTS)
    final = load_local_lens_sidecar(_DENSE_MODEL_ID, "relp")
    assert final is not None
    assert final["estimator_policy"] == _RELP_POLICY


def test_pre_fix_labels_outside_relp_stay_standard() -> None:
    _DenseStubSession().fit_jlens(_PROMPTS, corpus_spec="test")
    _, manifest = lens_paths(_DENSE_MODEL_ID, "default")
    before = manifest.read_text()
    sidecar = load_local_lens_sidecar(_DENSE_MODEL_ID, "default")
    assert sidecar is not None
    assert sidecar["estimator_policy"] == lens_estimator_policy()
    assert manifest.read_text() == before


def test_lens_writers_bind_relp_name_to_relp_rules() -> None:
    lens = JacobianLens({0: torch.eye(DENSE_D)}, n_prompts=1, d_model=DENSE_D)
    common: dict[str, Any] = dict(
        corpus_spec="test", corpus_sha256="0" * 64, seq_len=128,
        dim_batch=8, skip_first=16,
    )
    with pytest.raises(ValueError, match="local/relp"):
        save_lens(lens, _DENSE_MODEL_ID, name="relp", **common)
    with pytest.raises(ValueError, match="local/relp"):
        save_lens(lens, _DENSE_MODEL_ID, backward_rules="relp", **common)
    with pytest.raises(ValueError, match="local/relp"):
        save_lens_checkpoint_accumulator(
            {0: torch.eye(DENSE_D)}, 1, DENSE_D, _DENSE_MODEL_ID,
            name="relp", base=None, **common,
        )
