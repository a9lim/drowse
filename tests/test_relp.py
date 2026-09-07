"""CPU tests for the RelP (R-lens) backward rules and the dual-name fit.

``core/relp.py`` keeps the estimator's forward values bit-identical —
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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from drowse.core.errors import RelpUnsupportedError
from drowse.core.relp import (
    RELP_RULES,
    _act_backward_factor,
    _relp_norm_forward,
    relp_backward_rules,
)
from drowse.io.lens import (
    lens_estimator_policy,
    load_local_lens,
    load_local_lens_sidecar,
)
from drowse.io.lens_sources import (
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
