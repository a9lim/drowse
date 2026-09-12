"""Numerical checks against actual Transformers architectures, without downloads."""

from __future__ import annotations

from typing import Any

import pytest
import numpy as np
import torch
import transformers

from drowse.core.capture import _capture_all_hidden_states
from drowse.core.jlens import JacobianLens, lens_logits
from drowse.core.model import LogitReadout, get_final_norm, get_layers, get_logit_softcap, get_output_projection, get_unembedding


@pytest.fixture(params=["llama", "qwen2", "qwen3", "qwen3_5_text", "gemma2", "gemma3_text", "gpt2", "gpt_neox", "opt",
                        "opt_projected", "opt_post_norm", "phi", "cohere", "granite"])
def model(request: pytest.FixtureRequest):
    torch.manual_seed(812)
    extra = {}
    architecture = request.param
    if architecture == "opt_projected":
        architecture, extra = "opt", {"word_embed_proj_dim": 16}
    elif architecture == "opt_post_norm":
        architecture, extra = "opt", {"do_layer_norm_before": False}
    elif architecture == "granite":
        extra = {"logits_scaling": 3.0}
    if request.param == "qwen3_5_text":
        extra = dict(linear_key_head_dim=8, linear_value_head_dim=8,
                     linear_num_key_heads=4, linear_num_value_heads=4,
                     layer_types=["linear_attention", "linear_attention", "full_attention"],
                     rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.,
                                      "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    config = transformers.AutoConfig.for_model(
        architecture, vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, sliding_window=16,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attention_dropout=0., hidden_dropout=0., resid_pdrop=0.,
        embd_pdrop=0., attn_pdrop=0., dropout=0., layerdrop=0.,
        **extra,
    )
    config._attn_implementation = "eager"
    model = transformers.AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
    head = model.get_output_embeddings()
    if getattr(head, "bias", None) is not None:
        head.bias.copy_(torch.linspace(-1., 1., config.vocab_size))
    return model


def test_capture_matches_block_outputs_and_right_padded_single_rows(model: Any):
    layers = get_layers(model)
    ids = torch.tensor([[1, 7, 11, 13, 19], [1, 23, 29, 0, 0]])
    mask = (ids != 0).long()
    expected = {}

    def record(index: int):
        def hook(_module: Any, _args: Any, output: Any):
            value = output[0] if isinstance(output, tuple) else output
            expected[index] = value.detach().clone()
        return hook

    handles = [layer.register_forward_hook(record(i)) for i, layer in enumerate(layers)]
    try:
        with torch.inference_mode():
            model(ids, attention_mask=mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    actual = _capture_all_hidden_states(model, layers, ids, attention_mask=mask)
    pooled = _capture_all_hidden_states(
        model, layers, ids, attention_mask=mask, pool_index=torch.tensor([4, 2]),
    )
    for i in range(len(layers)):
        torch.testing.assert_close(actual[i], expected[i], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(pooled[i], expected[i][[0, 1], [4, 2]], atol=1e-6, rtol=1e-5)


def test_right_padded_capture_matches_single_rows_in_float64(model: Any):
    # Isolate padding correctness from platform-dependent fp32 batched GEMM rounding.
    model.double()
    layers = get_layers(model)
    ids = torch.tensor([[1, 7, 11, 13, 19], [1, 23, 29, 0, 0]])
    pooled = _capture_all_hidden_states(
        model, layers, ids, attention_mask=(ids != 0).long(),
        pool_index=torch.tensor([4, 2]), promote_pooled=False,
    )
    for row, length in [(0, 5), (1, 3)]:
        single = _capture_all_hidden_states(model, layers, ids[row:row + 1, :length])
        for i in range(len(layers)):
            torch.testing.assert_close(pooled[i][row], single[i][0, -1], atol=1e-6, rtol=1e-5)


def test_cached_logits_match_full_prefix(model: Any):
    ids = torch.tensor([[1, 7, 11, 13, 19, 23, 29]])
    with torch.inference_mode():
        cache = model(ids[:, :3], use_cache=True).past_key_values
        for end in range(4, ids.shape[1] + 1):
            cached = model(ids[:, end - 1:end], past_key_values=cache, use_cache=True)
            full = model(ids[:, :end], use_cache=False)
            torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], atol=2e-6, rtol=2e-5)
            cache = cached.past_key_values


def test_final_residual_identity_lens_matches_model_logits(model: Any):
    layers = get_layers(model)
    last = len(layers) - 1
    ids = torch.tensor([[1, 7, 11, 13, 19]])
    hidden = _capture_all_hidden_states(model, layers, ids, layer_indices=[last])
    lens = JacobianLens({last: torch.eye(hidden[last].shape[-1])}, n_prompts=1, d_model=hidden[last].shape[-1])
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
        actual = lens_logits(
            lens, hidden, unembed=get_unembedding(model), final_norm=get_final_norm(model),
            logit_softcap=get_logit_softcap(model),
            readout=LogitReadout(model),
        )[last]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("ablate", [False, True])
def test_steering_manager_matches_independent_block_intervention(model: Any, ablate: bool):
    from drowse.core.hooks import SteeringManager
    from drowse.core.manifold import LayerSubspace, SynthesizedSubspace

    ids = torch.tensor([[1, 7, 11, 13, 19]])
    layers = get_layers(model)
    dim = model.config.hidden_size
    basis = torch.eye(dim)[:2]
    mean = torch.linspace(-.1, .1, dim)
    target = torch.tensor([.03, 0. if ablate else -.02])
    kappa = torch.tensor([0., .7 if ablate else 0.])
    sub = LayerSubspace.affine(mean, basis)
    synth = SynthesizedSubspace({1: sub}, {1: target}, {1: 1.}, {1: kappa})
    manager = SteeringManager()
    manager.add_subspace("test", synth)
    manager.apply_to_model(layers, torch.device("cpu"), torch.float32)
    try:
        with torch.inference_mode():
            actual = model(ids, use_cache=False).logits
    finally:
        manager.clear_all()

    def intervene(_module: Any, _args: Any, output: Any):
        h = output[0] if isinstance(output, tuple) else output
        values = h.detach().numpy().astype(np.float64)
        coords = (values - mean.numpy()) @ basis.numpy().T
        expected = values + (16 * target.numpy() - kappa.numpy() * coords) @ basis.numpy()
        replaced = torch.tensor(expected, dtype=h.dtype)
        return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

    handle = layers[1].register_forward_hook(intervene)
    try:
        with torch.inference_mode():
            expected = model(ids, use_cache=False).logits
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert not manager.hooks


@pytest.mark.parametrize("temperature,top_k,top_p", [(1., 0, 1.), (.7, 13, .8), (0., 7, .9)])
def test_generation_logprobs_and_entropy_match_numpy(model: Any, temperature: float, top_k: int, top_p: float):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from drowse.core.generation import GenerationConfig, GenerationState, generate_steered

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f"t{i}": i for i in range(64)}, unk_token="t0",
    )))
    ids = torch.tensor([[1, 7, 11]])
    forced = [13, 19, 23]
    frames = []
    generated = generate_steered(
        model, tokenizer, ids,
        GenerationConfig(max_new_tokens=3, temperature=temperature, top_k=top_k, top_p=top_p),
        GenerationState(), on_token=lambda *frame: frames.append(frame),
        logprobs=5, forced_prefix=forced, seed=81,
    )
    assert generated == forced
    assert len(frames) == 3
    for step, frame in enumerate(frames):
        prefix = torch.cat([ids, torch.tensor([forced[:step]], dtype=torch.long)], dim=1)
        with torch.inference_mode():
            logits = model(prefix, use_cache=False).logits[0, -1].numpy().astype(np.float64)
        if temperature == 0:
            indices = np.array([logits.argmax()])
            probs = np.array([1.])
        else:
            indices = np.argsort(-logits)[:top_k or len(logits)]
            scaled = logits[indices] / temperature
            probs = np.exp(scaled - scaled.max())
            probs /= probs.sum()
            keep = np.cumsum(probs) - probs < top_p
            indices, probs = indices[keep], probs[keep]
            probs /= probs.sum()
        expected = dict(zip(indices.tolist(), np.log(probs).tolist(), strict=True))
        if forced[step] in expected:
            assert frame[3] == pytest.approx(expected[forced[step]], abs=3e-6)
        else:
            assert frame[3] is None
        assert frame[5] == pytest.approx(float(np.exp(-(probs * np.log(probs)).sum())), rel=2e-6)
        for alt in frame[4]:
            assert alt.logprob == pytest.approx(expected.get(alt.id, -float("inf")), abs=3e-6)


def test_live_and_batched_lens_use_model_logit_calibration(model: Any):
    from tests.test_jlens_session import _FakeCapture, _StubSession

    layers = get_layers(model)
    last = len(layers) - 1
    ids = torch.tensor([[1, 7, 11, 13, 19]])
    hidden = _capture_all_hidden_states(model, layers, ids, layer_indices=[last])[last][0, -1]
    lens = JacobianLens({last: torch.eye(hidden.numel())}, n_prompts=1, d_model=hidden.numel())
    session = _StubSession()
    session._model, session._layers = model, layers
    session._require_jlens = lambda: lens
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits[0, -1]
        actual = session._jlens_logits_rows(lens, [(last, hidden)])[0]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        session._lens_instrument.enable_live(layers=[last])
        session._capture = _FakeCapture({last: hidden})
        result = session._live_lens_readout_step(top_k=5)
        assert result is not None
        rows, aggregate, token_ids = result
    values, indices = expected.softmax(-1).topk(5)
    assert token_ids[last] == indices.tolist()
    torch.testing.assert_close(torch.tensor([value for _, value in rows[last]]), values, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(torch.tensor([row[1] for row in aggregate]), values, atol=1e-7, rtol=1e-5)
    session._lens_instrument.disable_live()


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=[pytest.mark.gpu, pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable",
)])])
def test_identity_lens_preserves_native_bfloat16_normalization(model: Any, device: str):
    model.to(device=device, dtype=torch.bfloat16)
    layers = get_layers(model)
    last = len(layers) - 1
    ids = torch.tensor([[1, 7, 11, 13, 19]], device=device)
    hidden = _capture_all_hidden_states(model, layers, ids, layer_indices=[last])
    lens = JacobianLens({last: torch.eye(hidden[last].shape[-1])}, n_prompts=1, d_model=hidden[last].shape[-1])
    with torch.inference_mode():
        actual = lens_logits(lens, hidden, unembed=get_unembedding(model), final_norm=get_final_norm(model),
                             logit_softcap=get_logit_softcap(model), readout=LogitReadout(model))[last]
        expected = model(ids, use_cache=False).logits.float()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_token_directions_and_decomposition_include_output_projection(model: Any):
    from drowse.core.session import DrowseSession
    from tests.test_jlens_session import _StubSession

    dim = model.config.hidden_size
    jacobian = torch.randn(dim, dim) / dim ** .5
    lens = JacobianLens({0: jacobian}, n_prompts=1, d_model=dim)
    unembed = get_unembedding(model)
    projection = get_output_projection(model)
    row = unembed[7].numpy().astype(np.float64)
    if projection is not None:
        row = row @ projection.numpy().astype(np.float64)
    expected = row @ jacobian.numpy().astype(np.float64)
    actual = lens.token_direction(7, unembed, output_projection=projection)[0]
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-7, rtol=1e-5)
    session: Any = _StubSession()
    session._model = model
    session._require_jlens = lambda: lens
    session.ensure_profile_registered = lambda _selector: {0: actual}
    result = DrowseSession.jspace_decompose(session, "test", layers=[0], k=1)
    assert result[0][0] == pytest.approx(1., abs=1e-5)
