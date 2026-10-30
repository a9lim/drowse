import json
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse.core.generation import GenerationConfig, GenerationState, generate_steered
from drowse.core.results import GenerationResult
from tests.conftest import FakeLogitsModel


class MetadataTokenizer:
    name_or_path = "metadata-regression"
    vocab_size = 4
    eos_token_id = 3
    added_tokens_encoder = {}
    all_special_ids = [3]

    def decode(self, ids: Any, **_kwargs: Any):
        return "".join(str(int(token)) for token in ids)

    def batch_decode(self, rows: Any):
        return [self.decode(row) for row in rows]


def test_missing_token_logprob_stays_null_in_api_responses():
    from drowse.server.app import _render_logprobs_chat, _render_logprobs_completions

    result = GenerationResult(text="02", tokens=[0, 2], token_count=2, tok_per_sec=1, elapsed=2,
        logprobs=[(0, -0.5, []), (2, None, [])])
    session = cast(Any, SimpleNamespace(tokenizer=MetadataTokenizer()))
    chat = _render_logprobs_chat(result, session)
    completion = _render_logprobs_completions(result, session)
    assert chat is not None and completion is not None
    assert [row["logprob"] for row in chat["content"]] == [-0.5, None]
    assert completion["token_logprobs"] == [-0.5, None]
    json.dumps([chat, completion], allow_nan=False)


@pytest.mark.parametrize("offset", [-1000., 0., 1000.])
@pytest.mark.parametrize("steering_active", [False, True])
def test_finite_logit_offsets_do_not_change_sampling(offset: float, steering_active: bool):
    logits = torch.tensor([[[2., 1., 0., -3.]]]) + offset
    model = FakeLogitsModel(lambda _ids: logits.clone(), with_past_key_values=True,
        config=SimpleNamespace(vocab_size=4), generation_config=SimpleNamespace(eos_token_id=3))
    events = []
    generate_steered(cast(Any, model), cast(Any, MetadataTokenizer()), torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=1, top_p=1., top_k=4), GenerationState(),
        forced_prefix=[0], logprobs=4, steering_active=steering_active, on_token=lambda *event: events.append(event))
    expected = torch.tensor([2., 1., 0., -3.]).log_softmax(-1)
    assert events[0][3] == pytest.approx(float(expected[0]), abs=1e-6)
    assert [alt.logprob for alt in events[0][4]] == pytest.approx(expected.tolist(), abs=1e-6)


@pytest.mark.parametrize("count", [257, 4096, 262144])
def test_generation_returns_large_alternative_counts_up_to_vocabulary(count: int):
    vocab = 4096
    logits = torch.linspace(0, 1, vocab).reshape(1, 1, vocab)
    model = FakeLogitsModel(lambda _ids: logits.clone(), with_past_key_values=True,
        config=SimpleNamespace(vocab_size=vocab), generation_config=SimpleNamespace(eos_token_id=3))
    tokenizer = MetadataTokenizer()
    tokenizer.vocab_size = vocab
    events = []
    generate_steered(cast(Any, model), cast(Any, tokenizer), torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=1, temperature=3, top_k=262144, top_p=1),
        GenerationState(), forced_prefix=[0], seed=7, logprobs=count,
        want_perplexity=False, on_token=lambda *event: events.append(event))
    alternatives = events[0][4]
    assert len(alternatives) == min(count, vocab)
    assert [entry.id for entry in alternatives] == list(range(vocab - 1, vocab - 1 - len(alternatives), -1))
    if count >= vocab:
        assert sum(math.exp(entry.logprob) for entry in alternatives) == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=[pytest.mark.gpu, pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS unavailable",
    )]),
])
@pytest.mark.parametrize("forced", [0, 2])
@pytest.mark.parametrize("top_p", [1.0, 0.6])
@pytest.mark.parametrize("logprobs,want_ppl", [(None, True), (0, False), (3, True)])
def test_generation_metadata_preserves_probability_units(
    device: str, forced: int, top_p: float, logprobs: int | None, want_ppl: bool,
):
    logits = torch.tensor([[[2.0, 1.0, 0.0, -3.0]]], device=device)
    model = FakeLogitsModel(lambda _ids: logits.clone(), with_past_key_values=True,
        config=SimpleNamespace(vocab_size=4), generation_config=SimpleNamespace(eos_token_id=3))
    events = []
    ids = generate_steered(cast(Any, model), cast(Any, MetadataTokenizer()), torch.tensor([[0]], device=device),
        GenerationConfig(max_new_tokens=1, temperature=1.0, top_k=2, top_p=top_p),
        GenerationState(), forced_prefix=[forced], seed=7, logprobs=logprobs,
        want_perplexity=want_ppl, on_token=lambda *event: events.append(event))
    assert ids == [forced]
    assert len(events) == 1
    _, _, token_id, selected, alternatives, perplexity, _ = events[0]
    assert token_id == forced
    support = [1.0] if top_p < 1 else [math.e / (math.e + 1), 1 / (math.e + 1)]
    expected_selected = math.log(support[0]) if forced == 0 else None
    if logprobs is None or expected_selected is None:
        assert selected is None
    else:
        assert selected == pytest.approx(expected_selected, abs=2e-6)
    if logprobs:
        assert [entry.id for entry in alternatives] == list(range(len(support)))
        assert [entry.logprob for entry in alternatives] == pytest.approx([math.log(p) for p in support], abs=2e-6)
    else:
        assert alternatives is None
    if want_ppl:
        assert perplexity == pytest.approx(math.exp(-sum(p * math.log(p) for p in support)), abs=2e-6)
    else:
        assert perplexity is None
