"""Tests for DrowseSession programmatic API.
Requires a GPU (CUDA or Apple Silicon MPS) and downloads the public
SmolLM2-360M-Instruct test model on first run. Override with
``DROWSE_TEST_MODEL``.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import pytest
import torch
from drowse.core.profile import Profile
from drowse.core.results import GenerationResult, RunSet, TokenEvent
from tests._gpu_model import gpu_model_id, load_or_skip_inaccessible

if TYPE_CHECKING:
    from drowse.core.session import DrowseSession

_HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _HAS_GPU,
        reason="No GPU backend available (neither CUDA nor MPS)",
    ),
]

MODEL_ID = gpu_model_id()


def _corpus(response: str) -> list[str]:
    from drowse.core.capture import _load_baseline_prompts

    return [response] * len(_load_baseline_prompts())


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory):
    import os
    from drowse.core.session import DrowseSession
    # Isolate $DROWSE_HOME so this module's extract/merge writes (e.g.
    # local/formal.casual, happy.sad, honest) land in a throwaway cache instead
    # of the user's real ~/.drowse — where they would shadow bundled manifolds of
    # the same name and break bare-name resolution (AmbiguousSelectorError)
    # across the whole suite on the next run.
    home = tmp_path_factory.mktemp("drowse_home")
    prev = os.environ.get("DROWSE_HOME")
    os.environ["DROWSE_HOME"] = str(home)
    # device="auto" picks cuda > mps > cpu; skipif above guarantees a GPU.
    try:
        s = load_or_skip_inaccessible(
            lambda: DrowseSession.from_pretrained(MODEL_ID, device="auto", probes=["register"]),
            MODEL_ID,
        )
        yield s
        s.close()
    finally:
        if prev is None:
            os.environ.pop("DROWSE_HOME", None)
        else:
            os.environ["DROWSE_HOME"] = prev

class TestConstruction:
    def test_model_info(self, session: DrowseSession) -> None:
        info = session.model_info
        assert isinstance(info["model_type"], str)
        assert info["model_type"]
        assert info["hidden_dim"] > 0
        assert info["num_layers"] > 0

    def test_config_defaults(self, session: DrowseSession) -> None:
        assert session.config.temperature == 1.0
        assert session.config.top_p == 0.9
        assert session.config.max_new_tokens == 1024

    def test_probes_loaded(self, session: DrowseSession) -> None:
        assert len(session.probes) > 0

    def test_history_starts_empty(self, session: DrowseSession) -> None:
        assert session.tree.messages_for() == []

    def test_vectors_starts_empty(self, session: DrowseSession) -> None:
        assert session.profiles == {}

    def test_last_result_starts_none(self, session: DrowseSession) -> None:
        assert session.last_result is None

class TestSteering:
    def test_extract_and_steer(self, session: DrowseSession) -> None:
        name, profile = session.extract_from_corpora(
            "happy", _corpus("I am happy"), _corpus("I am sad"),
        )
        assert isinstance(profile, Profile)
        assert all(isinstance(k, int) for k in profile)
        session.steer("happy", profile)
        assert "happy" in session.profiles
        # profile registry stores raw per-layer tensor dicts (wrap in Profile for the public view).
        assert isinstance(session.profiles["happy"], dict)

    def test_unsteer(self, session: DrowseSession) -> None:
        session.unsteer("happy")
        assert "happy" not in session.profiles

    def test_extract_curated(self, session: DrowseSession) -> None:
        name, profile = session.extract("happy", baseline="sad")
        assert name == "happy.sad"
        assert isinstance(profile, Profile)
        assert len(profile) > 0

    def test_extract_datasource(self, session: DrowseSession) -> None:
        name, profile = session.extract_from_corpora(
            "formal.casual", _corpus("formal"), _corpus("casual"),
        )
        assert isinstance(profile, Profile)

class TestMonitoring:
    def test_monitor_and_unmonitor(self, session: DrowseSession) -> None:
        # Extract registers the folded direction; ``add_probe`` resolves it
        # (the unified probe API — one attach for vector + manifold probes).
        name, _profile = session.extract_from_corpora(
            "honest", _corpus("I am honest"), _corpus("I am deceptive"),
        )
        session.add_probe(name, as_name="test_probe")
        assert "test_probe" in session.probes
        session.remove_probe("test_probe")
        assert "test_probe" not in session.probes

class TestLifecycle:
    def test_context_manager(self):
        from drowse.core.session import DrowseSession
        created = load_or_skip_inaccessible(
            lambda: DrowseSession.from_pretrained(MODEL_ID, device="auto", probes=[]),
            MODEL_ID,
        )
        with created as s:
            assert s.model_info["model_type"]

class TestGeneration:
    def test_generate_unsteered(self, session: DrowseSession) -> None:
        result = session.generate("Say hello in one word.")
        assert isinstance(result, RunSet)
        assert isinstance(result.first, GenerationResult)
        single = result.first
        assert len(single.text) > 0
        assert single.token_count > 0
        assert single.tok_per_sec > 0
        assert single.elapsed > 0
        assert single.steering_alphas == {}

    def test_generate_blocking_messages(self, session: DrowseSession) -> None:
        result = session.generate([
            {"role": "user", "content": "Say hello in one word."},
        ])
        assert isinstance(result, RunSet)
        assert isinstance(result.first, GenerationResult)
        assert len(result.first.text) > 0

    def test_generate_appends_to_history(self, session: DrowseSession) -> None:
        session.clear_history()
        session.generate("Say hi.")
        messages = session.tree.messages_for()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_generate_with_alphas(self, session: DrowseSession) -> None:
        name, profile = session.extract_from_corpora(
            "formal.casual", _corpus("formal"), _corpus("casual"),
        )
        session.steer(name, profile)
        result = session.generate("Hello.", steering=f"0.1 {name}").first
        assert result.steering_alphas == {name: 0.1}
        session.unsteer(name)

    def test_generate_with_probes(self, session: DrowseSession) -> None:
        session.clear_history()
        result = session.generate("Tell me something exciting!").first
        if session.probes:
            assert isinstance(result.probe_readings, dict)

    def test_last_result(self, session: DrowseSession) -> None:
        session.clear_history()
        result = session.generate("Hello.")
        assert session.last_result is result.first

    def test_ab_comparison(self, session: DrowseSession) -> None:
        """A/B test: same prompt, with and without steering."""
        name, profile = session.extract_from_corpora(
            "happy", _corpus("I am happy"), _corpus("I am sad"),
        )
        session.steer(name, profile)
        session.clear_history()
        steered = session.generate("Describe a sunset.", steering=f"0.2 {name}").first
        session.clear_history()
        unsteered = session.generate("Describe a sunset.").first
        assert steered.steering_alphas == {name: 0.2}
        assert unsteered.steering_alphas == {}
        # Both should produce text
        assert len(steered.text) > 0
        assert len(unsteered.text) > 0
        session.unsteer(name)

    def test_unknown_vector_raises(self, session: DrowseSession) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            session.generate("Hello.", steering="0.1 nonexistent")

class TestCliRoundTrip:
    def test_extract_cli_roundtrip(self, tmp_path: Path) -> None:
        import subprocess
        import sys
        from drowse.io.paths import manifold_dir, safe_model_id

        folder = manifold_dir("local", "happy.sad")
        sid = safe_model_id(MODEL_ID)
        tensor_path = folder / f"{sid}.safetensors"
        created_here = not tensor_path.exists()

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "drowse", "manifold", "extract",
                    "happy.sad", "-m", MODEL_ID, "--namespace", "local",
                ],
                capture_output=True, text=True, timeout=600,
            )
            assert proc.returncode == 0, (
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
            assert tensor_path.exists(), f"expected {tensor_path} to exist"
        finally:
            # Only unlink the per-model tensor if this test created it;
            # leave the corpus and any pre-existing tensor alone.
            if created_here and tensor_path.exists():
                tensor_path.unlink()
                sidecar = folder / f"{sid}.json"
                if sidecar.exists():
                    sidecar.unlink()


class TestStreamingGeneration:
    def test_generate_stream(self, session: DrowseSession) -> None:
        session.clear_history()
        tokens = []
        for event in session.generate_stream("Say hello."):
            assert isinstance(event, TokenEvent)
            tokens.append(event)
        assert len(tokens) > 0
        assert all(isinstance(t.text, str) for t in tokens)
        assert session.last_result is not None
        assert session.last_result.token_count == len(tokens)

    def test_stream_with_alphas(self, session: DrowseSession) -> None:
        name, profile = session.extract_from_corpora(
            "happy", _corpus("I am happy"), _corpus("I am sad"),
        )
        session.steer(name, profile)
        session.clear_history()
        tokens = list(session.generate_stream("Hello.", steering=f"0.15 {name}"))
        assert len(tokens) > 0
        assert session.last_result is not None  # generate_stream guarantees last_result is set
        assert session.last_result.steering_alphas == {name: 0.15}
        session.unsteer(name)


class TestAblation:
    @pytest.mark.skip(reason=(
        "Pinned the Euclidean TraitMonitor, removed in the monitor "
        "unification (Mahalanobis-only). The ablation operator is Euclidean "
        "(h' = h - alpha(h.d_hat - mu.d_hat)d_hat) but the unified Monitor "
        "reads the Mahalanobis coordinate, which by design does NOT collapse "
        "under Euclidean ablation, so a monitor-based assertion no longer "
        "matches the operator. Recommended rewrite: assert directly on the "
        "metric-agnostic Euclidean projection of the post-ablation hidden "
        "onto folded_directions(session._monitor.manifolds[probe]) "
        "(the exact component the operator zeros) instead of a monitor read. "
        "Deferred to the monitor-shape pass (GPU, owner: a9); also revisit the "
        "fixture, which bootstraps the dropped 'affect' category."
    ))
    def test_ablation_suppresses_self_probe_score(self, session: DrowseSession) -> None:
        """Ablating a concept suppresses its own direction in activation space.

        See the skip reason: the original pinned the removed Euclidean monitor;
        the faithful Mahalanobis-era rewrite tests the operator via a direct
        Euclidean projection onto the probe's folded direction.
        """


def test_return_hidden_round_trip(session: DrowseSession) -> None:
    """return_hidden=True populates hidden_states; score_hidden round-trips."""
    from drowse import SamplingConfig

    num_layers = len(session._layers)
    hidden_dim = session.model_info["hidden_dim"]

    result = session.generate(
        "Count to three.",
        sampling=SamplingConfig(
            max_tokens=16, temperature=0.0, return_hidden=True,
        ),
    ).first
    assert result.hidden_states is not None
    # All layers captured.
    assert len(result.hidden_states) == num_layers
    # Shape: [T, D] per layer, T == len(generated tokens).
    T = len(result.tokens)
    for layer_idx, h in result.hidden_states.items():
        assert h.shape == (T, hidden_dim), (
            f"layer {layer_idx}: expected ({T}, {hidden_dim}), got {tuple(h.shape)}"
        )
        assert h.device.type == "cpu"

    # Round-trip: re-score the captured dict and compare against the
    # per-token scores the session computed inline. The fixture bootstraps
    # with a probe roster; if that invariant ever regresses, silent-skip
    # would hide the real test.
    assert session._monitor.probe_names, (
        "fixture must have probes loaded for round-trip coverage"
    )
    _, per_token = session.score_hidden(
        result.hidden_states, per_token=True,
    )
    expected = session.last_per_token_scores or {}
    # Both sides route through the same _score_probes kernel; the only
    # noise is the GPU→CPU move on the result-stored tensors, which is
    # well below 1e-4 for bf16. Looser tolerances let a one-token
    # pooling shift slip past.
    tol = 1e-4
    for name, vals in per_token.items():
        if name not in expected:
            continue
        assert len(vals) == len(expected[name])
        for a, b in zip(vals, expected[name]):
            assert abs(a - b) < tol, f"probe {name}: {a} vs {b}"


def test_return_hidden_false_leaves_hidden_states_none(session: DrowseSession) -> None:
    from drowse import SamplingConfig

    result = session.generate(
        "Hello.",
        sampling=SamplingConfig(max_tokens=4, temperature=0.0),
    ).first
    assert result.hidden_states is None


class TestPrefixCache:
    """Prefix KV cache: opt-in optimization for batch workloads with a
    shared chat prefix.  See ``DrowseSession.cache_prefix``.
    """

    def _shared_prefix_messages(self, session: DrowseSession, prompt_body: str):
        """Build a (prefix_messages, full_messages) pair where the
        full chat-template encoding of full_messages begins with the
        prefix_messages encoding.

        Trick: encode the prefix-only messages with
        ``add_generation_prompt=False`` and prepend its decoded text
        to the prompt body in a single user message — that shape
        always satisfies the byte-prefix invariant on tokenizers
        whose user-turn tokens partition cleanly on whitespace
        (Gemma's chat template is one such).
        """
        # Prefix is its own user turn; full is the same prefix +
        # additional user turn.  The chat-template encoding of
        # [u1] is a prefix of [u1, u2] under add_generation_prompt
        # = False for prefix_only and True for full.  But that
        # ISN'T sufficient — the False-encoded prefix may differ
        # by trailing tokens.  Instead, encode the prefix as a token
        # tensor to bypass the issue.
        from drowse.core.generation import build_chat_input

        prefix_msg = [
            {"role": "user", "content": "Be concise. Always end with a period."},
        ]
        full_msg = prefix_msg + [
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": prompt_body},
        ]
        full_ids = build_chat_input(
            session._tokenizer, full_msg, session.config.system_prompt,
        )
        # Find the longest token prefix that's a clean head of full_ids.
        # We just take everything up to (but not including) the second
        # user-turn opener.  Falling back to a deterministic split by
        # token-string match keeps this independent of model family.
        prefix_ids = build_chat_input(
            session._tokenizer, prefix_msg, session.config.system_prompt,
            add_generation_prompt=False,
        )
        # Trim prefix_ids to the longest matching head of full_ids.
        L = 0
        max_L = min(prefix_ids.shape[1], full_ids.shape[1])
        for i in range(max_L):
            if int(prefix_ids[0, i]) == int(full_ids[0, i]):
                L = i + 1
            else:
                break
        prefix_ids_trim = prefix_ids[:, :L]
        return prefix_ids_trim, full_msg

    def test_cache_hit_matches_no_cache_output(self, session: DrowseSession) -> None:
        from drowse import SamplingConfig

        session.clear_history()
        prefix_ids, full_msg = self._shared_prefix_messages(
            session, "Say the word 'banana' three times.",
        )

        # Baseline: no cache, deterministic.
        session.cache_prefix(None)
        baseline = session.generate(
            full_msg,
            sampling=SamplingConfig(max_tokens=12, temperature=0.0, seed=42),
            stateless=True,
        ).first

        # Warm the cache and rerun the same prompt; outputs must match.
        prefix_len = session.cache_prefix(prefix_ids)
        assert prefix_len > 0
        cached = session.generate(
            full_msg,
            sampling=SamplingConfig(max_tokens=12, temperature=0.0, seed=42),
            stateless=True,
        ).first
        # Same prompt + same seed + same model state → identical token
        # stream regardless of how prefill was sliced.
        assert cached.tokens == baseline.tokens, (
            f"prefix-cache hit produced different tokens than no-cache:\n"
            f"  no-cache: {baseline.tokens}\n"
            f"  cached:   {cached.tokens}"
        )
        # Cleanup so other tests aren't affected.
        session.cache_prefix(None)

    def test_steering_invalidates_cache(self, session: DrowseSession) -> None:
        # Warm the cache.
        prefix_ids, _ = self._shared_prefix_messages(session, "Hello.")
        session.cache_prefix(prefix_ids)
        assert session._prefix_cache is not None

        # Make sure there's a steering vector to push.
        if not session.has_profile("happy"):
            _, prof = session.extract_from_corpora(
                "happy", _corpus("I am happy"), _corpus("I am sad"),
            )
            session.steer("happy", prof)
        # steer() itself invalidates; re-warm and verify scope-entry
        # is what we're really testing.
        session.cache_prefix(prefix_ids)
        assert session._prefix_cache is not None

        with session.steering("0.1 happy"):
            # Scope entry must drop the cache.
            assert session._prefix_cache is None

        # And the post-scope rebuild must NOT magically reinstate it.
        assert session._prefix_cache is None
        session.unsteer("happy")
