"""Tests for the cross-branch joint-logprob computation
(``drowse.loom.joint_logprobs``).

Covers the pure-math helpers in isolation — building a real model would
load weights and a tokenizer, but the inner alignment + lookup is a
plain tensor + index walk that we can exercise with hand-built
log-softmax tensors.  The IO wrapper (``JointLogprobReplayer``) is
checked via a thin in-memory ``MockSession`` shim so we cover the full
forward-pass + cache + reorient path without HF weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drowse.model.chat import ChatTraits

from drowse.generation.decode import GenerationConfig, build_chat_input
from drowse.loom.joint_logprobs import (
    JointLogprobRow,
    JointLogprobs,
    SamplerRows,
    _approx_kl_topk,
    _cache_key,
    _compute_rows,
    reorient_for_request,
)
from drowse.generation.scoring import shared_prefix_len as _shared_prefix_len
from drowse.server.tree_routes import (
    _JOINT_LOGPROB_CACHE_MAX,
    _cached_joint_logprobs,
    _remember_joint_logprobs,
)
from drowse.steering.triggers import TriggerContext
from tests.conftest import FakeLogitsModel
from tests.support.lens import stub_lens_instrument


# ---------------------------------------------------------------------------
# Shared-prefix walker
# ---------------------------------------------------------------------------


def test_joint_logprob_cache_is_bounded_and_lru() -> None:
    cache: dict[int, str] = {}
    for key in range(_JOINT_LOGPROB_CACHE_MAX):
        _remember_joint_logprobs(cache, key, str(key))

    assert _cached_joint_logprobs(cache, 0) == "0"
    _remember_joint_logprobs(cache, _JOINT_LOGPROB_CACHE_MAX, "new")

    assert len(cache) == _JOINT_LOGPROB_CACHE_MAX
    assert 0 in cache
    assert 1 not in cache
    assert cache[_JOINT_LOGPROB_CACHE_MAX] == "new"


def test_shared_prefix_len_identical():
    assert _shared_prefix_len([1, 2, 3, 4], [1, 2, 3, 4]) == 4


def test_shared_prefix_len_divergent():
    assert _shared_prefix_len([1, 2, 3, 4], [1, 2, 99, 4]) == 2


def test_shared_prefix_len_empty():
    assert _shared_prefix_len([], [1, 2]) == 0
    assert _shared_prefix_len([1, 2], []) == 0


def test_shared_prefix_len_one_is_prefix_of_other():
    assert _shared_prefix_len([1, 2, 3], [1, 2, 3, 4]) == 3


# ---------------------------------------------------------------------------
# Cache key symmetry + reorient
# ---------------------------------------------------------------------------


def test_cache_key_is_symmetric():
    assert _cache_key("a", "b") == _cache_key("b", "a") == ("a", "b")


def test_reorient_for_request_noop():
    res = JointLogprobs(
        a_id="a", b_id="b", parent_id="p",
        rows=(JointLogprobRow(
            a_index=0, b_index=0, a_text="x", b_text="x", aligned=True,
            lp_a_in_a=-0.5, lp_b_in_b=-0.7,
            lp_a_in_b=-1.0, lp_b_in_a=-1.2,
            rank_changed=False, approx_kl=0.1,
        ),),
        n_rank1_changed=0,
    )
    out = reorient_for_request(res, "a", "b")
    assert out is res  # no copy when orientation matches


def test_reorient_for_request_flips_labels_and_columns():
    res = JointLogprobs(
        a_id="a", b_id="b", parent_id="p",
        rows=(JointLogprobRow(
            a_index=3, b_index=4, a_text="x", b_text="y", aligned=True,
            lp_a_in_a=-0.5, lp_b_in_b=-0.7,
            lp_a_in_b=-1.0, lp_b_in_a=-1.2,
            rank_changed=True, approx_kl=0.1,
        ),),
        n_rank1_changed=1,
    )
    out = reorient_for_request(res, "b", "a")
    assert out.a_id == "b" and out.b_id == "a"
    assert out.parent_id == "p"
    assert out.n_rank1_changed == 1
    row = out.rows[0]
    # Indices and texts swap; chosen-prob columns swap; cross columns swap.
    assert row.a_index == 4 and row.b_index == 3
    assert row.a_text == "y" and row.b_text == "x"
    assert row.lp_a_in_a == -0.7 and row.lp_b_in_b == -0.5
    assert row.lp_a_in_b == -1.2 and row.lp_b_in_a == -1.0
    assert row.rank_changed is True
    assert row.approx_kl == 0.1


# ---------------------------------------------------------------------------
# Approx-KL math
# ---------------------------------------------------------------------------


def _logits_to_logsoftmax(rows: list[list[float]]) -> torch.Tensor:
    """Build a ``[T, V]`` log-softmax tensor from raw logits rows."""
    return torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1)


def _pool(dense_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A full-vocab logprob row as an (exact) sorted sampler pool."""
    logps, ids = dense_row.sort(descending=True, stable=True)
    return ids, logps


def _rows(dense: torch.Tensor) -> SamplerRows:
    logps, ids = dense.sort(dim=-1, descending=True, stable=True)
    return SamplerRows({r: r for r in range(dense.shape[0])}, ids, logps)


def test_approx_kl_identical_distributions_is_zero():
    lp = _logits_to_logsoftmax([[0.0, 1.0, 2.0, 0.5]])[0]
    assert abs(_approx_kl_topk(_pool(lp), _pool(lp), top_k=4)) < 1e-6


def test_approx_kl_topk_grows_with_divergence():
    # Two distributions with the same support: B is a sharper version
    # of A's argmax.  KL(A || B) is strictly positive and grows when we
    # widen the cross-entropy.
    lp_a = _logits_to_logsoftmax([[0.0, 1.0, 0.5, -0.5]])[0]
    lp_b = _logits_to_logsoftmax([[-3.0, 5.0, -1.0, -3.0]])[0]
    kl = _approx_kl_topk(_pool(lp_a), _pool(lp_b), top_k=4)
    assert kl > 0
    # KL should be larger when B is even sharper.
    lp_b2 = _logits_to_logsoftmax([[-9.0, 9.0, -3.0, -9.0]])[0]
    kl2 = _approx_kl_topk(_pool(lp_a), _pool(lp_b2), top_k=4)
    assert kl2 > kl


def test_approx_kl_topk_respects_truncation():
    # With top_k=1 we only see the argmax token; coarser estimate.
    lp_a = _logits_to_logsoftmax([[0.0, 1.0, 0.5, -0.5]])[0]
    lp_b = _logits_to_logsoftmax([[-3.0, 5.0, -1.0, -3.0]])[0]
    kl_full = _approx_kl_topk(_pool(lp_a), _pool(lp_b), top_k=4)
    kl_top1 = _approx_kl_topk(_pool(lp_a), _pool(lp_b), top_k=1)
    # Full top-K should be >= top-1 (more mass observed).  Equality
    # is allowed (both argmax-dominated distributions) but inequality
    # is the typical shape.
    assert kl_full >= kl_top1 - 1e-6


# ---------------------------------------------------------------------------
# Per-row computation
# ---------------------------------------------------------------------------


def _build_logp(rows: list[list[float]]) -> torch.Tensor:
    return _logits_to_logsoftmax(rows)


def test_compute_rows_self_evaluation_only_on_divergent_tail():
    # Setup: vocab = 4 tokens.  Prefix is 2 tokens long.
    # Branch A's assistant tokens: [1, 2].
    # Branch B's assistant tokens: [1, 3] — diverges at position 1.
    vocab = 4
    prefix_len = 2

    # logp_a: T=4 rows.  Position 1 (predicts assistant[0]==1) puts
    # mass on token 1.  Position 2 (predicts assistant[1]==2) puts
    # mass on token 2.
    logp_a = _build_logp([
        [0.0] * vocab,                # position 0 (unused as predictor here)
        [-3.0, 0.0, -3.0, -3.0],      # position 1 — A's first assistant token (id=1)
        [-3.0, -3.0, 0.0, -3.0],      # position 2 — A's second assistant token (id=2)
        [-3.0, -3.0, -3.0, -3.0],     # tail
    ])
    # logp_b: B's second assistant token is 3, so position 2 puts mass
    # on token 3 instead.  Position 1 matches A (still on token 1) so
    # the aligned row should show lp_a_in_b == lp_a_in_a.
    logp_b = _build_logp([
        [0.0] * vocab,
        [-3.0, 0.0, -3.0, -3.0],
        [-3.0, -3.0, -3.0, 0.0],
        [-3.0, -3.0, -3.0, -3.0],
    ])
    ids_a = [10, 20, 1, 2]   # prefix tokens are dummy; assistant tail is [1, 2]
    ids_b = [10, 20, 1, 3]
    strs_a = ["P0", "P1", "tok_one", "tok_two"]
    strs_b = ["P0", "P1", "tok_one", "tok_three"]

    rows = _compute_rows(
        _rows(logp_a), _rows(logp_b), ids_a, ids_b, strs_a, strs_b, prefix_len,
    )
    # Two assistant positions on each side → two rows for the common
    # prefix.  Position 0 is byte-aligned ("tok_one" matches), position
    # 1 is not ("tok_two" vs "tok_three").
    assert len(rows) >= 2
    aligned0 = rows[0]
    assert aligned0.aligned is True
    assert aligned0.a_index == 0 and aligned0.b_index == 0
    # Self-eval: A's chosen token (id=1) at position prefix_len+0 is
    # predicted by logits at position prefix_len+0-1 == 1.  logp_a[1][1]
    # ≈ 0 after softmax (heaviest mass on token 1).
    assert aligned0.lp_a_in_a is not None
    assert aligned0.lp_a_in_a > -0.2  # near zero
    # Cross: B's distribution at the same position has the same mass on
    # token 1, so lp_a_in_b ≈ lp_a_in_a.
    assert aligned0.lp_a_in_b is not None
    assert abs(aligned0.lp_a_in_b - aligned0.lp_a_in_a) < 1e-5

    # Position 1: not byte-aligned (token strings differ), so cross-
    # evaluation should be None and rank_changed is False (it's only
    # set on aligned rows).
    misaligned = rows[1]
    assert misaligned.aligned is False
    assert misaligned.lp_a_in_b is None
    assert misaligned.lp_b_in_a is None
    assert misaligned.rank_changed is False


def test_compute_rows_rank_changed_flag():
    # Build two distributions that disagree on the argmax at the
    # aligned position.  ``pa_pos = prefix_len + a_idx - 1`` so with
    # prefix_len=2 and a_idx=0 we read logp[1] — the row we shaped.
    # Token id 0 wins under A, token id 1 wins under B.  Both branches
    # actually emit id=0 at position 0 so the strings byte-align.
    vocab = 3
    prefix_len = 2
    logp_a = _build_logp([
        [0.0] * vocab,          # position 0 (unused as predictor)
        [0.0, -5.0, -5.0],      # position 1 → predicts assistant[0]; argmax=0
        [0.0] * vocab,          # tail
    ])
    logp_b = _build_logp([
        [0.0] * vocab,
        [-5.0, 0.0, -5.0],      # position 1 → predicts assistant[0]; argmax=1
        [0.0] * vocab,
    ])
    ids_a = [99, 100, 0]
    ids_b = [99, 100, 0]
    strs_a = ["P0", "P1", "same"]
    strs_b = ["P0", "P1", "same"]
    rows = _compute_rows(
        _rows(logp_a), _rows(logp_b), ids_a, ids_b, strs_a, strs_b, prefix_len,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.aligned is True
    assert r.rank_changed is True
    # approx_kl should be strictly positive when the argmaxes diverge.
    assert r.approx_kl is not None and r.approx_kl > 0


def test_compute_rows_empty_when_no_assistant_tokens():
    logp = _build_logp([[0.0, 0.0]])
    rows = _compute_rows(
        _rows(logp), _rows(logp), [0], [0], ["P"], ["P"], prefix_len=1,
    )
    assert rows == []


# ---------------------------------------------------------------------------
# IO wrapper — verified against an in-memory mock session.
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Whitespace-split tokenizer with stable ids.

    Just enough surface area to feed ``build_chat_input``'s base-model
    fallback branch (no ``chat_template`` attribute) plus
    the replayer's ``decode([id])`` calls.
    """

    chat_template = None  # forces the base-model branch in build_chat_input

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}

    def _intern(self, tok: str) -> int:
        if tok not in self._vocab:
            tid = len(self._vocab)
            self._vocab[tok] = tid
            self._rev[tid] = tok
        return self._vocab[tok]

    def __call__(self, text: str, return_tensors: str = "pt"):
        # Whitespace split — good enough for the assertion shape.  The
        # base-model branch of ``build_chat_input`` calls this; the
        # tensor shape it expects is ``{"input_ids": Tensor[1, T]}``.
        toks = text.split()
        ids = [self._intern(t) for t in toks]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids: Any) -> str:
        if isinstance(ids, list):
            return " ".join(self._rev.get(int(i), "<unk>") for i in ids)
        return self._rev.get(int(ids), "<unk>")


def _MockModel(vocab: int) -> FakeLogitsModel:
    """Returns a constant log-uniform distribution at every position.

    All chosen-token logprobs equal ``-log(vocab)``; cross-evaluation
    matches self-evaluation exactly (KL is zero everywhere).  The shared
    ``FakeLogitsModel`` (conftest) supplies the ``.logits`` wrapper and the
    throwaway ``parameters()`` that ``next(model.parameters()).device`` needs;
    the flat-logits ``logits_fn`` is the only thing specific to this test.
    """

    def _logits(input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[-1]
        # Logits flat across vocab → log_softmax = -log(vocab).
        return torch.zeros((1, T, vocab), dtype=torch.float32)

    return FakeLogitsModel(_logits)


class _MockTree:
    """Minimal LoomTree surface the joint-logprob path consumes."""

    def __init__(self):
        from drowse.loom.tree import LoomNode

        # parent (user node) and two assistant children with different
        # assistant text.  The base-model fallback in build_chat_input
        # serializes via ``render_scene_raw`` ("Role: content" lines).
        # The shared "end" tail gives the byte-offset walker a genuinely
        # aligned position after the a/b divergence (the closed render
        # carries no trailing generation header to align on).
        root = LoomNode(id="root", parent_id=None, role="system", text="")
        user = LoomNode(id="u1", parent_id="root", role="user", text="ask")
        a1 = LoomNode(id="a1", parent_id="u1", role="assistant", text="hello a end")
        a2 = LoomNode(id="a2", parent_id="u1", role="assistant", text="hello b end")
        self.nodes = {"root": root, "u1": user, "a1": a1, "a2": a2}

    def messages_for(
        self, leaf_id: str, *, include_system: bool = False,
        with_labels: bool = False,
    ):
        from drowse.loom.tree import LoomNode

        # Walk parent chain; return [{role, content}], skip synthetic
        # root unless include_system is true.
        path: list[LoomNode] = []
        cur = leaf_id
        while cur is not None:
            n = self.nodes[cur]
            path.append(n)
            cur = n.parent_id
        path.reverse()
        out: list[dict[str, str]] = []
        for n in path:
            if n.id == "root" and not include_system:
                continue
            out.append({"role": n.role, "content": n.text})
        return out


class _MockSteering:
    """Steering manager surface: trigger ctx + the static-path predicates."""

    def __init__(self) -> None:
        self.ctx = TriggerContext()
        self.static = True

    def all_fast_path(self) -> bool:
        return self.static

    def static_steerable(self) -> bool:
        return False


class _MockSession:
    """Just enough surface for a :class:`JointLogprobReplayer` to run
    (:meth:`replayer` builds one from the mock's current attributes).

    Carries REAL instruments and a real read plane over them, so the
    replay's run lifecycle is exercised: ``begin`` here binds all three
    families exactly like the real planner does (without attaching capture),
    which lets the lifecycle regression assert the replay's ``finally``
    closes them (the former no-op stubs made the bind leak invisible to this
    suite)."""

    def __init__(self):

        from tests.support.geometry import stub_geometry_instrument
        from tests.support.read_plane import make_read_plane
        from tests.support.sae import stub_sae_instrument
        from drowse.readout.types import ReadRequest

        self.tokenizer = _MockTokenizer()
        # Build a vocabulary that covers the strings we'll feed.  The
        # decode path needs every id to round-trip.
        for word in ("User:", "ask", "Assistant:", "hello", "a", "b", "end"):
            self.tokenizer._intern(word)
        self._model = _MockModel(vocab=len(self.tokenizer._vocab) + 16)
        self.model = self._model
        self.chat_traits = ChatTraits(self.tokenizer, self._model)
        self.tree = _MockTree()
        self.config = GenerationConfig()
        self.profiles: dict[str, Any] = {}
        self._steering = _MockSteering()
        self._monitor = type(
            "MonitorState", (),
            {
                "begin_live": lambda _self: None,
                "end_live": lambda _self: None,
                "probe_names": [],
            },
        )()
        self._engine = SimpleNamespace(gen_config=lambda _sampling: self.config)
        self._geometry_instrument = stub_geometry_instrument(self)
        self._lens_instrument = stub_lens_instrument(self)
        self._sae_instrument = stub_sae_instrument(self)

        request = ReadRequest(final_aggregate=False)
        plane = make_read_plane(
            capture=SimpleNamespace(detach=lambda: None),
            monitor=self._monitor,
            geometry=self._geometry_instrument,
            lens=self._lens_instrument,
            sae=self._sae_instrument,
        )
        plane.resolve_demand = lambda **_kwargs: None
        plane.begin = lambda *_args, **_kwargs: (
            plane.bind_runs(request, request, request) and False
        )
        self._read_plane = plane
        self._steering_composer = SimpleNamespace(
            steering_needs_probe_gating=lambda: False,
        )
        self.renderer = self.mock_renderer(self._prepare_input)

    def mock_renderer(self, prepare_input: Any) -> Any:
        """The renderer surface the replay reads: the prompt render, the
        model type, the scene grammar and the default system prompt."""

        from drowse.generation.sampling import InheritSystemPrompt

        return SimpleNamespace(
            prepare_input=prepare_input, model_type=None, scene_grammar=None,
            resolve_system_prompt=lambda prompt: (
                self.config.system_prompt
                if isinstance(prompt, InheritSystemPrompt) else prompt
            ),
        )

    def replayer(self) -> Any:
        """A real replayer over this mock's current collaborators."""
        from drowse.loom.joint_logprobs import JointLogprobReplayer

        return JointLogprobReplayer(
            tree=lambda: self.tree,
            model=self.model,
            tokenizer=self.tokenizer,
            traits=self.chat_traits,
            renderer=self.renderer,
            profile_names=lambda: self.profiles,
            gen_config=lambda sampling: self._engine.gen_config(sampling),
            open_scope=lambda value: self.steering(value),  # type: ignore[attr-defined]
            steering=self._steering,
            needs_probe_gating=lambda: (
                self._steering_composer.steering_needs_probe_gating()
            ),
            reads=self._read_plane,
        )

    def _prepare_input(self, _input: Any, **kwargs: Any) -> torch.Tensor:
        node_id = kwargs["parent_node_id"]
        return build_chat_input(
            self.tokenizer, self.tree.messages_for(node_id),
            system_prompt=kwargs.get("system_prompt"),
            add_generation_prompt=True,
        )

    def force_stepwise_gated(self, gate: Any = None) -> None:
        """Route replay through the per-step gated path."""

        self._steering.static = False
        self._steering_composer = SimpleNamespace(
            steering_needs_probe_gating=lambda: True,
        )
        self._read_plane.gate_score_callback = lambda: gate or (lambda _step: {})

    def assert_runs_closed(self) -> None:
        for instrument in (
            self._geometry_instrument,
            self._lens_instrument,
            self._sae_instrument,
        ):
            assert instrument.current_run.bound is False, instrument.family


def test_gated_replay_passes_the_forward_step_to_the_gate_callback():
    """The replay's gate callback receives the loop's ``forced_idx`` — the
    same per-forward step identity the live decode loop threads.  A
    zero-arg call TypeErrors under the step-aware contract (sol's observe
    round-2 P1: the rest of this suite never exercises gated steering, so
    the regression was invisible here)."""
    session: Any = _MockSession()
    steps: list[int] = []

    def _gate(step_id: int) -> dict[str, float]:
        steps.append(step_id)
        return {}

    session.force_stepwise_gated(_gate)

    result = session.replayer().compute("a1", "a2")

    assert result.a_id == "a1"
    assert steps and steps[0] == 0
    # Each branch's replay counts its own forwards 0, 1, 2, … — this
    # two-branch fixture must show exactly two resets (one per branch),
    # with no skips or repeats within a branch.
    resets = [i for i, s in enumerate(steps) if s == 0]
    assert len(resets) == 2
    for start, end in zip(resets, resets[1:] + [len(steps)]):
        assert steps[start:end] == list(range(end - start))


@pytest.mark.parametrize("offset", [-1000., 0., 1000.])
def test_joint_replay_preserves_finite_logit_offsets(offset: float):
    session: Any = _MockSession()
    vocab = len(session.tokenizer._vocab) + 16
    logits = torch.arange(vocab, dtype=torch.float32) / 4
    session.config = GenerationConfig(top_p=1., top_k=None)
    session.model = session._model = FakeLogitsModel(
        lambda ids: (logits + offset).expand(1, ids.shape[1], vocab).clone(),
    )
    result = session.replayer().compute("a1", "a2")
    expected = logits.log_softmax(-1)
    for row in result.rows:
        if row.a_text:
            tid = session.tokenizer._vocab[row.a_text.strip()]
            assert row.lp_a_in_a == pytest.approx(float(expected[tid]), abs=1e-6)
        if row.b_text:
            tid = session.tokenizer._vocab[row.b_text.strip()]
            assert row.lp_b_in_b == pytest.approx(float(expected[tid]), abs=1e-6)
    session.assert_runs_closed()


def test_joint_logprobs_runs_end_to_end_on_mock():
    session: Any = _MockSession()
    result = session.replayer().compute("a1", "a2")
    assert result.a_id == "a1"
    assert result.b_id == "a2"
    assert result.parent_id == "u1"
    # Both branches: prefix is identical up to "hello ", then diverges
    # ("a" vs "b").  At least one row should be aligned ("hello"),
    # and there should be at least one row where the byte alignment
    # fails ("a" vs "b").
    assert any(r.aligned for r in result.rows)
    # Under the log-uniform mock, every chosen logprob equals
    # -log(vocab).  Aligned rows therefore have lp_a_in_a == lp_a_in_b.
    aligned_rows = [r for r in result.rows if r.aligned]
    assert aligned_rows  # at least the "hello" position aligns
    for r in aligned_rows:
        assert r.lp_a_in_a is not None and r.lp_a_in_b is not None
        assert abs(r.lp_a_in_a - r.lp_a_in_b) < 1e-5
        # KL between identical distributions is ~0.
        assert r.approx_kl is not None and abs(r.approx_kl) < 1e-5
        # Argmax is the same on both sides (any token is tied) — the
        # ``argmax`` resolves deterministically to id 0, so rank_changed
        # is False.
        assert r.rank_changed is False
    # n_rank1_changed mirrors the aligned-row count where rank flipped.
    assert result.n_rank1_changed == 0
    # The replay is a full capture transaction: every family's run must be
    # closed on the way out (a leaked bound run pins a stale lens between
    # generations and freezes SAE units past their generation).
    session.assert_runs_closed()


def test_joint_logprobs_closes_runs_on_replay_failure():
    """A mid-replay model failure must still close every bound run — the
    bind happens inside the replay ``try`` so even a partial-bind failure
    reaches the teardown."""
    session: Any = _MockSession()
    session.force_stepwise_gated()

    def _boom(_input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("mid-replay model failure")

    session._model = FakeLogitsModel(_boom)
    session.model = session._model
    with pytest.raises(RuntimeError, match="mid-replay"):
        session.replayer().compute("a1", "a2")
    session.assert_runs_closed()


def test_joint_logprobs_closes_runs_when_detach_raises():
    """Teardown is exception-hard: a raising hook detach must not skip run
    closure (a leaked bound run pins a stale lens between generations)."""
    session: Any = _MockSession()
    session.force_stepwise_gated()

    def _detach_boom() -> None:
        raise RuntimeError("detach failed")

    session._read_plane.capture.detach = _detach_boom
    with pytest.raises(RuntimeError, match="detach failed"):
        session.replayer().compute("a1", "a2")
    session.assert_runs_closed()


def test_joint_logprobs_to_dict_round_trip():
    session: Any = _MockSession()
    result = session.replayer().compute("a1", "a2")
    payload = result.to_dict()
    assert payload["a_id"] == "a1" and payload["b_id"] == "a2"
    assert payload["parent_id"] == "u1"
    assert payload["n_rank1_changed"] == result.n_rank1_changed
    assert len(payload["rows"]) == len(result.rows)
    # Every row has every expected key.
    expected_keys = {
        "a_index", "b_index", "a_text", "b_text", "aligned",
        "lp_a_in_a", "lp_b_in_b", "lp_a_in_b", "lp_b_in_a",
        "rank_changed", "approx_kl",
    }
    for row in payload["rows"]:
        assert set(row.keys()) == expected_keys


def test_branch_inputs_rebuilds_generated_human_seat_prompt():
    session: Any = _MockSession()
    node = session.tree.nodes["a1"]
    node.role = "user"
    node.role_label = "narrator"
    node.tokens = [{"token_id": 2, "text": "reply"}]
    seen: dict[str, Any] = {}

    def prepare_input(input: Any, **kwargs: Any) -> torch.Tensor:
        seen["input"] = input
        seen.update(kwargs)
        return torch.tensor([[0, 1]], dtype=torch.long)

    session.renderer = session.mock_renderer(prepare_input)
    branch = session.replayer()._branch_inputs(session.tree, "a1")

    assert branch.prompt_ids == [0, 1]
    assert branch.response_ids == [2]
    assert seen["input"] is None
    assert seen["parent_node_id"] == "u1"
    assert seen["gen_seat"] == "user"
    assert seen["user_role"] == "narrator"
    assert seen["assistant_role"] is None
    assert seen["to_device"] is False


def _causal_model(vocab: int, calls: list[int]) -> FakeLogitsModel:
    """Each position's logits depend on the whole causal prefix."""

    def _logits(input_ids: torch.Tensor) -> torch.Tensor:
        calls.append(int(input_ids.shape[1]))
        prefix = torch.cumsum(input_ids[0].to(torch.float32) * 1.7, dim=0)
        grid = torch.arange(vocab, dtype=torch.float32)
        return torch.sin(prefix[:, None] * 0.3 + grid[None, :]).unsqueeze(0) * 3

    return FakeLogitsModel(_logits)


def _long_branch_session(calls: list[int]) -> Any:
    session: Any = _MockSession()
    words = [f"w{i}" for i in range(40)]
    for word in words:
        session.tokenizer._intern(word)
    session.tree.nodes["a1"].text = "hello " + " ".join(words[:20]) + " end"
    session.tree.nodes["a2"].text = "hello " + " ".join(words[20:]) + " end"
    session.model = session._model = _causal_model(
        len(session.tokenizer._vocab) + 4, calls,
    )
    session.config = GenerationConfig(temperature=0.8, top_k=6, top_p=0.9)
    return session


def test_teacher_forced_replay_matches_stepwise_replay():
    """The chunked teacher-forced path reproduces the per-step replay."""
    forced_calls: list[int] = []
    forced = _long_branch_session(forced_calls).replayer().compute("a1", "a2")
    step_calls: list[int] = []
    stepwise_session = _long_branch_session(step_calls)
    stepwise_session._steering.static = False
    stepwise = stepwise_session.replayer().compute("a1", "a2")

    assert forced.rows == stepwise.rows
    assert any(r.lp_a_in_a is not None for r in forced.rows)
    # One forward per branch versus one per forced token.
    assert len(forced_calls) == 2
    assert len(step_calls) > 40


def test_replay_keeps_only_the_sampler_pool_per_row():
    session = _long_branch_session([])
    replayer = session.replayer()
    branch = replayer._branch_inputs(session.tree, "a1")
    rows = replayer._replay(branch)

    assert rows.ids.shape == (len(branch.response_ids), 6)   # top_k pool, not vocab
    assert rows.ids.device.type == "cpu"
    first = len(branch.prompt_ids) - 1
    assert sorted(rows.index) == list(
        range(first, first + len(branch.response_ids)),
    )
