"""Restricted-choice completion scoring — the logit read of a template.

Given a conversation context and a fixed set of candidate completions, score the
**conditional probability of each candidate** and report the distribution over the
set. This is forced-choice / multiple-choice scoring: the model's belief about
which slot-fill comes next, not a sampled token.

It is the logit counterpart to the manifold fit. The manifold pools each
candidate's slot-filled *activation* into a node; the scorer reads each
candidate's slot-filled *logprob*. Both run off the same
:class:`drowse.store.templates.TemplateFolder`.

Scoring is against the **raw** model distribution (temperature 1, no top-k/p
truncation), so the probabilities are the model's beliefs, not a reshaping by
sampler knobs. Multi-token candidates are handled two ways and both are reported:
``sum_logprob`` (the true joint ``log P(candidate | context)``) and
``mean_logprob`` (length-normalized), each with its own restricted-choice softmax
over the candidate set — length bias is real when candidates differ in token
count, so neither view is silently chosen for you.

An optional ``steering=`` runs the scoring forward under a steering expression, so
the project's core question — *did steering shift the distribution, not just the
argmax?* — answers directly: ``P("one second")`` before vs after a steer.

Every forward runs inside ``session._model_exclusive`` (a concurrent generation
or fit in another thread would otherwise race the steering hooks and the MPS
command buffer); the steering scope opens *outside* that guard, because the
composer's push/pop take the same non-reentrant lock.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from drowse.errors import GenerationRequestError
from drowse.generation.decode import build_chat_input
from drowse.model.chat import logits_keep_kwarg

if TYPE_CHECKING:  # avoid an import cycle at module load
    from drowse.session.session import DrowseSession
    from drowse.store.templates import TemplateFolder

# Choices are scored in chunks of this many rows per forward.  The forward keeps
# only the completion-producing tail of the logits (``logits_to_keep``), so the
# chunk bounds the ``[n, tail, vocab]`` head output, not the full prompt.
_SCORE_BATCH = 16


@dataclass(frozen=True)
class ChoiceScore:
    """One candidate's score within a :class:`ChoiceScores` set."""

    text: str
    label: str
    token_ids: tuple[int, ...]
    n_tokens: int
    sum_logprob: float
    mean_logprob: float
    prob_sum: float       # restricted-choice softmax over the set's sum_logprobs
    prob_mean: float      # restricted-choice softmax over the set's mean_logprobs

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "n_tokens": self.n_tokens,
            "sum_logprob": self.sum_logprob,
            "mean_logprob": self.mean_logprob,
            "prob_sum": self.prob_sum,
            "prob_mean": self.prob_mean,
        }


@dataclass(frozen=True)
class ChoiceScores:
    """A restricted-choice distribution over one context's candidate set."""

    choices: tuple[ChoiceScore, ...]
    steering: str | None = None

    def ranked(self, by: str = "sum") -> list[ChoiceScore]:
        """Choices sorted most- to least-probable.

        ``by`` selects the statistic: ``"sum"`` (joint logprob, length-biased
        toward short completions) or ``"mean"`` (length-normalized).
        """
        key = (lambda c: c.sum_logprob) if by == "sum" else (lambda c: c.mean_logprob)
        return sorted(self.choices, key=key, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steering": self.steering,
            "choices": [c.to_dict() for c in self.choices],
        }


def shared_prefix_len(ids_a: list[int], ids_b: list[int]) -> int:
    """Longest common prefix length between two token-id lists.

    Recovers where two renders of the same context diverge — the scored span
    of a candidate (which may retokenize the last prefix token, ``"Monday"`` vs
    ``"Tuesday"``) and the divergence point of two loom siblings.
    """
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


def _tok_ids(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    enc = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(t) for t in enc]


def _pad_id(tokenizer: Any) -> int:
    pad: Any = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if isinstance(pad, (list, tuple)):
        pad = pad[0] if pad else None
    return int(pad) if pad is not None else 0


@contextlib.contextmanager
def _scoring_scope(
    session: "DrowseSession", steering: "str | Any | None",
) -> Iterator[None]:
    """Steering scope (outside) + exclusive model guard (inside)."""
    steering_cm: Any = (
        session.steering(steering) if steering is not None
        else contextlib.nullcontext()
    )
    with steering_cm, session._model_exclusive(
        "completion scoring called while another model use is in flight",
        phase_msg="completion scoring called while a generation is in flight",
    ):
        yield


def score_choices(
    session: "DrowseSession",
    messages: list[dict[str, str]],
    choices: list[str],
    *,
    assistant_prefix: str = "",
    labels: list[str] | None = None,
    steering: "str | Any | None" = None,
    system_prompt: str | None = None,
) -> ChoiceScores:
    """Score each candidate completion and return the restricted-choice set.

    ``messages`` is the conversation history (ending on a user turn);
    ``assistant_prefix`` is any assistant text *before* the slot (the model
    conditions on it); ``choices`` are the candidate slot-fills. Each candidate's
    completion span is recovered with :func:`shared_prefix_len`, which absorbs
    the boundary-token merge (``"Monday"`` vs ``"Tuesday"`` may retokenize the
    last prefix token). One batched teacher-forced forward per chunk, no sampling.
    """
    if not choices:
        raise GenerationRequestError("score_choices: empty choice set")
    labels = labels if labels is not None else list(choices)
    if len(labels) != len(choices):
        raise GenerationRequestError("score_choices: labels and choices length mismatch")
    with _scoring_scope(session, steering):
        return _score_choice_set(
            session, messages, choices, labels,
            assistant_prefix=assistant_prefix, system_prompt=system_prompt,
            steering=steering,
        )


def score_template(
    session: "DrowseSession",
    template: "TemplateFolder",
    *,
    steering: "str | Any | None" = None,
    system_prompt: str | None = None,
) -> list[ChoiceScores]:
    """Score a template's value set against each of its contexts.

    Returns one :class:`ChoiceScores` per context (the per-context shape the
    template artifact is designed around). One steering scope and one exclusive
    model section cover every context.
    """
    specs = template.score_inputs()
    for spec in specs:
        if not spec["choices"]:
            raise GenerationRequestError("score_choices: empty choice set")
    with _scoring_scope(session, steering):
        return [
            _score_choice_set(
                session, spec["messages"], spec["choices"], spec["labels"],
                assistant_prefix=spec["assistant_prefix"],
                system_prompt=system_prompt, steering=steering,
            )
            for spec in specs
        ]


def _score_choice_set(
    session: "DrowseSession",
    messages: list[dict[str, str]],
    choices: list[str],
    labels: list[str],
    *,
    assistant_prefix: str,
    system_prompt: str | None,
    steering: "str | Any | None",
) -> ChoiceScores:
    """The forward + gather for one context; the caller holds the model."""
    model = session.model
    tokenizer = session.tokenizer
    device = next(model.parameters()).device

    render = build_chat_input(
        tokenizer, messages, system_prompt=system_prompt, add_generation_prompt=True,
    )
    render_ids = [int(t) for t in render[0].tolist()]
    prefix_ids = render_ids + _tok_ids(tokenizer, assistant_prefix)

    # Per-choice full sequence + the start index of its scored completion span.
    # A cut of ``len(full)`` marks a degenerate candidate (no distinct token);
    # position 0 has no predictor, so a span never starts there.
    seqs: list[list[int]] = []
    cuts: list[int] = []
    for c in choices:
        full = render_ids + _tok_ids(tokenizer, assistant_prefix + c)
        seqs.append(full)
        cuts.append(max(1, shared_prefix_len(prefix_ids, full)))

    pad_id = _pad_id(tokenizer)
    keep_kwarg = logits_keep_kwarg(model)
    sum_lps = [0.0] * len(choices)
    n_toks = [max(0, len(seq) - cut) for seq, cut in zip(seqs, cuts)]

    scored = [i for i in range(len(choices)) if n_toks[i] > 0]
    with torch.inference_mode():
        for start in range(0, len(scored), _SCORE_BATCH):
            rows = scored[start:start + _SCORE_BATCH]
            max_len = max(len(seqs[i]) for i in rows)
            input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(rows), max_len), dtype=torch.long)
            pos: list[int] = []
            owner: list[int] = []
            target: list[int] = []
            for r, i in enumerate(rows):
                seq = seqs[i]
                input_ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                attn[r, :len(seq)] = 1
                # Logits at position p predict token p + 1.
                for p in range(cuts[i] - 1, len(seq) - 1):
                    pos.append(p)
                    owner.append(r)
                    target.append(seq[p + 1])
            first_needed = min(pos)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids.to(device),
                "attention_mask": attn.to(device),
                "use_cache": False,
            }
            if keep_kwarg is not None:
                kwargs[keep_kwarg] = max_len - first_needed
            logits = model(**kwargs).logits
            # ``logits[:, 0]`` sits at absolute position ``offset`` whether or
            # not the forward honored ``logits_to_keep``.
            offset = max_len - int(logits.shape[1])
            owner_t = torch.tensor(owner, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.long, device=device) - offset
            target_t = torch.tensor(target, dtype=torch.long, device=device)
            # Normalize only completion producers, in fp32, without subtracting
            # two large uncentered values to recover a small log probability.
            logp = torch.log_softmax(
                logits[owner_t, pos_t], dim=-1, dtype=torch.float32,
            )
            token_lp = logp.gather(1, target_t[:, None]).squeeze(1)
            sums = torch.zeros(len(rows), dtype=torch.float32, device=device)
            sums.index_add_(0, owner_t, token_lp)
            for i, value in zip(rows, sums.tolist()):
                sum_lps[i] = float(value)
            del logits, logp

    mean_lps = [sl / nt if nt else sl for sl, nt in zip(sum_lps, n_toks)]
    # Degenerate candidates (no distinct completion token — an empty-string
    # choice, or one fully absorbed by the prefix) carry ``sum_logprob`` 0.0,
    # which is the *largest* value in the restricted-choice softmax (every real
    # candidate's joint logprob is strictly negative) and would wrongly dominate
    # it.  Exclude them: -inf in the softmax input gives them ~0 probability,
    # with an all-degenerate guard (softmax over all -inf is NaN).
    degenerate = [nt == 0 for nt in n_toks]

    def _restricted_softmax(values: list[float]) -> list[float]:
        masked = [
            float("-inf") if deg else v
            for v, deg in zip(values, degenerate)
        ]
        if all(deg for deg in degenerate):
            return [0.0] * len(masked)
        return [
            float(p) for p in torch.softmax(
                torch.tensor(masked, dtype=torch.float32), dim=0,
            ).tolist()
        ]

    prob_sum = _restricted_softmax(sum_lps)
    prob_mean = _restricted_softmax(mean_lps)

    return ChoiceScores(
        choices=tuple(
            ChoiceScore(
                text=choices[i],
                label=labels[i],
                token_ids=tuple(seqs[i][cuts[i]:]),
                n_tokens=n_toks[i],
                sum_logprob=sum_lps[i],
                mean_logprob=mean_lps[i],
                prob_sum=prob_sum[i],
                prob_mean=prob_mean[i],
            )
            for i in range(len(choices))
        ),
        steering=None if steering is None else str(steering),
    )
