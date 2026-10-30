"""Conversational corpus generation — the model answers in character.

A manifold node's corpus is the model's own answers to the shared baseline
prompts, elicited *in character*: the concept rides the system prompt
(:func:`system_for`, led by the shared
:data:`~drowse.fit.capture._LENGTH_DIRECTIVE`) and a swapped assistant-role
label (:func:`role_for`).  The neutral baseline is the same loop with the
length directive as the whole system prompt and the standard assistant label.
Responses come back samples-outer / prompts-inner, so ``response[i]`` answers
``baseline_prompt[i % k]`` — the alignment the capture and the node corpus
files assume.

:class:`CorpusGenerator` owns that one loop.  Its dependencies are a
``model.generate``-shaped callable and the tokenizer-side render facts (chat
traits, the validated scene grammar, the role ``model_type``), never a session.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from drowse.fit import capture as _capture
from drowse.generation.decode import build_chat_input
from drowse.naming import BIPOLAR_SEP, _slug
from drowse.store.manifold_authoring import append_discover_manifold_node

if TYPE_CHECKING:
    from drowse.model.chat import ChatTraits
    from drowse.model.scene import TurnGrammar
    from drowse.store.manifold_folder import DiscoverGenerationPlan

# Token cap per in-character response.
RESPONSE_MAX_TOKENS = 256
# Baseline prompts per batched ``generate`` call.  One left-padded batch over
# this many prompts replaces that many sequential batch-1 decodes, the
# dominant cost of the corpus-generation phase.
CORPUS_GEN_BATCH = 16
# An EOS-only sampled row carries no conversational evidence and cannot be
# installed as a manifold node.  Retry only that aligned row greedily while
# suppressing EOS long enough to require a short content continuation.
CORPUS_RETRY_MIN_TOKENS = 8

# Elicitation templates keyed by node ``kind``.  The concept goes in the system
# prompt; {c} is the humanized concept, {art} its a/an article.  ``abstract``
# traits read as "someone {c}", ``concrete`` entities as "{art} {c}".
KIND_TEMPLATES = {
    "abstract": "You are someone {c}. Respond exactly as someone {c} would.",
    "concrete": "You are {art} {c}. Respond exactly as {art} {c} would.",
}


def humanize_concept(name: str) -> str:
    """Invert the slug ``_`` convention for LLM-facing prompts.

    Manifold names use underscores (``artificial_intelligence``); the model
    reads them better as spaces.  Disk paths, canonical names, and progress
    messages keep the slug form.
    """
    return name.replace("_", " ")


def article(word: str) -> str:
    """Naive a/an for the concrete-entity template (an alien, a deer)."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def system_for(concept_h: str, kind: str | None, custom: str | None = None) -> str:
    """The elicitation system prompt for a humanized concept under its kind.

    ``abstract`` (the default) reads as "someone {c}", ``concrete`` as
    "{art} {c}", and ``custom`` formats the caller-supplied ``custom`` template
    (``{c}``/``{art}`` placeholders): a free-form frame for a concept that is
    neither a trait nor an entity, such as a month embodying its season.
    """
    if (kind or "abstract") == "custom":
        if not custom:
            raise ValueError("system_for: kind='custom' requires a custom system template")
        return custom.format(c=concept_h, art=article(concept_h))
    template = KIND_TEMPLATES.get(kind or "abstract", KIND_TEMPLATES["abstract"])
    return template.format(c=concept_h, art=article(concept_h))


def role_for(slug: str, kind: str | None) -> str | None:
    """The elicitation role label (the swapped assistant header) for a node.

    Abstract traits get a ``someone_{slug}`` speaker ("someone happy");
    concrete entities are the bare slug ("pirate").  The underscore de-slugs to
    a space at render (:func:`drowse.model.role_templates._render_label`).
    ``custom`` returns ``None``: no role swap, the custom system prompt carries
    the framing, and the corpus pools in standard-assistant space.
    """
    k = kind or "abstract"
    if k == "custom":
        return None
    return f"someone_{slug}" if k == "abstract" else slug


def split_composite_source(concept: str, baseline: str | None) -> tuple[str, str | None]:
    """Split a composite ``pos.neg`` slug when no explicit baseline is given.

    ``canonical_concept_name`` performs the same split for the storage name;
    corpus generation needs it too, so ``pos`` and ``neg`` are elicited as two
    distinct poles in the declared order rather than one composite blob.
    """
    if baseline is None and BIPOLAR_SEP in concept:
        pos, neg = concept.split(BIPOLAR_SEP, 1)
        return pos.strip(), neg.strip()
    return concept, baseline


class CorpusGenerator:
    """Generate node and neutral corpora by batched in-character decoding.

    ``generate`` is a ``model.generate``-shaped callable
    (``generate(input_ids, attention_mask=..., **kwargs) -> ids``).
    ``tokenizer``, ``traits`` (the render cache), ``scene`` (a provider for the
    validated turn grammar, resolved lazily by its owner) and ``model_type``
    (the role-header family) are what :func:`build_chat_input` needs to render
    each ``[system, user]`` prompt under a swapped assistant label.  The caller
    owns any model-exclusivity locking.
    """

    def __init__(
        self,
        generate: Callable[..., torch.Tensor],
        tokenizer: Any,
        *,
        device: torch.device,
        traits: "ChatTraits | None" = None,
        scene: "Callable[[], TurnGrammar | None]" = lambda: None,
        model_type: str | None = None,
    ) -> None:
        self._generate = generate
        self._tokenizer = tokenizer
        self._device = device
        self._traits = traits
        self._scene = scene
        self._model_type = model_type

    def responses(
        self,
        concepts: Sequence[str],
        kinds: Sequence[str | None],
        *,
        roles: dict[str, str | None] | None = None,
        custom_system: str | None = None,
        samples_per_prompt: int = 1,
        max_new_tokens: int = RESPONSE_MAX_TOKENS,
        seed: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, list[str]]:
        """Each concept's in-character corpus, keyed by concept.

        The system prompt is the length directive followed by
        :func:`system_for`; the generation label is ``roles[concept]`` when
        given, else :func:`role_for`.  ``custom`` kinds format
        ``custom_system`` and do not role-swap, so they work on every family;
        ``abstract``/``concrete`` raise ``RoleSubstitutionUnsupportedError`` on
        a family without a substitutable role header.  Each list holds
        ``max(1, samples_per_prompt) * len(baseline_prompts)`` responses.
        ``seed`` seeds the sampler once before the first concept, so a seeded
        call reproduces its corpora on the same device.
        """
        if any((k or "abstract") == "custom" for k in kinds) and not custom_system:
            raise ValueError(
                "generate_responses: kind='custom' requires custom_system "
                "(a system template with a {c} placeholder)"
            )
        if seed is not None:
            torch.manual_seed(seed)
        roles = roles or {}
        out: dict[str, list[str]] = {}
        for concept, kind in zip(concepts, kinds, strict=True):
            # The length directive leads the persona system prompt (and is the
            # whole system for the neutral baseline and for capture), so it is
            # common-mode and cancels at extraction.
            system = (
                f"{_capture._LENGTH_DIRECTIVE} "
                f"{system_for(humanize_concept(concept), kind, custom_system)}"
            )
            role = roles.get(concept) or role_for(_slug(concept), kind)
            out[concept] = self._corpus(
                system, role, repr(concept),
                samples_per_prompt=samples_per_prompt,
                max_new_tokens=max_new_tokens, on_progress=on_progress,
            )
        return out

    def fill(
        self,
        plan: "DiscoverGenerationPlan",
        *,
        custom_system: str | None = None,
        samples_per_prompt: int = 1,
        seed: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Generate and write the corpus of every pending node of a discover plan.

        Nodes run in node order, each under its manifest ``kind`` and ``role``
        (:attr:`DiscoverGenerationPlan.kinds` / ``roles``), and each lands on
        disk as soon as it is done: a run that dies part-way keeps the nodes it
        finished, and the next plan resumes from them.  With ``seed`` a node is
        seeded with ``seed + <its node index>``, so its corpus depends only on
        the seed and its position, and a resumed run writes what a full run
        would have.
        """
        total = len(plan.index_of)
        for label in plan.pending:
            index = plan.index_of[label]
            corpus = self.responses(
                [label], [plan.kinds[label]],
                roles={label: plan.roles[label]},
                custom_system=custom_system,
                samples_per_prompt=samples_per_prompt,
                seed=None if seed is None else seed + index,
                on_progress=on_progress,
            )[label]
            append_discover_manifold_node(plan.folder, index, label, corpus)
            if on_progress:
                on_progress(
                    f"Wrote node {label!r} ({index + 1}/{total}, "
                    f"{len(corpus)} responses)"
                )

    def neutral_responses(
        self,
        *,
        samples_per_prompt: int = 1,
        max_new_tokens: int = RESPONSE_MAX_TOKENS,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        """The neutral baseline corpus: the model's own brief voice.

        The length directive is the only system prompt and the assistant keeps
        its own label, so the directive is the only framing neutral shares with
        the node corpora and it cancels at extraction.
        """
        return self._corpus(
            _capture._LENGTH_DIRECTIVE, None, "neutral",
            samples_per_prompt=samples_per_prompt,
            max_new_tokens=max_new_tokens, on_progress=on_progress,
        )

    def _corpus(
        self,
        system: str,
        role: str | None,
        label: str,
        *,
        samples_per_prompt: int,
        max_new_tokens: int,
        on_progress: Callable[[str], None] | None,
    ) -> list[str]:
        """One elicitation frame over the baseline prompts, samples-outer."""
        prompts = _capture._load_baseline_prompts()
        reps = max(1, samples_per_prompt)
        total = reps * len(prompts)
        responses: list[str] = []
        for _ in range(reps):
            # Prompts-inner in chunks: order is preserved within the rep, so
            # ``response[i] -> prompt[i % k]`` holds.
            for start in range(0, len(prompts), CORPUS_GEN_BATCH):
                chunk = prompts[start:start + CORPUS_GEN_BATCH]
                if on_progress:
                    on_progress(
                        f"Generating {label} responses "
                        f"{len(responses) + 1}-{len(responses) + len(chunk)}"
                        f"/{total}..."
                    )
                responses.extend(
                    self._repaired_batch(system, chunk, max_new_tokens, role=role)
                )
        return responses

    def _repaired_batch(
        self, system: str, prompts: list[str], max_new_tokens: int, *, role: str | None,
    ) -> list[str]:
        """Generate one aligned chunk, retrying EOS-only rows in place.

        Sampling can legitimately choose EOS as its first token for one row of
        an otherwise healthy batch.  Dropping the row would shift every later
        response off its baseline prompt; keeping it would install a blank
        node corpus.  So exactly that row is retried under the same system
        prompt, user prompt and role, greedily with a short EOS-suppression
        window: still model output under the identical frame, and the other
        rows' order and sampling RNG are untouched.
        """
        texts = self._batch(system, prompts, max_new_tokens, role=role)
        if len(texts) != len(prompts):
            raise RuntimeError(
                "corpus generation returned a different number of responses "
                f"than prompts ({len(texts)} != {len(prompts)})"
            )
        repaired: list[str] = []
        for prompt, text in zip(prompts, texts, strict=True):
            if text.strip():
                repaired.append(text.strip())
                continue
            retry = self._batch(
                system, [prompt], max_new_tokens, role=role,
                deterministic=True, min_new_tokens=CORPUS_RETRY_MIN_TOKENS,
            )
            if len(retry) != 1 or not retry[0].strip():
                raise RuntimeError(
                    f"the model produced no content for corpus prompt {prompt!r}, "
                    "including the deterministic retry"
                )
            repaired.append(retry[0].strip())
        return repaired

    def _batch(
        self,
        system: str,
        prompts: list[str],
        max_new_tokens: int,
        *,
        role: str | None,
        deterministic: bool = False,
        min_new_tokens: int = 0,
    ) -> list[str]:
        """One left-padded ``generate`` over ``[system?, user]`` chats.

        Decoder-only generation aligns continuations on the right, so every
        row's prompt must end at the same column.  Sampling is per-row
        independent at ``temperature=1.0`` / ``top_p=0.9``; ``deterministic``
        switches to greedy decoding with an optional ``min_new_tokens`` floor.
        """
        if not prompts:
            return []
        # The tokenizer stubs type pad/eos ids loosely (``int | list[int] |
        # str``); the id is an int.
        pad_id = cast(
            int,
            self._tokenizer.pad_token_id or self._tokenizer.eos_token_id or 0,
        )
        # ``model_type`` only matters (and only enters the render cache key)
        # when a role label is substituted.
        model_type = self._model_type if role is not None else None
        scene = self._scene()
        seqs: list[torch.Tensor] = []
        for prompt in prompts:
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            ids = build_chat_input(
                self._tokenizer, messages, system_prompt=None, thinking=False,
                gen_role=role, model_type=model_type, scene=scene,
                traits=self._traits,
            )
            seqs.append(ids[0])

        lengths = [int(seq.shape[0]) for seq in seqs]
        max_len = max(lengths)
        # Left-pad: each row's prompt occupies its rightmost ``lengths[i]``
        # columns, so ``out[:, max_len:]`` is the batch's generated tail.
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=seqs[0].dtype)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, max_len - lengths[i]:] = seq
            attn[i, max_len - lengths[i]:] = 1

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": not deterministic,
            "pad_token_id": pad_id,
        }
        if deterministic:
            if min_new_tokens > 0:
                kwargs["min_new_tokens"] = min(min_new_tokens, max_new_tokens)
        else:
            kwargs.update(temperature=1.0, top_p=0.9)
        with torch.inference_mode():
            out = self._generate(
                input_ids.to(self._device), attention_mask=attn.to(self._device), **kwargs,
            )
        return list(self._tokenizer.batch_decode(out[:, max_len:], skip_special_tokens=True))
