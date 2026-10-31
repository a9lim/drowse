"""Representative chat templates and a Jinja-rendering fake tokenizer.

Each ``*_TEMPLATE`` emits the exact byte sequence its family's role header
uses, so role-substitution and scene-grammar code meets the bytes it would
meet in production; :class:`FakeTokenizer` renders them with plain Jinja and
tokenizes on whitespace (ids are arbitrary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jinja2.sandbox import ImmutableSandboxedEnvironment


# ---------------------------------------------------------------------------
# Per-family representative chat templates
# ---------------------------------------------------------------------------


# Mirrors the Qwen2/Qwen3 chat-template shape: ``<|im_start|>{role}\n{content}<|im_end|>``.
QWEN_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


# Exact chat-template bytes from the pinned
# HuggingFaceTB/SmolLM2-360M-Instruct@a10cc151 tokenizer config.  SmolLM2 is
# model_type ``llama`` but deliberately uses ChatML turn headers.
SMOLLM2_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, "
    "trained by Hugging Face<|im_end|>\n' }}"
    "{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
    "'<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


# Gemma 2/3: ``<start_of_turn>{role}\n{content}<end_of_turn>`` — and the
# assistant role is labeled ``model``, not ``assistant``.
GEMMA_TEMPLATE = (
    "{% for m in messages %}"
    "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "<start_of_turn>{{ role }}\n{{ m['content'] }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


# Gemma 4: same ``model`` label, new turn delimiters
# (``<|turn>...<turn|>`` instead of ``<start_of_turn>...<end_of_turn>``).
GEMMA4_TEMPLATE = (
    "{% for m in messages %}"
    "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "<|turn>{{ role }}\n{{ m['content'] }}<turn|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)


# Llama-3: ``<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>``.
LLAMA_TEMPLATE = (
    "{% for m in messages %}"
    "<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


# GLM-4 chat-template-ish: ``<|{role}|>\n{content}``.
GLM_TEMPLATE = (
    "{% for m in messages %}"
    "<|{{ m['role'] }}|>\n{{ m['content'] }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


# Talkie-1930: ``<|role|>content<|end|>`` — same <|role|> shape as GLM but no
# post-marker newline.  Faithful to the real chat template captured from
# a9lim/talkie-1930-13b-it-hf-cached (model_type "talkie").
TALKIE_TEMPLATE = (
    "{% for m in messages %}"
    "<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


# GPT-OSS: ``<|start|>{role}<|channel|>{content}<|end|>``.
GPT_OSS_TEMPLATE = (
    "{% for m in messages %}"
    "<|start|>{{ m['role'] }}<|channel|>{{ m['content'] }}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant<|channel|>{% endif %}"
)


# Mistral-3-ish: positional [INST] / [/INST], no role label in the rendered
# string.  Included to test that the registry's None entry raises rather
# than silently no-ops.
MISTRAL_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}[INST] {{ m['content'] }} [/INST]"
    "{% else %}{{ m['content'] }}{% endif %}"
    "{% endfor %}"
)


# ---------------------------------------------------------------------------
# FakeTokenizer — minimal apply_chat_template + __call__ surface
# ---------------------------------------------------------------------------


@dataclass
class _FakeBatch:
    input_ids: Any  # list[int] or Tensor depending on return_tensors

    def __getitem__(self, key: str) -> Any:
        if key == "input_ids":
            return self.input_ids
        raise KeyError(key)


class FakeTokenizer:
    """Minimal stand-in for an HF tokenizer.

    Implements just enough of the surface that
    :func:`apply_with_role` needs: ``apply_chat_template`` (Jinja-rendered)
    and ``__call__`` (whitespace tokenization, deterministic id assignment).
    """

    def __init__(self, template: str):
        self.chat_template = template
        # Rendered the way transformers renders chat templates: sandboxed,
        # with no HTML escaping (a chat template is not markup).
        self._template = ImmutableSandboxedEnvironment().from_string(template)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        rendered = self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        if not tokenize:
            return rendered
        return self._tokenize(rendered, return_tensors=return_tensors)

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> _FakeBatch:
        return _FakeBatch(input_ids=self._tokenize(text, return_tensors=return_tensors))

    @staticmethod
    def _tokenize(text: str, *, return_tensors: str | None = None) -> Any:
        # Stable deterministic mapping: token id is the hash of each
        # whitespace-split chunk mod a small constant.  Real values
        # don't matter for these tests.
        ids = [hash(chunk) % 50_000 for chunk in text.split()]
        if return_tensors == "pt":
            import torch

            return torch.tensor([ids])
        return ids


# Faithful-strict gemma: <bos> prelude, system folded into the first user
# turn (the real gemma has no system role), and non-alternating roles raise
# (``raise_exception`` is undefined in the plain jinja2 environment, so
# calling it throws — the same observable behavior as the real template).
GEMMA_STRICT_TEMPLATE = (
    "<bos>"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set sys = messages[0]['content'] %}{% set msgs = messages[1:] %}"
    "{% else %}{% set sys = '' %}{% set msgs = messages %}{% endif %}"
    "{% for m in msgs %}"
    "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "{% set expected = 'user' if loop.index0 % 2 == 0 else 'model' %}"
    "{% if role != expected %}"
    "{{ raise_exception('roles must alternate') }}"
    "{% endif %}"
    "<start_of_turn>{{ role }}\n"
    "{% if loop.first and sys %}{{ sys }}\n\n{% endif %}"
    "{{ m['content'] }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


# Qwen3-style: strips think blocks from assistant history and appends an
# empty think scaffold to the generation prompt (the gen_extra case).
QWEN_STRIP_TEMPLATE = (
    "{% for m in messages %}"
    "{% set c = m['content'] %}"
    "{% if m['role'] == 'assistant' and '</think>' in c %}"
    "{% set c = c.split('</think>')[-1] %}"
    "{% endif %}"
    "<|im_start|>{{ m['role'] }}\n{{ c }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "{% endif %}"
)
