"""Candidate GPT-2/GPT-NeoX instrumentation for the pinned MLC compiler.

These adapters are intentionally separate from the attested production overlay.
Export validation is not a model release or a numerical inference certification.
"""

from functools import wraps

from tvm.relax.frontend import nn

from mlc_llm.model.gpt2.gpt2_model import GPT2Config, GPT2LMHeadModel, GPT2Model
from mlc_llm.model.gpt_neox.gpt_neox_model import GPTNeoXForCausalLM, GPTNeoXModel
from mlc_llm.model.qwen3.qwen3_model import Qwen3LMHeadModel, Qwen3Model


def _with_absolute_positions(method):
    @wraps(method)
    def forward(self, inputs, paged_kv_cache, *args):
        positions = paged_kv_cache.get_query_positions(inputs.shape[1])
        return method(self, inputs + self.wpe(positions), paged_kv_cache, *args)

    return forward


class InstrumentedGPT2Config(GPT2Config):
    @property
    def num_hidden_layers(self):
        return self.n_layer


class InstrumentedGPT2Model(GPT2Model):
    @property
    def layers(self):
        return self.h

    @property
    def norm(self):
        return self.ln_f

    def _drowse_prepare_inputs(self, inputs):
        return inputs


class InstrumentedGPTNeoXModel(GPTNeoXModel):
    @property
    def norm(self):
        return self.final_layer_norm

    def _drowse_prepare_inputs(self, inputs):
        return inputs


for _name, _method in vars(Qwen3Model).items():
    if _name.startswith("drowse_") and callable(_method):
        setattr(InstrumentedGPT2Model, _name, _with_absolute_positions(_method))
        setattr(InstrumentedGPTNeoXModel, _name, _method)


def _instrumented_spec(model, native_spec):
    hook_spec = Qwen3LMHeadModel.get_default_spec(model)
    methods = dict(zip(native_spec.method_names, native_spec.method_specs))
    methods.update(
        (name, spec)
        for name, spec in zip(hook_spec.method_names, hook_spec.method_specs)
        if name.startswith("drowse_")
    )
    return nn.spec.ModuleSpec.from_raw(methods, model)


class InstrumentedGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = InstrumentedGPT2Model(config)
        self.num_hidden_layers = config.n_layer
        self.hidden_size = config.n_embd
        self.vocab_size = config.vocab_size
        # The upstream GPT-2 loader materializes the tied head under lm_head.
        self.tie_word_embeddings = False

    @property
    def model(self):
        return self.transformer

    def get_default_spec(self):
        return _instrumented_spec(self, super().get_default_spec())


class InstrumentedGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = InstrumentedGPTNeoXModel(config)
        self.tie_word_embeddings = False

    @property
    def model(self):
        return self.gpt_neox

    @property
    def lm_head(self):
        return self.embed_out

    def get_default_spec(self):
        return _instrumented_spec(self, super().get_default_spec())


for _head in (InstrumentedGPT2LMHeadModel, InstrumentedGPTNeoXForCausalLM):
    for _name, _method in vars(Qwen3LMHeadModel).items():
        if callable(_method) and (
            _name.startswith(("drowse_", "_drowse_")) or _name == "_get_logits"
        ):
            setattr(_head, _name, _method)


CANDIDATE_MODELS = {
    "gpt2": InstrumentedGPT2LMHeadModel,
    "gpt_neox": InstrumentedGPTNeoXForCausalLM,
}


def register_candidate_models(*, include_hybrid=False):
    """Register candidates for MLC compile/convert without altering release locks."""
    from dataclasses import replace
    from mlc_llm.model.model import MODELS
    from mlc_llm.quantization import make_quantization_functions

    candidates = dict(CANDIDATE_MODELS)
    if include_hybrid:
        from qwen35_adapter import InstrumentedQwen35LMHeadModel
        candidates.update(qwen3_5=InstrumentedQwen35LMHeadModel, qwen3_5_text=InstrumentedQwen35LMHeadModel)
    for architecture, model in candidates.items():
        MODELS[architecture] = replace(
            MODELS[architecture], model=model,
            config=InstrumentedGPT2Config if architecture == "gpt2" else MODELS[architecture].config,
            quantize=make_quantization_functions(model, supports_ft_quant=False),
        )
