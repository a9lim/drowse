"""Candidate GPT-2/GPT-NeoX instrumentation for the pinned MLC compiler.

These adapters are intentionally separate from the attested production overlay.
Export validation is not a model release or a numerical inference certification.
"""

from functools import wraps
from math import prod

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tirx as T

from mlc_llm.model.gpt2.gpt2_model import GPT2Config, GPT2LMHeadModel, GPT2Model
from mlc_llm.model.gpt_neox.gpt_neox_model import GPTNeoXForCausalLM, GPTNeoXModel
from mlc_llm.model.qwen3.qwen3_model import Qwen3LMHeadModel, Qwen3Model


class PairwiseOutputLinear(nn.Linear):
    @classmethod
    def from_linear(cls, linear):
        result = cls(linear.in_features, linear.out_features, linear.bias is not None,
                     linear.weight.dtype, linear.out_dtype)
        result.weight = linear.weight
        result.bias = linear.bias
        return result

    def forward(self, x):
        if x.dtype != "float32" or self.weight.dtype != "float32" or self.out_dtype not in (None, "float32"):
            return super().forward(x)
        rows, hidden = prod(x.shape[:-1]), x.shape[-1]
        vocab = self.weight.shape[0]

        @T.prim_func(private=True, s_tir=True)
        def project(var_x: T.handle, var_weight: T.handle, var_output: T.handle):
            T.func_attr({"tirx.noalias": True, "tirx.is_scheduled": True})
            n = T.int64()
            v = T.int64()
            source = T.match_buffer(var_x, (n, hidden), "float32")
            weight = T.match_buffer(var_weight, (v, hidden), "float32")
            output = T.match_buffer(var_output, (n, v), "float32")
            partial = T.sblock_alloc_buffer((6, 128), "float32", scope="shared")
            correction = T.sblock_alloc_buffer((6, 128), "float32", scope="shared")
            high = T.sblock_alloc_buffer((1,), "float32", scope="local")
            low = T.sblock_alloc_buffer((1,), "float32", scope="local")
            for row in T.thread_binding(n, thread="blockIdx.y"):
                for tile in T.thread_binding(T.ceildiv(v, 4), thread="blockIdx.x"):
                    for thread in T.thread_binding(128, thread="threadIdx.x"):
                        with T.sblock("drowse_pairwise_output_projection"):
                            T.reads(source[row, 0:hidden], weight[0:v, 0:hidden])
                            T.writes(output[row, 0:v])
                            lane = T.meta_var(thread)
                            token = T.meta_var(tile * 4 + lane // 32)
                            column = T.meta_var(lane % 32)
                            high[0] = 0.0
                            low[0] = 0.0
                            if token < v:
                                for step in T.serial(T.ceildiv(hidden, 32)):
                                    k = T.meta_var(step * 32 + column)
                                    if k < hidden:
                                        product = source[row, k] * weight[token, k]
                                        combined = high[0] + product
                                        recovered = T.fma(-1.0, high[0], combined)
                                        error = T.fma(-1.0, T.fma(-1.0, recovered, combined), high[0]) + T.fma(-1.0, recovered, product)
                                        low[0] = low[0] + (error + T.fma(source[row, k], weight[token, k], -product))
                                        high[0] = combined
                            partial[0, lane] = high[0]
                            correction[0, lane] = low[0]
                            T.tvm_storage_sync("shared")
                            for stage in T.unroll(5):
                                offset = T.meta_var(16 >> stage)
                                partner = T.meta_var(lane + T.if_then_else(column < offset, offset, 0))
                                combined = partial[stage, lane] + partial[stage, partner]
                                recovered = T.fma(-1.0, partial[stage, lane], combined)
                                error = T.fma(-1.0, T.fma(-1.0, recovered, combined), partial[stage, lane]) + T.fma(-1.0, recovered, partial[stage, partner])
                                correction[stage + 1, lane] = correction[stage, lane] + (correction[stage, partner] + error)
                                partial[stage + 1, lane] = combined
                                T.tvm_storage_sync("shared")
                            if T.And(column == 0, token < v):
                                output[row, token] = partial[5, lane] + correction[5, lane]

        output = op.tensor_ir_op(
            project, "drowse_pairwise_output_projection",
            args=[op.reshape(x, [rows, hidden]), self.weight],
            out=Tensor.placeholder([rows, vocab], "float32"),
        )
        output = op.reshape(output, [*x.shape[:-1], vocab])
        return self._apply_bias(output)

    def _apply_bias(self, output):
        return output if self.bias is None else output + self.bias


class CenteredGPTNeoXQKV(PairwiseOutputLinear):
    def _apply_bias(self, output):
        if self.rotary_dim == self.head_dim:
            return super()._apply_bias(output)
        query, key, value = op.split(self.bias, 3, axis=0)
        key = op.reshape(key, [-1, self.head_dim])
        # A non-rotated key bias adds the same score to every key and cancels
        # in softmax. Omit that offset before rounding the float32 keys.
        if self.rotary_dim:
            rotated, _ = op.split(key, [self.rotary_dim], axis=1)
            key = op.concat([rotated, op.zeros([key.shape[0], self.head_dim - self.rotary_dim], key.dtype)], dim=1)
        else:
            key = op.zeros(key.shape, key.dtype)
        return output + op.concat([query, op.reshape(key, [-1]), value], dim=0)


class CompensatedLayerNorm(nn.LayerNorm):
    def forward(self, x):
        if x.dtype != "float32" or not self.elementwise_affine:
            return super().forward(x)
        reduction = PairwiseOutputLinear(x.shape[-1], 1, bias=False, dtype="float32")
        reduction.weight = op.full([1, x.shape[-1]], 1.0, "float32")
        mean = reduction(x) / x.shape[-1]
        centered = x - mean
        variance = reduction(centered * centered) / x.shape[-1]
        return centered / op.sqrt(variance + self.eps) * self.weight + self.bias


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
    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            layer.input_layernorm = CompensatedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            layer.post_attention_layernorm = CompensatedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            layer.attention.query_key_value = CenteredGPTNeoXQKV.from_linear(layer.attention.query_key_value)
            layer.attention.query_key_value.head_dim = config.head_dim
            layer.attention.query_key_value.rotary_dim = int(config.head_dim * config.rotary_pct)
            layer.attention.dense = PairwiseOutputLinear.from_linear(layer.attention.dense)
            layer.mlp.dense_h_to_4h = PairwiseOutputLinear.from_linear(layer.mlp.dense_h_to_4h)
            layer.mlp.dense_4h_to_h = PairwiseOutputLinear.from_linear(layer.mlp.dense_4h_to_h)
        self.final_layer_norm = CompensatedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        self.embed_out = PairwiseOutputLinear(config.hidden_size, "vocab_size", bias=False, dtype="float32")
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
