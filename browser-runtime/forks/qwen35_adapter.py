"""Candidate hybrid-state hook exports; requires a hybrid-aware capture consumer."""

import inspect
from types import MethodType

from tvm.relax.frontend import nn
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.model.qwen3.qwen3_model import Qwen3LMHeadModel, Qwen3Model
from mlc_llm.model.qwen35.qwen35_model import Qwen35LMHeadModel


class _HybridLayer:
    def __init__(self, layer, view):
        self.layer = layer
        self.view = view

    def __call__(self, hidden, cache, _layer_id):
        hidden, self.view.state = self.layer.forward(hidden, cache, self.view.state)
        return hidden


class _HybridModelView:
    def __init__(self, model, state):
        self.original = model
        self.state = state
        self.norm = model.norm
        self.layers = [_HybridLayer(layer, self) for layer in model.layers]

    def _drowse_prepare_inputs(self, inputs):
        return inputs

    def __getattr__(self, name):
        method = getattr(Qwen3Model, name, None)
        if callable(method):
            return MethodType(method, self)
        return getattr(self.original, name)


class _HybridHeadView:
    def __init__(self, head, state):
        self.head = head
        self.model = _HybridModelView(head.model, state)

    def __getattr__(self, name):
        method = getattr(Qwen3LMHeadModel, name, None)
        if callable(method) and name.startswith(("drowse_", "_drowse_", "_get_logits")):
            return MethodType(method, self)
        return getattr(self.head, name)


class _SpecView:
    def __init__(self, head):
        self.head = head

    def __getattr__(self, name):
        method = getattr(Qwen3LMHeadModel, name, None)
        if callable(method):
            return MethodType(method, self.head)
        return getattr(self.head, name)


def _hybrid_forward(method):
    arguments = list(inspect.signature(method).parameters)[1:]
    state_index = arguments.index("paged_kv_cache") + 1

    def forward(self, *args):
        view = _HybridHeadView(self, args[state_index])
        result = method(view, *args[:state_index], *args[state_index + 1:])
        return (*result[:2], view.model.state, *result[2:])

    return forward


class InstrumentedQwen35LMHeadModel(Qwen35LMHeadModel):
    def __init__(self, config):
        from hybrid_state_kernels import install_candidate_state_kernels
        install_candidate_state_kernels()
        super().__init__(config)

    def get_default_spec(self):
        native = super().get_default_spec()
        methods = dict(zip(native.method_names, native.method_specs))
        template = Qwen3LMHeadModel.get_default_spec(_SpecView(self))
        for name, spec in zip(template.method_names, template.method_specs):
            if not name.startswith("drowse_"):
                continue
            names, specs = list(spec.arg_names), list(spec.arg_specs)
            if "paged_kv_cache" in names:
                index = names.index("paged_kv_cache") + 1
                names.insert(index, "rnn_state")
                specs.insert(index, nn.spec.Object(object_type=RNNState))
            methods[name] = nn.spec.MethodSpec(
                getattr(self, name), names, specs, spec.param_mode, spec.effect_mode,
            )
        return nn.spec.ModuleSpec.from_raw(methods, self)


for _name, _method in vars(Qwen3LMHeadModel).items():
    if not callable(_method) or not (
        _name.startswith(("drowse_", "_drowse_")) or _name == "_get_logits"
    ):
        continue
    setattr(InstrumentedQwen35LMHeadModel, _name,
            _hybrid_forward(_method) if "paged_kv_cache" in inspect.signature(_method).parameters else _method)
