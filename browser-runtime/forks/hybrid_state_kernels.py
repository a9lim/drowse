"""Well-formed recurrent-state gather/scatter kernels for the candidate toolchain."""

from tvm.script import tirx as T


def state_get(shape, dtype, max_batch_size, max_history, state_id):
    @T.prim_func(s_tir=True)
    def gather(var_storage: T.handle, var_slots: T.handle, var_history: T.handle, var_output: T.handle):
        batch = T.int32()
        T.func_attr({"global_symbol": f"rnn_state_get_{state_id}"})
        storage = T.match_buffer(var_storage, (max_batch_size, max_history, *shape), dtype)
        slots = T.match_buffer(var_slots, (batch,), "int32")
        history = T.match_buffer(var_history, (batch,), "int32")
        output = T.match_buffer(var_output, (batch, *shape), dtype)
        for i in range(batch):
            for s in T.grid(*shape):
                with T.sblock("copy"):
                    vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *([s] if len(shape) == 1 else s)])
                    T.buffer_store(output, T.BufferLoad(storage, [slots[vi], history[vi], *vs]), [vi, *vs])

    return gather


def state_set(shape, dtype, max_batch_size, max_history, state_id):
    @T.prim_func(s_tir=True)
    def scatter(var_storage: T.handle, var_slots: T.handle, var_history: T.handle, var_data: T.handle):
        batch = T.int32()
        T.func_attr({"global_symbol": f"rnn_state_set_{state_id}"})
        storage = T.match_buffer(var_storage, (max_batch_size, max_history, *shape), dtype)
        slots = T.match_buffer(var_slots, (batch,), "int32")
        history = T.match_buffer(var_history, (batch,), "int32")
        data = T.match_buffer(var_data, (batch, *shape), dtype)
        for i in range(batch):
            for s in T.grid(*shape):
                with T.sblock("copy"):
                    vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *([s] if len(shape) == 1 else s)])
                    T.buffer_store(storage, T.BufferLoad(data, [vi, *vs]),
                                   [slots[vi], (history[vi] + 1) % T.cast(max_history, "int32"), *vs])

    return scatter


def install_candidate_state_kernels():
    from mlc_llm.nn.rnn_state import RNNState
    RNNState.create_get_func = staticmethod(state_get)
    RNNState.create_set_func = staticmethod(state_set)
