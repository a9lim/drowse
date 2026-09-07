# Hosted release inputs

`core-manifolds/welcoming.detached` is the model-independent seed manifold for
every required hosted core pack. Pass that directory as `--manifold-source` when
building SmolLM2-360M, Gemma 3 270M, or Qwen3-1.7B. The physical builder captures
the same original response pairs through each model and writes a separately
fingerprinted, steering-ready `.drowse`.

The checked-in source intentionally contains no fitted tensors. `provenance.json`
binds its corpus to the canonical baseline prompts and records authorship and
license; the release-tool validator rejects unexpected files, changed prompt
bindings, malformed corpora, and any pre-fitted payload.
