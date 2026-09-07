# Small base-model artifact audit

For the newer candidate compiler adapters, completion UI, tests, and remaining release work, see [Base-model integration status](BASE_MODEL_SUPPORT_QA.md). The audit below records the earlier artifact checks; it does not establish current public catalog availability.

Checked 2026-09-04. This is an upstream-artifact shortlist, **not a list of tested browser releases**. No J-lens or SAE was fitted. No runtime locks or public repositories were changed.

## Result

Gemma 3 1B PT is the closest fit to the current browser runtime. GPT-2 small and Pythia 70M deduped provide genuinely smaller, different families, but require architecture support before browser installation. Qwen 3.5 2B Base has official pretrained SAEs and an existing J-lens, but needs both hybrid-attention and Top-K SAE support. None is currently published as a verified Polythetic base-model bundle.

The live Polythetic catalog returned HTTP 200, sequence 6, with Gemma 3 IT and Qwen3 post-trained variants only. A missing “Instruct” suffix does not make Qwen3 1.7B or 4B a base model.

## Shortlist and evidence

All Neuronpedia J-lens paths below are pinned to revision `0731326edff4ae730ffc5356fe1a4728c748b3a6`. Each checked `config.yaml` names the specified base checkpoint, not its instruction-tuned sibling. These configs do not establish the exact source-model weight commit used for fitting; that provenance must be resolved before release.

| Model | Existing J-lens | Existing SAE | Browser status |
| --- | --- | --- | --- |
| **Gemma 3 1B PT** | [Neuronpedia base lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/0731326edff4ae730ffc5356fe1a4728c748b3a6/gemma-3-1b/jlens/Salesforce-wikitext), `google/gemma-3-1b-pt`; 25 matrices, 1152 × 1152 | [Google Gemma Scope 2 PT](https://huggingface.co/google/gemma-scope-2-1b-pt/tree/b738dc06961818c011fb2e44a316352ca0f4e873/resid_post/layer_13_width_16k_l0_medium), residual output layer 13, 16,384 features, JumpReLU | Supported architecture and SAE activation primitives; still needs a base-specific conversion, prompt/stop policy, runtime lock, packed instruments, and physical generation/readout validation. Official model access is gated. |
| **GPT-2 small, 124M** | [Neuronpedia GPT-2 lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/0731326edff4ae730ffc5356fe1a4728c748b3a6/gpt2-small/jlens/Salesforce-wikitext), `openai-community/gpt2`; 11 matrices, 768 × 768 | [Bloom residual SAEs](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted/tree/57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9/blocks.8.hook_resid_pre), 24,576 features | GPT-2 is not a supported production compiler target. Need LayerNorm/unembedding support and an explicit check of TransformerLens preprocessing and pre-block versus post-block capture coordinates. Matching dimensions alone does not prove compatibility. |
| **Pythia 70M deduped** | [Neuronpedia deduped lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/0731326edff4ae730ffc5356fe1a4728c748b3a6/pythia-70m-deduped/jlens/Salesforce-wikitext), `EleutherAI/pythia-70m-deduped`; 5 matrices, 512 × 512 | [EleutherAI deduped SAEs](https://huggingface.co/EleutherAI/sae-pythia-70m-deduped-32k/tree/7a64bade597212176dfe0782f9d839b94f0addaf/layers.3), 32,768 features, Top-K **16** | GPT-NeoX and Top-K are not supported production paths. Do not substitute non-deduped weights, a different training step, or ReLU. The lens ran to 1000 prompts with final relative change 0.00951 versus requested 0.001; evaluate readout quality before recommending it. |
| **Qwen 3.5 2B Base** | [Neuronpedia base lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen3.5-2b-pt/jlens/Salesforce-wikitext), `Qwen/Qwen3.5-2B-Base` | [Official Qwen SAE](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/tree/132ea3697b591df9ee46d738aa1d528e3c6082f7), residual post, width 32,768, Top-K **50**, hidden size 2048 | Not interchangeable with Qwen3. Its hybrid linear/full-attention architecture is not a production compiler target. SAE weights alone are roughly 537 MB per layer in FP32, so this is not a low-memory default. |

Additional candidates checked: [Gemma 2 2B base](https://huggingface.co/google/gemma-scope-2b-pt-res/tree/fd571b47c1c64851e9b1989792367b9babb4af63) has Google SAEs and a Neuronpedia J-lens, but needs Gemma 2 compiler support and NPZ import verification. [Gemma 3 4B base](https://huggingface.co/google/gemma-scope-2-4b-pt/tree/a0ffd6132a985bc84077a66d1a1033e10b604fa8) also has both; it is a larger desktop candidate, not a compact-device fallback. Gemma 270M remains excluded from the app in accordance with the earlier removal request. Searches did not establish matching precomputed pairs for SmolLM2 or Pythia 160M; do not claim that no such artifacts can exist elsewhere.

## Actual artifact checks

Run from the repository root:

```sh
.venv/bin/python browser-runtime/audit-base-model-artifacts.py
```

The script downloads approximately 233 MB into memory, verifies pinned SHA-256 and exact lengths, uses restricted `torch.load(weights_only=True)`, and checks every downloaded matrix for expected dimensions and finite values. It downloads no model weights, executes no repository code, trains nothing, writes no packs, and publishes nothing.

Observed results:

| Artifact | Download bytes | FP32 tensor storage | Result |
| --- | ---: | ---: | --- |
| Gemma 3 1B base J-lens, all fitted layers 0–24 | 66,363,652 | 132,710,400 | Passed |
| Gemma 3 1B base SAE, layer 13 | 151,131,000 | 151,130,624 | Passed; all five FP32 tensors, nonnegative JumpReLU thresholds |
| GPT-2 J-lens, layers 0–10 | 12,980,477 | 25,952,256 | Passed |
| Pythia 70M deduped J-lens, layers 0–4 | 2,624,492 | 5,242,880 | Passed |

GPT-2 and Pythia SAE repository/config/file metadata were checked, not their complete tensor payloads. Qwen's config and lens provenance were checked, not its complete tensors. No base model has passed a generation, quantized-activation parity, or physical-phone test in this audit.

## Memory and accuracy constraints

- Parameter count and download size are not peak memory. Include model weights, KV cache, prefill/decode scratch buffers, instrument matrices, exact vocabulary readouts, CPU/GPU copies during loading, and browser overhead.
- Gemma 1B's full J-lens plus one SAE is about **284 MB of FP32 tensors before the model and working memory**. Avoid downloading the 815 MB `examples.safetensors` file for inference. It is not required by the encoder/decoder.
- Keep the provider's actual activation function, thresholds, bias convention, capture position, layer indexing, normalization, tokenizer, and unembedding. Do not approximate Top-K with ReLU or replace J-lens with R-lens/logit lens.
- Existing `import-provider-instruments.py` samples at most eight J-lens layers. A new release must not silently describe that as the full provider lens. For small models, preserve all fitted layers when the measured memory budget allows; otherwise explicitly disclose the selected-layer coverage. No fitting is necessary to retain existing matrices.
- The runtime currently admits ReLU and JumpReLU SAE packs, not Top-K. The production compiler explicitly accepts `qwen3`, `llama`, and `gemma3_text`, not GPT-2, GPT-NeoX, Gemma 2, or Qwen 3.5.
- Existing instruction-model binaries, fingerprints, core packs, lenses, or SAE manifests must not be relabeled as base-model releases. The current importer's manifest stamping is not proof of an upstream checkpoint match.

## UI and release gate

The picker now has a native, keyboard-accessible **Base Models** disclosure, closed initially. Its warning explains text completion versus instruction following. Empty catalogs say there are no verified base downloads; candidates are not presented as usable installs.

An optional signed `modelType: "base"` field separates explicitly classified base models. Older catalogs without it retain their existing chat behavior. As of the 2026-09-05 implementation pass, base-model generation requires a validated model/core closure but not a J-lens or SAE. These tools are optional and unselected on a fresh install; selecting them still enforces compatibility and their combined instrument budget. Base models remain excluded from automatic recommendations and first-run default selection. Explicitly reopening an already selected base model remains possible and reveals its section.

Before publishing entries: resolve exact checkpoint provenance and redistribution/access prerequisites, create base-specific immutable artifacts, implement and test raw-completion prompting/stopping, validate model generation and instrument readouts on physical hardware, measure context/memory limits, and sign the catalog. Deploy the reader that understands `modelType` before publishing that field: older strict-schema clients will reject it. Existing signature verification and runtime identity checks remain enabled.

## Local application verification

The following records the earlier compulsory-tool setup tests, superseded by the optional-instrument tests in [Base-model integration status](BASE_MODEL_SUPPORT_QA.md).

- 65 runtime-foundation tests passed, including strict signed-catalog admission and the new requirement that both instruments cover every base-model context.
- 43 shell-controller tests passed, including compulsory base-model instrument downloads on mobile, missing-SAE rejection, and combined-memory-budget rejection.
- Catalog-builder and entry-selection tests passed. Invalid model types and incomplete base-model tool coverage are rejected; a hidden base model is never implicitly selected.
- Three browser regressions passed in Chromium/WebKit: keyboard disclosure/selection, required SAE labeling, the empty-catalog state, light/dark layouts at 320, 390, and 1440 pixels, and an accessibility scan of the disclosure. Screenshots were inspected. These are UI fixtures, not real-model inference tests or physical iPhone measurements.
- Svelte check reported zero errors and warnings; theme, runtime-boundary, color, interface-policy, and hosted build checks passed. The existing large-chunk build advisory remains.
