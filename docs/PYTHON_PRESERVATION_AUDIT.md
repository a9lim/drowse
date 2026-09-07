# Python preservation audit

Checked on 2026-09-06 against original Saklas 5.3.0 at
`e32e368b3b08626b47b54e39e0042c37ae27023e`, using the current `dev` working tree.
The working tree already contained the Drowse rename, browser runtime, and
Python improvements. This audit preserves those changes. Compatibility here
means the original capabilities under Drowse names, as requested; old import,
command, and URL aliases are outside the contract.

## What remains available

All 119 original Python modules have counterparts in `drowse/`. A snapshot
captured by importing the unmodified original package verifies these original
surfaces against the current implementation:

- 81 public exports, 329 callable signatures, and 65 properties.
- 53 CLI parser paths, covering all eight verbs: `serve`, `manifold`, `pack`,
  `experiment`, `config`, `template`, `lens`, and `sae`.
- 86 HTTP/WebSocket method-and-path entries, normalized for the rename.

The signature check exercises original positional and keyword call forms and
checks unchanged optional defaults. It permits additive capabilities. These
checks are in `tests/test_python_preservation.py`; the baseline contract and
artifacts written by original Saklas live in `tests/fixtures/saklas_v5_3/`.

| Capability | Current Python surface and verification |
| --- | --- |
| Generation and steering | `DrowseSession`, streaming, stateless/stateful generation, expressions, ablation, and hook cleanup; unit tests and real-model MPS tests |
| Extraction and manifolds | Original extraction, authored/discover fitting, composition, monitoring, transfer, and comparison modules retained; unit tests and real-model discover/fit/steer pipeline |
| Templates and experiments | Original template scoring and experiment CLI/API surfaces retained and covered by the non-GPU suite |
| Jacobian/R lenses | Fitting, readout, atoms, probes/gates, decomposition, and source lifecycle retained; unit tests plus the Jacobian-lens GPU suite |
| SAEs | Local training and external-source lifecycle, steering, and instrumentation retained; unit tests and installed SAE-Lens registry checks; gated real-release smoke not run |
| Artifacts | Original Profile, LoomTree, and baked manifold files load; original baked directions survive installation; GGUF tests run with the optional dependency installed |
| Server | OpenAI chat/completions, Ollama chat/generate, and native WebSocket generation exercised against an installed wheel and a real model |
| Dashboard | Default build installs `HttpRuntimeClient`; hosted WebGPU build remains separate; the packaged default dashboard runs with `navigator.gpu` disabled |

Resolving every public Python export imports neither `drowse.server`,
`drowse.web`, nor FastAPI. A regression test enforces this boundary. Python
operation does not require Node, a browser, or WebGPU. `create_app(web=False)`
and the CLI's `--no-web` surface remain available.

## Regressions repaired

1. **Original conversation loading:** migrate the original version key before
   the file loader checks required fields, matching the existing dictionary
   loader's behavior.
2. **Original manifold reuse:** recognize renamed predecessor sidecars as stale
   fits that can be regenerated. Original baked v10 artifacts, which have no
   corpus to refit, accept the exact predecessor field set with an unknown
   context-binding proof. No proof is invented; other schema/version damage is
   still rejected, and source bytes remain unchanged.
3. **Prompt pooling:** trim template suffix whitespace only when it follows a
   closing special token. Legitimate trailing assistant-content whitespace
   remains part of the pooled span.
4. **Published manifold discovery:** search original Saklas/Polythetic tags as
   well as the current Drowse tag, deduplicating within the original result cap.
5. **Python distribution assets:** package the complete default dashboard tree,
   including the theme initializer and icons; serve icon/social directories as
   files. The packaging check now detects omissions shared by both wheel and
   sdist, and missing icons return 404 instead of the SPA document.
6. **Python dashboard autosave:** accept the Python tree's explicit null session
   ID while retaining model matching and rejection of missing or conflicting
   session IDs. The snapshot regression uses the original Python tree fixture.

Also corrected two blocked browser-corpus candidate algorithm identifiers and
the existing test typing/lint failures exposed by the project checks. Runtime
test thresholds and package version were not relaxed or changed.

## Verification

Commands use the project's Python 3.12 environment on Apple Silicon macOS.

| Check | Result |
| --- | --- |
| `.venv/bin/pytest -q -m 'not gpu'` | 3,751 passed, 18 skipped, 50 deselected |
| MPS `test_smoke.py`, `test_session.py`, `test_jlens_gpu.py` | 42 passed, 1 skipped; public SmolLM2-360M-Instruct and the lens suite's small model fixtures |
| MPS generation lifecycle/concurrency suite | 5 passed; state recovery after exceptions and concurrent-generation rejection verified |
| Final Python preservation regressions | 10 passed |
| Ruff across the checkout | Passed |
| Pyright using the project interpreter | 0 errors; 1 existing dynamic script-import warning in `test_hosted_jlens.py` |
| Generated REST type check | Current |
| Default and hosted frontend checks; runtime tests | Passed; conversation snapshot/library tests rerun after the autosave fix |
| Default and hosted builds | Passed; build isolation passed for both |
| Wheel and sdist build; package isolation | Passed; all 971 default dashboard files present and matching |
| Installed wheel outside the checkout | Public imports, static asset MIME types, OpenAI/Ollama generation, OpenAI SSE, native WebSocket generation passed |
| Packaged dashboard in Chromium, WebGPU disabled | Generation, autosave, backup download, and restore passed; restored nodes and token metadata matched the saved snapshot; no page errors or failed requests |

The 18 non-GPU skips are uncached representative-tokenizer drift checks with
downloads disabled. The real-model suite's one skip is a pre-existing obsolete
monitor-based ablation assertion; the other steering/ablation tests ran.
CUDA, all model families and Python versions, gated SAE/model downloads, and a
fresh real-device hosted-WebGPU matrix were not exercised. Passing interface
and regression checks does not establish identical numerical outputs for every
model and hardware combination.

Changes remain local on `dev`; no version bump, commit, push, or deployment was
performed by this audit.
