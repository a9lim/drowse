# Drowse

[![CI](https://github.com/stdlogits/drowse/actions/workflows/ci.yml/badge.svg)](https://github.com/stdlogits/drowse/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Drowse is a local workbench for mechanistic interpretability on large language
models.

It comes with a local dashboard, along with a Python API, and a server compatible
with both OpenAI and Ollama.

## Quick start

The PyPI distribution name is `drowse.ai`; the Python import and CLI are
`drowse`.

```bash
pip install drowse.ai
drowse serve google/gemma-3-4b-it
```

Open [http://localhost:8000](http://localhost:8000).

The first launch downloads the model and fits the 17 bundled concept probes. This
can take a while; they get stored in `~/.drowse/` for future launches.

For NVIDIA CUDA:

```bash
pip install "drowse.ai[cuda,flash]"
drowse serve google/gemma-3-4b-it --device cuda
```

## Browser edition

[drowse.ai](https://drowse.ai) runs the same dashboard in the browser, with
inference, steering, probes, and the J-lens on the device through WebGPU and no
server. Its source, the in-browser runtime, and its release tooling live in
[stdlogits/drowse.ai](https://github.com/stdlogits/drowse.ai), which pins this
repository as a submodule for the shared dashboard and the Python reference
implementation. The Python server and the bundled dashboard described here are
independent of it.

## WebUI

### Threads

Every authored or generated turn becomes a node in a branching tree. You can reroll
from any point and the tree lets you save and load conversations.

The bottom of the panel has the standard sampling controls: temperature, top-p,
top-k, repetition and presence penalties, and everything else you'd expect.

### Completion modes

Chat mode renders the model's template as turns with collapsible thinking. 
Roles are handled dynamically:

- **you write** selects the role the text you write will be appended as;
- **model writes** selects the role the model continues as (or `none` to just append);
- the cast button lets you assign steering to specific roles.

Compatible chat templates support arbitrary role labels beyond `user` and `assistant`.

Raw mode exposes one raw buffer for base models.

### Token-level inspection

Tokens can be highlighted by probe scores or **surprise (logprob)**.

Clicking on a token opens one detail drawer with four views:

- **geometry** — attached probe readings across layers;
- **logits** — the chosen token and captured alternatives with logprobs, plus
  token forking;
- **SAE** — sparse feature activations;
- **J-lens** — the aggregate workspace and layer-by-vocabulary readout.

The detail cursor can walk tokens, thinking/response segments, and turns without
closing the drawer. Historical rows use their captured measurements when present
and can replay the producing prefix for newly attached instrumentation.

Clicking on the info icon by a probe shows a fitted concept's geometry layer by layer.

### Instruments

The right side of the workbench has four tabs.

| Tab | Purpose |
|---|---|
| **Subspace** | Flat fitted subspaces |
| **Manifold** | Curved fitted surfaces |
| **SAE** | SAE features |
| **Lens** | Jacobian-lens workspace |

### Analysis tools

Choose **Menu → All tools** in the workbench for further options:

- build, fit, merge, install, and inspect manifolds;
- author and score restricted-choice templates;
- manage the cast and open steering workflows;
- inspect correlations and pairwise layer geometry;
- check model, device, source, authentication, and server health;
- open the built-in help surface.

## Concepts, subspaces, and manifolds

In Drowse, you extract concepts as **manifolds** or **subspaces**.

- A 1D flat subspace is just a steering vector.
- A higher-rank flat subspace is a group of orthogonal steering vectors.
- A curved manifold fits a nonlinear surface and lets you steer along it.

### Bundled concepts

Drowse comes with 17 concept pairs that are attached as probes by default:

| Category | Concepts |
|---|---|
| **Epistemic** | `confident.uncertain`, `honest.deceptive`, `curious.disinterested` |
| **Alignment** | `refusing.compliant`, `sycophantic.blunt`, `sincere.manipulative` |
| **Register** | `formal.casual`, `direct.indirect`, `verbose.concise`, `creative.conventional`, `humorous.serious`, `warm.clinical`, `technical.accessible` |
| **Cultural** | `masculine.feminine`, `individualist.collectivist`, `traditional.progressive`, `religious.secular` |

Three larger concept sets ship as well, but aren't fitted by default:

- **`personas`** — 107 personas;
- **`emotions`** — 20 emotional states;
- **`months`** — 12 months.

## Steering expressions

Every surface uses the same expression format. The same recipe can be used
across Python, YAML, OpenAI, Ollama, or the native API.

```text
0.3 honest + 0.4 warm
0.5 formal - 0.2 verbose
0.3 honest.deceptive|sycophantic.blunt
0.3 honest.deceptive~confident.uncertain
!sycophantic.blunt
0.5 personas%pirate
0.7,0.4 months%january@after
0.4 warm@when:confident.uncertain>0.4
0.3 jlens/orange + 0.2 sae/9143
```

| Syntax | Meaning |
|---|---|
| `+`, `-` | Add or subtract terms |
| leading number | Steering coefficient; omitted terms default to `0.5` |
| `~` | Keep the component shared with another direction (Mahalanobis) |
| `\|` | Remove the component shared with another direction (Mahalanobis) |
| `!` | Mean-ablate a direction; a bare `!x` ablates fully |
| `%label` or `%x,y,…` | Choose a named node or coordinates on a manifold |
| `@prompt`, `@response`, `@after`, `@thinking`, … | Restrict the token phase where a term applies |
| `@first:N`, `@after:N` | Restrict a term to a counted decode window |
| `@when:<probe><op><value>` | Apply a term only while a live probe gate is true |
| `jlens/<word>`, `jlens/#<id>`, `sae/<id>` | Steer along a J-lens token direction or an SAE decoder row |

A bare pole or node label such as `honest` or `pirate` resolves to its concept
or manifold; an ambiguous name raises instead of guessing. The operands of `~`,
`|` and `!` must reduce to one direction: a whole two-pole concept
(`sycophantic.blunt`, not `sycophantic`), an extracted vector, or a
`jlens/`/`sae/` atom. `@response` covers every generated token, thinking
included; `@after` applies only after the thinking block closes. A gate names
an attached probe, so `confident.uncertain` above reads the bundled probe. The
larger sets (`personas`, `emotions`, `months`) steer once they are fitted for
the model.

Manifold coefficients use two coordinates: `along` and `onto`. `along` controls
movement within the manifold toward the target; `onto` reduces the off-surface
component inside the manifold's fitted tube. For a flat subspace, aim for a
coefficient near `0.5` and raise it toward `1.0` for stronger expression.

## How Drowse works

### Extraction

Drowse first has the model answer a shared set of baseline prompts as each concept,
then it takes the resulting hidden states and fits them to either a curved manifold
or a flat subspace.

Layer allocation uses a Mahalanobis metric estimated from neutral activations.
Discriminative layer selection removes flat axes that fail to straddle the neutral
baseline across the fitted nodes.

The full data flow, artifact boundaries, instrument protocol, and concurrency
invariants are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

### Monitoring

A reading includes fitted coordinates, the centered activation's subspace
fraction, the nearest nodes and their soft assignment, and, for curved manifolds,
an off-surface residual and tube membership.

### Jacobian lens

The Jacobian lens implementation follows Gurnee et al.'s
[work](https://transformer-circuits.pub/2026/workspace/index.html).
Drowse lets you use one of the published J-lens artifacts, or fit your own.

### Sparse autoencoders

Drowse can either use a published SAELens release or train a local SAE.

## Installation

Drowse requires Python 3.11 or newer and PyTorch 2.13 or newer. CUDA or Apple
Silicon MPS is strongly recommended for interactive use; CPU is supported for
smaller models and non-GPU workflows.

```bash
pip install drowse.ai
```

The base package includes the HTTP server and the prebuilt Svelte WebUI.
Optional extras add specialized workflows:

| Extra | Adds |
|---|---|
| `flash` | FlashAttention 2 for supported NVIDIA CUDA models; stable and tested |
| `cuda` | `bitsandbytes` quantization and Hugging Face `kernels` acceleration |
| `sae` | SAELens 6.x, for published SAE releases (`sae fetch`); local SAEs need no extra |
| `hf` | `datasets` for streamed J-LENS and SAE corpora |
| `gguf` | GGUF import/export support |
| `research` | `datasets`, NumPy, SciPy, scikit-learn, pandas, Matplotlib, and image helpers |
| `notebook` | Plotly, pandas, and Kaleido notebook helpers |
| `dev` | Test, lint, type-check, and build tooling |

Extras can be combined:

```bash
pip install "drowse.ai[cuda,flash]"     # full tested NVIDIA path
pip install "drowse.ai[sae,hf]"         # SAELens sources and streamed corpora
pip install "drowse.ai[hf,research]"    # dataset-backed research workflows
pip install "drowse.ai[notebook]"       # interactive figures
```

`cuda` and `flash` are Linux/NVIDIA CUDA extras. FlashAttention is selected
automatically when installed; there is no runtime flag to enable it.

From source:

```bash
uv pip install -e ".[dev]"
```

## Running the server

```bash
drowse serve MODEL [options]
```

Repository Python is disabled by default. For a trusted model that requires custom
code, use `DROWSE_TRUST_REMOTE_CODE=1 drowse serve MODEL`; Python callers can
pass `trust_remote_code=True` to `DrowseSession.from_pretrained`. This grants
the model repository permission to execute Python locally.

Common options:

| Option | Default | Purpose |
|---|---:|---|
| `-d`, `--device` | `auto` | `cuda`, `mps`, `cpu`, or automatic selection |
| `-q`, `--quantize` | none | `4bit` or `8bit` bitsandbytes quantization on CUDA |
| `-p`, `--probes` | `all` | Bundled probe categories, `all`, or `none` |
| `-H`, `--host` | `127.0.0.1` | Bind address; non-loopback requires an API key |
| `-P`, `--port` | `8000` | Bind port |
| `-S`, `--steer` | none | Default steering expression |
| `-C`, `--cors` | none | Allowed CORS origin; repeatable |
| `--top-k-alts` | `0` | Alternative tokens captured at each decode step |
| `--compile` | off | Opt into `torch.compile` after Drowse probes the path |
| `--cuda-graphs` | off | Pair static cache and CUDA graph capture with `--compile` |
| `-k`, `--api-key` | none | Require bearer authentication; also reads `$DROWSE_API_KEY` |
| `--no-web` | off | Run the APIs without mounting the dashboard |
| `-c`, `--config` | none | Setup YAML; repeatable, later files win |

`serve` and every subcommand that accepts `-c/--config` read
`~/.drowse/config.yaml` first and then compose any explicit `-c PATH` files on top.
For example:

```yaml
model: google/gemma-3-4b-it
vectors: "0.3 honest + 0.2 warm"
temperature: 0.8
top_p: 0.9
max_tokens: 512
return_top_k: 8
```

Command-line flags override YAML, and `vectors:` sets the server's default
steering only; the `experiment` verbs use exactly their own `-S`. Inspect the
resolved configuration with `drowse config show` and validate a file with
`drowse config validate path.yaml`.

## Command-line artifact workflows

The CLI has eight top-level verbs. The WebUI covers the interactive versions of
most workflows; the CLI is useful for reproducible preparation, distribution, and
batch work.

| Verb | Role |
|---|---|
| `serve` | Launch the WebUI and the three HTTP protocol surfaces |
| `manifold` | Extract, generate, derive from a template, fit, bake, merge, transfer, compare, or diagnose manifolds |
| `pack` | List, inspect, install, search, push, clear, refresh, remove, or export manifold packs |
| `experiment` | Run alpha fans, replay transcripts, and evaluate naturalness |
| `config` | Show or validate composed configuration |
| `template` | Create and score restricted-choice completion templates |
| `lens` | Fit, fetch, select, read, decompose, or remove Jacobian lenses |
| `sae` | Train, fetch, select, inspect, or remove SAE sources |

Representative commands:

```bash
# Extract and fit a two-pole concept for one model
drowse manifold extract patient impatient -m google/gemma-3-4b-it

# Fit a bundled many-node manifold
drowse manifold fit personas -m google/gemma-3-4b-it

# Install or publish manifold packs through Hugging Face
drowse pack search creativity
drowse pack install OWNER/REPO
drowse pack push local/patient.impatient -a OWNER/REPO -m google/gemma-3-4b-it

# Move one manifold closure, plus its referenced template, as an archive
drowse pack export archive local/patient.impatient -o patient.drowse
drowse pack install patient.drowse

# Fetch a provider J-LENS (Neuronpedia by default), or fit a local one:
# --kind r (the default) is the RelP R-lens at local:r, --kind j the standard
# J-lens at local:j
drowse lens fetch google/gemma-3-4b-it
drowse lens fit org/model --prompts 100
drowse lens fit org/model --kind j --layers workspace
# Or on another commit-pinned Hub dataset (a shuffled one grows by nested samples)
drowse lens fit org/model --hf-corpus HuggingFaceFW/dclm_100BT-shuffled --prompts 1000
# A bilingual mix: 3 English documents per Chinese one, the Chinese side spread
# over 64 of its (unshuffled) data files
drowse lens fit org/model --prompts 1000 \
  --hf-corpus HuggingFaceFW/dclm_100BT-shuffled,weight=3 \
  --hf-corpus HuggingFaceFW/fineweb-2:cmn_Hani,files=64
# Pick which lens the session reads
drowse lens use org/model local:r

# Fetch an SAE (needs the [sae] extra) or train a local source
drowse sae fetch google/gemma-3-4b-it saelens:gemma-scope-2-4b-it-res
drowse sae train org/model my-sae --layer 20 --tokens 1000000
```

Lens fits are resumable: an interrupted fit checkpoints and continues, and an
exact repeat is recognized without loading the model. `--layers workspace`
(the 40–90% depth band) is the main lever on fit time. Run `drowse <verb> -h`
and `drowse <verb> <subcommand> -h` for the complete flag surface.

## HTTP APIs

The same `drowse serve` process exposes four surfaces on one port:

- `/` — the Drowse WebUI;
- `/v1/*` — OpenAI-compatible models and chat completions;
- `/api/*` — Ollama-compatible generation and chat;
- `/drowse/v1/*` — native sessions, loom trees, probes, manifolds, templates,
  SAE/J-LENS lifecycle and replay, SSE, and token-plus-measurement WebSockets.

Interactive OpenAPI documentation is available at
[http://localhost:8000/docs](http://localhost:8000/docs).

### OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Describe a rainy afternoon."}],
    extra_body={"steering": "0.3 warm + 0.2 concise"},
)
print(response.choices[0].message.content)
```

### Ollama client

```bash
curl -N http://localhost:8000/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Write a short haiku."}],
  "options": {"steer": "0.3 warm - 0.2 formal.casual"}
}'
```

Drowse targets a trusted local machine or lab network. It is not a hardened
multi-tenant inference service. If you bind it beyond a trusted host, read
[SECURITY.md](SECURITY.md), set an API key, and add TLS, rate limits, request
limits, and isolation outside Drowse.

## Python API

```python
from drowse import DrowseSession, SamplingConfig

with DrowseSession.from_pretrained(
    "google/gemma-3-4b-it",
    device="auto",
    return_top_k=8,
) as session:
    name, profile = session.extract("patient", baseline="impatient")
    session.add_probe(name)

    result = session.generate(
        "How should I learn a difficult skill?",
        steering=f"0.3 {name} + 0.2 concise",
        sampling=SamplingConfig(
            temperature=0.8,
            top_p=0.9,
            max_tokens=256,
            seed=42,
        ),
    ).first

    print(result.text)
    print(result.applied_steering)
    print(result.probe_readings[name].coords)
```

`generate` and `generate_stream` accept the same steering expression as the WebUI;
`session.steering(expr)` applies one to every generation inside a `with` block.
Generation returns a list-like `RunSet`; `.first` is convenient for a single
completion. `GenerationResult` carries text, token IDs, throughput and timing,
finish reason, the canonical applied expression, captured log probabilities,
aggregate probe readings, and the versioned measurement envelope every
instrument family writes into.

Batch and sweep helpers return the same result shape:

```python
batch = session.generate_batch(
    ["Describe a sunset.", "Describe a storm."],
    steering="0.3 warm",
)

sweep = session.generate_sweep(
    "Describe a forest.",
    sweep={"warm.clinical": [-0.4, 0.0, 0.4]},
)
```

Restricted-choice scoring reads the model's raw probability distribution over a
set of candidate completions, optionally under steering, so you can see whether
steering moved the distribution and not just the top choice:

```python
scores = session.score_choices(
    [{"role": "user", "content": "The first weekday is"}],
    ["Monday", "Tuesday", "Wednesday"],
    steering="0.3 confident",
)
```

The three read families are available as `session.geometry`, `session.lens`
and `session.sae`; `session.add_probe` attaches a concept, a `jlens/<word>`
token probe, or an `sae/<id>` feature to the right one.

Notebook helpers are available from `drowse.notebook` after installing
`drowse.ai[notebook]`: `plot_alpha_sweep`, `plot_probe_correlation`,
`plot_layer_norms`, `plot_trait_history`, and `to_dataframe`.

## Model support

Drowse has end-to-end tested paths for:

- Qwen 2, Qwen 3, and Qwen 3.5, including its text and MoE variants;
- Gemma 2, Gemma 3, and Gemma 4, including text-only extraction from supported
  multimodal checkpoints;
- Mistral 3 and Ministral 3;
- Llama, GLM, gpt-oss, and Talkie.

Other decoder-only models load when their transformer blocks sit at a standard
Transformers location, which covers most families (Mixtral, Phi, Cohere,
DeepSeek, OLMo, Granite, GPT-2, Falcon, MPT, OPT, and others). Drowse warns once
when an architecture has not been exercised end to end.

CUDA and Apple Silicon MPS both have real-model smoke coverage. Model-specific
features still depend on the checkpoint: role-augmented extraction and persona
regeneration need a chat template with a substitutable assistant label (Mistral's
does not have one), official SAE and J-LENS sources cover only some models, and
FlashAttention depends on the model's Transformers attention implementation.

## State and distribution

Drowse keeps local state under `~/.drowse/`; set `$DROWSE_HOME` to move it. The
store contains authored manifolds, per-model fits and neutral statistics, local
SAE/J-LENS artifacts, source bindings, and templates. Conversation saves are
explicit browser-downloaded JSON files (or caller-selected `LoomTree.save()`
paths); they are not autosaved under `~/.drowse/`.

Manifold packs are folders with metadata, node corpora, integrity hashes, and
optional fitted tensors. They can be installed from a local path or distributed as
Hugging Face model repositories. A `.drowse` is the narrow ZIP transport for
exactly one manifold closure and its referenced template, when present; export it
with `drowse pack export archive` and install it with `drowse pack install`.
It never contains model weights, J-lenses, or SAEs. A fitted two-node PCA manifold
can instead be exported as a llama.cpp control-vector GGUF with
`drowse pack export gguf`.

A Hugging Face pack is a model repository tagged `drowse-manifold` that holds
exactly one `.drowse` at its root, a root `manifold.json` summary, and a model
card; `drowse pack push` writes that layout, and both drowse and the drowse.ai
browser edition install from it. An install pins the repository to an immutable
commit and records that commit as the manifold's source; the browser also checks
that the archive's declared repository and revision match it.

Treat model repositories and downloaded artifacts as executable or otherwise
untrusted input. Drowse validates archive paths, declared structure, tensor
headers, and hashes, but integrity is not publisher identity.

Drowse reads only its current artifact formats and does not migrate older ones.
Releases through 5.3.0 were published as `saklas`, and drowse does not read a
`~/.saklas` directory. J-LENS artifacts fitted or fetched before 5.4 must be
converted once from a source checkout with
`python scripts/migrate_lens_v7.py --home ~/.drowse` (try `--dry-run` first).

## Development

```bash
pip install -e ".[dev]"

ruff check .
pyright
pytest -q -m "not gpu"              # the CI suite
pytest -q -n auto -m "not perf"     # everything this machine can run
pytest -m perf -p no:xdist          # throughput gates, serially
python -m build
```

GPU integration tests download the public SmolLM2-360M-Instruct weights by
default. Set `DROWSE_TEST_MODEL` to exercise another compatible model:

```bash
pytest -q tests/gpu/test_smoke.py
DROWSE_TEST_MODEL=google/gemma-3-4b-it pytest -q tests/gpu/test_smoke.py
```

The WebUI is a Svelte 5 + Vite application in `webui/`. Its compiled bundle under
`drowse/web/dist/` is committed package data and ships in the wheel.

```bash
cd webui
npm ci
npm run check
npm run build
git diff --exit-code ../drowse/web/dist
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and adding a
model architecture.

## Research lineage and credits

Drowse builds on Representation Engineering
([Zou et al., 2023](https://arxiv.org/abs/2310.01405)).
[repeng](https://github.com/vgel/repeng) by Theia Vogel is the best-known compact
implementation of that approach; Drowse takes the workbench route, adding live
monitoring, manifold geometry, branching experiments, and server protocols.

Two-pole extraction uses difference-of-means following
[Im & Li, 2025](https://arxiv.org/abs/2502.02716). Manifold steering follows
[Goodfire's manifold work](https://arxiv.org/abs/2605.05115). The `personas`
source is derived from the framing in Anthropic's
[Assistant Axis paper](https://arxiv.org/abs/2601.10387). J-LENS support implements
the verbalizable-workspace method of
[Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html).

If you use Drowse in published research, please cite the relevant upstream methods
alongside the Drowse version and exact model checkpoint you used.

## Contact, issues, and security

For questions, feedback, or research inquiries, email
[contact@drowse.ai](mailto:contact@drowse.ai) or use the
[contact form](https://drowse.ai/contact).

Please update to the latest Drowse release before filing a bug. Include the model
ID, device, dtype or quantization mode, Drowse version, and a minimal reproduction
in [GitHub Issues](https://github.com/stdlogits/drowse/issues).

Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## License

Drowse is licensed under AGPL-3.0-or-later. See [LICENSE](LICENSE).
