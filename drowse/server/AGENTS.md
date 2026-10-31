# drowse/server

One `DrowseSession` served over OpenAI `/v1/*`, Ollama `/api/*`, and the native
`/drowse/v1/*` tree (REST, SSE, one WebSocket), plus the dashboard at `/` when
`web=True` (`drowse/web/AGENTS.md`). Handlers validate, lock, call public
session and collaborator APIs, and serialize; engine semantics are documented
where they live. This file covers wire contracts, locking, errors, and the
security boundary.

## Module map

| Module | Owns |
|---|---|
| `app.py` | `create_app(session, default_steering, cors_origins, api_key, *, web, max_request_bytes)`, the only export of `server/__init__.py` |
| `runtime.py` | `session_lock(app)` (503 on timeout), `refuse_if_busy` (409), `run_in_thread` / `finish_worker` / `stream_events`, `ClosingStreamingResponse`, `NativeRequest`, `session_router()` |
| `errors.py` | `describe_error` (the one classifier), `protocol_error` (the three envelopes), `validation_message` |
| `security.py` | `HttpSecurityMiddleware`, `ws_auth_ok` |
| `operations.py` | `run_operation` (SSE-or-JSON jobs), `OperationLedger` (receipts), `Preparation` / `PreparationGroup` (polled jobs) |
| `request_helpers.py` | `build_sampling_config` (the one `SamplingConfig` construction site for all protocols), `compose_request_steering`, `flatten_content`, `validate_logit_bias` |
| `compat/` | `__init__`: `CompatGeneration` + the shared driver; `openai.py`, `ollama.py`: request models and formatters; `model_names.py`: Ollama aliases (`_HF_TO_OLLAMA_ALIASES` wins, else `<family>:<size>`) |
| `*_routes.py` + `*_models.py` | Native REST: sessions, profiles, probes, tree, manifolds, templates, instruments |
| `ws_stream.py`, `ws_models.py`, `ws_events.py` | `WS /sessions/{id}/stream`, its frame schemas, `GET /sessions/{id}/requests/{id}` |
| `response_models.py` | Response `TypedDict`s for every non-streaming native route |

`create_app` registers OpenAI, Ollama, `register_operation_routes` (it
installs `app.state.operations`, which later registrars read), the native
routers, the WebSocket, and the dashboard last so its catch-all cannot shadow
an API path.

## Rules for editing

- `server/` is in the top layer with `cli`, `web`, `notebook`. It never reads
  a session-private attribute (`tests/invariants/test_layering.py`), and no
  server module imports a sibling's `_private` helper
  (`tests/invariants/test_boundary_guard.py`); shared helpers go in
  `runtime.py`, `errors.py`, `operations.py`, or `request_helpers.py`.
- Native request bodies subclass `NativeRequest` (`extra="forbid"`), nested
  objects included. Ollama bodies are `extra="allow"` (clients send fields
  drowse ignores; an explicit `null` reads as unset).
- A non-streaming native route annotates its return type with a
  `response_models.py` `TypedDict` named like the dashboard's TS interface;
  `scripts/generate_webui_types.py` renders `webui/src/lib/types.gen.ts` from
  the OpenAPI schema, and CI plus `tests/invariants/test_webui_types_parity.py`
  run `--check`. `_Open` (`extra="allow"`) marks payloads produced by
  engine/io serializers, which a closed model would silently truncate.
  Request bodies, SSE frames, and WS frames are hand-written in
  `webui/src/lib/types.ts`.
- Only `PATCH /sessions/{id}` mutates `session.config`; per-request sampling
  and steering ride the call.
- Client-input failures raise the `RequestError` family (or another typed
  `DrowseError`) or an `HTTPException`; a bare `ValueError` is an engine
  invariant and reaches the client as a scrubbed 500.

## Locks and workers

| Guard | On contention |
|---|---|
| Session lock: `runtime.session_lock(app)`, one FIFO `asyncio.Lock` on `app.state` for all three protocols | waits `SESSION_LOCK_TIMEOUT_SECONDS = 300`, then `SessionLockTimeout` (503) |
| Exclusive-GPU section `session.gen_lock`, taken non-blocking by the engine and every model op | `ConcurrentGenerationError` / `ConcurrentExtractionError` (409) |
| `refuse_if_busy(session)`, a non-blocking probe of `gen_lock` | `EngineBusyError` (409) |
| Loom reservation (`LoomTree.reserve`) | `MutationDuringGenerationError` (409) |
| `PreparationGroup`: one lens or SAE preparation at a time | 409 |

- Anything that runs a forward pass or touches hook state holds the session
  lock: compat generations, a WS turn (across its whole `n`-way fan),
  `steering/validate`, `PUT /tree`, `tree/reset`, `tree/transcript/load`,
  `tree/joint-logprobs`, instrument `live` / `source` / `token-readout` / SAE
  load and unload, every `run_operation` job, and manifold merge, delete, and
  forced from-template.
- Lens fit and SAE train run outside the session lock but hold the
  exclusive-GPU section, so a generation arriving mid-fit gets an immediate
  409 instead of queueing behind a long job. That is why a route that replaces
  or removes a manifold folder calls `refuse_if_busy` even under the session
  lock, then `session.evict_artifact(ns, name)` after writing. `session.fit`
  and `session.generate_manifold` move or evict their consumers themselves.
- Blocking work goes through `run_in_thread` (or `finish_worker` around a
  task), never a bare `asyncio.to_thread`: a cancelled request stays parked,
  shielded from AnyIO and repeated asyncio cancellation, until its worker
  exits, so the lock is never released while a worker still writes.
  `ClosingStreamingResponse` closes its body iterator even when a send fails,
  and the compat driver is wrapped in `contextlib.aclosing`, so a dropped
  stream joins the engine worker before the lock releases.

## Errors

`errors.describe_error(exc, *, failure=...) -> ErrorInfo(status, message,
code)` is the only classifier: the FastAPI handlers, `run_operation`, polled
preparations, and WS error frames all use it. `code` is always the exception
class name, so clients key on typed failures (`PoisednessError`,
`RecipeOverrideError`) with no per-route classifier. It logs what it scrubs;
callers never log the exception again.

| Exception | Status | Client text |
|---|---|---|
| `DrowseError` | `user_message()` status; a class still at the base 500 falls back on its stdlib parent (`KeyError` 404, `ValueError` 400) | `user_message()` text |
| `HTTPException` raised `from` a `DrowseError` or `OSError` | the `HTTPException`'s | the cause's safe text |
| other `HTTPException` | its own | `detail` flattened to one string |
| pydantic `ValidationError` / `RequestValidationError` | 400, not FastAPI's 422 | every error as `field.path: msg`, joined by `; ` |
| `OSError` | `FileNotFoundError` 404, `FileExistsError` 409, else 500 | fixed text (OS messages carry paths) |
| `ImportError` naming a third-party module | 503 | `optional dependency not installed: <module>` |
| anything else, incl. a bare `ValueError` or a failed `drowse.*` import | 500 | `<failure>. Check the server log for details.` |

- To change a typed error's status but keep its text, `raise
  HTTPException(status) from exc`. Never put `str(e)` of an untyped exception
  in a detail; detail is sent verbatim.
- Envelopes by path prefix: `/api/` → `{"error": msg}`; `/v1/` →
  `{"error": {message, type, param, code}}` (`code` is the HTTP status; `type`
  is `conflict` for 409, `invalid_request_error` for other 4xx, else
  `server_error`; `param` is the first invalid field); anything else →
  `{"detail": msg}`. A catch-all handler renders the same scrubbed 500.
- After headers are sent, errors go in band: OpenAI SSE `data: {"error":
  {...}}` then `data: [DONE]` (clients wait for the sentinel); Ollama NDJSON
  `{..., "error"}` then a `done_reason: "error"` frame; native SSE
  `event: error {message, code, status}`; WS an `error` frame.

## Security posture

`drowse serve` is a trusted-local/lab server; TLS, rate limits, quotas, and
multi-tenant isolation are deployment concerns.

- `HttpSecurityMiddleware` is pure ASGI (streaming cancellation and worker
  ownership pass through) and authenticates before body parsing, only on the
  private prefixes `/v1/`, `/api/`, `/drowse/v1/`. With a key (`api_key` or
  `DROWSE_API_KEY`) it requires `Authorization: Bearer`, compared with
  `hmac.compare_digest` (401). Without one, the TCP peer must be loopback
  (IPv4-mapped included) and the single `Host` loopback or the ASGI server
  address, a DNS-rebinding defense (403).
- Public by design: the dashboard shell and assets, `HEAD /`, and FastAPI's
  `/docs`, `/redoc`, `/openapi.json`. None carries session data; the dashboard
  keeps the key client-side and sends it on each API call.
- Origin: on private paths (except `OPTIONS`) and on any other
  non-`GET/HEAD/OPTIONS` request, a present `Origin` must be one bare origin
  equal to the request's own or listed in `cors_origins` (403). A wildcard
  CORS entry grants nothing; a request with no `Origin` (non-browser client)
  passes.
- `Content-Length` must be one decimal value (400); declared or streamed
  bodies over `max_request_bytes` (64 MiB, `DROWSE_MAX_REQUEST_BYTES`) are 413.
- Every response gets `nosniff`, `X-Frame-Options: DENY`, `no-referrer`, a
  closed `Permissions-Policy`, and the dashboard CSP (`connect-src`: self,
  Neuronpedia, `https://drowse.ai/api/contact`); private paths add
  `Cache-Control: no-store`.
- `ws_auth_ok` runs before `accept()` (failure closes 1008). It first scrubs
  any `token` query parameter from the scope so the access log never records
  one, applies the same host and origin rules, and takes the key only as a
  bearer header or exactly one `drowse.auth.<base64url-unpadded-key>`
  subprotocol (browsers cannot set headers on a WebSocket). Only `drowse.v1`
  is echoed.
- The CLI binds `127.0.0.1` by default and refuses a non-loopback bind without
  a key; `drowse/cli/verbs/serve.py` also owns the startup live-lens/SAE policy.

## OpenAI and Ollama (`compat/`)

Both lower a request to a `CompatGeneration` and run it through one driver:
`run_generation` or `stream_generation` (`StreamStarted`, `TokenEvent`s, then
`StreamEnd` or a final `ErrorInfo`). Compat generations are always
`stateless=True`: the request's messages are the whole context and the loom is
untouched. Streams pass `live_scores=False, live_readouts=False`; probe data
comes only as the end-of-generation aggregate. A client disconnect stops the
engine at the next token.

- Steering (`compose_request_steering`): `None` inherits the server default
  (`create_app(default_steering=)`, CLI `-S`); an expression overrides it per
  key; an empty string clears it and reaches the session as an empty
  `Steering`, not `None`, because the cast roster refills an unset `None`.
- Attached probes' aggregates (Monitor, lens, and SAE rosters unioned) ride
  `x-drowse-probe-readings`, on each OpenAI choice and at top level on the
  Ollama final frame; the key is absent when nothing is attached.
- `DROWSE_STRICT_MODEL` (`1/true/yes/on`) 404s a `model` outside the HF id and
  its aliases; unset accepts any name.
- OpenAI: `/v1/models[/{id}]`, `/v1/chat/completions`, `/v1/completions` (raw
  prompt). Extensions: `steering` (expression string), `thinking` (`None` =
  auto). Accepted and ignored: `user`, `n`, `response_format` `text`, empty
  `tools`, `tool_choice` `none|auto`; other values of those and non-text
  content parts are `UnsupportedContentError` (400). Streams send a role delta
  first (chat), thinking as `reasoning_content`, a final chunk with
  `finish_reason` and readings, an optional usage chunk, and always `[DONE]`.
- Ollama: version, tags, ps, show, chat, generate, pull (no-op success for a
  hosted name, else 404), `HEAD /` liveness, and 501 for push, create, copy,
  delete, embeddings, embed. `/api/show` returns the real HF Jinja template;
  Go-template parsers fail either way. Streaming NDJSON is the default (`null`
  does not stream). `repeat_penalty` p maps to `presence_penalty = ln p` when
  p > 1 and none is set: Ollama divides positive logits by p, which subtracts
  a per-seen-token constant. Unknown options are logged at debug and dropped.
  `steer` (in `options` or top level) must be a string, else
  `OllamaBadRequest` before headers flush. `think` takes a bool or
  `low|medium|high`; there is no effort knob, so any level means on.
  `/api/generate` with `system` renders through the chat template and never
  returns `context`.

## Native REST

Paths carry `{session_id}`, but only `"default"` resolves (`session_router()`
404s others); `S` below is `/drowse/v1/sessions/default`. Manifolds and
templates are top-level artifacts under `/drowse/v1/{manifolds,templates}`.

- **Sessions.** `session_info` (`GET /drowse/v1/sessions`, `GET S`) carries
  `instruments`, one `{family, live, source, probes, capabilities}` block per
  family: the read plane's only listing. `jlens_fitted` uses
  `session.lens.has_compatible()`, which never pages a lens in. `PATCH S`:
  explicit `null` clears `top_k` or `system_prompt`; `null` elsewhere
  preserves. `POST S/steering/validate` enters and exits
  `session.steering(parsed)` under the lock, so a green result composes
  exactly as a generation would; a rejection is in band (`{valid: false,
  error}`), not a failed request.
- **Profiles.** `pairwise` and `correlation` read `session.analytics_profile`,
  CPU snapshots built inside the exclusive-GPU section, so a polled read never
  copies off the device during a model op. Both are Mahalanobis-only and 409
  without a whitener. A snapshot not built yet, or a whitener missing a layer,
  is 409 on `pairwise` (retryable) and `null` in `correlation`. `pairwise`
  registers before `profiles/{name}` so the literal path wins.
- **Probes.** One collection over the Monitor and the lens/SAE readout
  registries. Each row has a `family` and only fields that family produces;
  clients key on `family`. `GET S/probes/{name}/geometry` is in the whitened
  read frame; readout probes have no subspace and 404.
- **Tree.** Conflicts come from the tree's own reservation; navigate, star,
  note, branch, and cast edits never conflict. `PUT S/tree` restores
  atomically and refuses another model's tree. `cast_json` gives each label an
  `origin` (`configured`; `structural` for `user`/`assistant`; `observed`) and
  rides the full-tree GET and every `tree_mutated` frame. `joint-logprobs`
  replays both branches under the lock; the route bounds
  `session.joint_logprob_cache` to 8 entries (LRU, sorted-pair key,
  double-checked after the lock), the session clears it on edit, delete, reset,
  finalize, and restore, and `reorient_for_request` flips a hit to the
  requested order.
- **Manifolds.** Detail and list rows are
  `store.manifold_lifecycle.manifold_summary` (the shape `pack show -j`
  prints) plus session-aware `fitted_for_session`, `stale` (sidecar weight
  fingerprint or `nodes_sha256` mismatch), `resolved_fit_mode` (`pca` or
  `spectral` once a discover folder is fitted, `null` for an unfitted `auto`),
  and a fitted discover folder's per-model geometry. The list stays cheap:
  `_fitted_geometry` reads only the safetensors header and the small
  `node_coords` entry (affine = no `layer_<L>.node_params` key), the weight
  fingerprint is computed once, and an unreadable artifact reports unresolved
  geometry instead of failing the list. Authored creation returns soft
  poisedness/flat-axis `advisories` before a fit is paid for. `install`,
  `generate` (resumes by default; `force` starts clean), and `{ns}/{name}/fit`
  are progress operations. Steering a fitted manifold needs no route; a `%`
  term loads it on scope entry.
- **Templates.** CRUD is pure IO; `score` is a progress operation over
  `session.score_template`.
- **Instruments** (`S/instruments/{family}`). `CAPABILITIES` declares what each
  family supports and what an unsupported call returns:

  | Family | `sources` | `preparations` | `token-readout` | `PUT source` |
  |---|---|---|---|---|
  | geometry | 404 | 404 | yes | 404 |
  | lens | yes | `fetch`, `fit` | yes | yes |
  | sae | yes | `fetch`, `train` | yes | 409: loading weights is the `fetch` preparation |

  `POST live` forwards extras to `instrument.set_live`, whose `TypeError` is
  the one rejection rule (400); `top_k` is refused for every family because
  readout width is the generation's `return_top_k`. `PUT source` (lens) is
  refused while a lens preparation runs, then turns live off, selects, and
  turns it back on (`LensNotFittedError` 409). `token-readout` runs
  `instrument.token_readout`, which builds the whole `scope="replay"`
  envelope and rejects knobs it can't honor (400); the route reshapes nothing.
  `sae/features/metadata` is network and disk cache only and takes no lock.

## Long-running operations (`operations.py`)

**Progress operations** (`/extract`, manifold `install` / `generate` / `fit`,
template `score`) call `run_operation`. The job runs in a server-owned task
that holds the session lock for its real lifetime; the response is a view.

- `Accept: text/event-stream` waits for the lock unbounded, then streams
  `event: progress {message}` and one terminal `event: done <payload>` or
  `event: error {message, code, status}`, with a `: heartbeat` comment after
  15 s idle so a proxy read timeout cannot drop a slow job. Progress is
  coalesced off the worker thread into a 256-line queue (oldest dropped); the
  terminal frame never waits on a consumer that left.
- Otherwise one JSON body under the bounded lock wait (`/extract` adds its
  `progress` lines).
- Without a receipt, an SSE client that leaves while the job is still queued
  withdraws it; a started job always runs to completion.

**Receipts.** `X-Drowse-Request-Id` (1–128 of `[A-Za-z0-9._:-]`) admits the
operation into `OperationLedger` under a fingerprint of `[kind, path,
model_id, body]`. A reused id is 409 (read the receipt instead); 128 active
receipts is 503. `GET /drowse/v1/operations/{id}` returns `state`
(`queued|running|completed|failed|interrupted|cancelled`), the last 64
progress lines, `result`, and `error`; 404 means unknown or evicted, never
completed. The ledger keeps 256 records and 16 MiB of payloads (4 MiB each;
past budget the oldest finished payloads drop and are marked
`result_truncated`, receipts stay). Shutdown cancels queued jobs (receipts
read `interrupted`) and joins started ones.

**Preparations** (lens `fetch` / `fit`, SAE `fetch` / `train`) are polled,
never streamed: `POST S/instruments/{family}/preparations {operation,
...fields}` (202; fields re-parsed into the operation's model), `GET` for the
running-or-latest status, `DELETE` to cancel. Status: `{state:
idle|running|done|error, operation, progress: {current, total, unit} | null,
message, error, started_at, finished_at, cancellable, ...extras}`; fit counts
`prompts` and train `tokens`, parsed from the job's own progress lines. A
failure reports `message: "<label> failed"` with the scrubbed error; a
cooperative cancel settles to `state: "done", message: "cancelled"`. `DELETE`
cancels a running lens fit (else 409; a fetch is not cancellable) or SAE train
(else unloads the resident SAE). Shutdown sets cancel events and joins.

- Lens `fetch` takes a provider from `LENS_PROVIDERS` (400 at POST),
  downloads into the HF cache without the lock, then selects it and goes live.
- Lens `fit {kind: r|j = r, ...}`: `kind` picks the estimator and destination
  `local:<kind>`; `layers=sample` is 400. The corpus is pinned in the isolated
  spawned corpus worker (cancellable, never resolved in the server process).
  Unless `force`, a proven no-op (`session.lens.fit_is_noop`) only selects
  `local:<kind>`; otherwise documents stream in the worker and
  `session.lens.fit` runs.
- SAE `fetch` takes `local:<name>` or `saelens:<release>` (a bare release is
  400) and loads under the lock. SAE `train` streams `ceil(tokens / seq_len)`
  default-corpus documents, then trains.
- Live readout after a fit or train is best-effort: a failure is logged, not a
  failed preparation.

## WebSocket: `WS S/stream`

Inbound frames (`ws_models.py`):

- `submit` is the only authored-turn contract. `text` + `authored_role`
  appends a span; `generated_role` then generates from it in that seat, and
  alone continues from the selected leaf. Append-only uses
  `session.append_turn` (no decode); append+generate commits the turn once
  inside the generation worker, under the fan's lock. The authored turn's cast
  label rides `sampling.user_role` / `assistant_role` by seat. Cross-field
  mistakes are error frames with code `RequestError`.
- `generate` is the specialist frame: continue (`input: null`), explicit
  message lists, seats (`generate_seat`), `recipe_override` (composed by
  `loom/recipe.py::compose_call`), and the token fork. `stateless` defaults to
  `true`. `system_prompt` omitted inherits the default; `null` sends none. A
  fork needs `fork_node_id`, `fork_raw_index`, and exactly one of
  `fork_alt_token_id` / non-empty `fork_replacement_text`, and takes steering
  and sampling from the source node's recipe. A message's `label` is the
  per-turn cast label the scene stitcher renders, so a replayed
  custom-labelled turn keeps its prompt. Field-group rules live in the model
  validators (`PydanticCustomError` keeps messages verbatim).
- `stop {request_id?}` stops the running turn (only a matching one when
  `request_id` is set); a no-op when idle.

Outbound frames:

- `started {generation_id, node_id: null, sibling_index, sibling_count}` per
  sibling.
- `token {text, thinking, token_id, node_id, raw_index?, logprob?,
  perplexity?, top_alts?, measurements?}`. `measurements` is the token-scope
  envelope the tap built and stored on the loom row, forwarded verbatim
  (`ws_events.build_token_event`), so live and rehydrated tokens share one
  authority. The consumer opts into every live channel; each family's live
  toggle decides what appears.
- `done {result: {text, tokens, finish_reason, applied_steering, usage,
  measurements?, mean_logprob, mean_surprise}, node_id, sibling_index,
  sibling_count}`, with the engine's aggregate-scope envelope. After a client
  `stop`, `finish_reason: "stop"` reads `"cancelled"` on the wire only; the
  loom recipe keeps the engine's reason. A commit-only `submit` sends one
  `started` and one `done` (`{role, text, node_id, ...}`), no tokens.
- `tree_mutated {op, rev, added, removed, updated, active_node_id, cast}`
  forwards every `LoomMutated`, from any source. `done` waits until its node's
  `finalize_assistant` delta is sent, so a client never sees a finished but
  empty node.
- `error {message, code, status, node_id, sibling_index}` on a connection that
  stays open: FastAPI's handlers don't reach WebSockets, so the handler
  classifies with `describe_error`. Schema failures, bad JSON, and unknown
  types carry code `ValidationError` and list every error (a union field's
  real failure is often the second branch).
- `request_complete {request_id, state, completed_siblings}` ends every
  correlated turn.

A `request_id` admits a turn into the same `OperationLedger` (kind
`generate`), and every frame of that turn carries it. Resubmitting a finished
id replays its recorded frames without generating. A live id or different
parameters gets an `error` coded `REQUEST_RECONCILE_REQUIRED` /
`REQUEST_ID_CONFLICT` (this frame has only `type`, `request_id`, `code`,
`message`); a full ledger, `REQUEST_QUEUE_FULL` (503).
`GET S/requests/{id}` reports `state: "unknown"` when absent, never 404.

Concurrency and limits:

- One reader task owns `receive_text()` (the `websockets` receive flag makes
  overlapping receives a `RuntimeError`) and feeds a bounded FIFO: 1 MiB per
  frame (close 1009; the CLI passes the same `ws_max_size`) and 16 pending
  frames / 4 MiB (close 1013). `stop` is a coalesced control outside that
  budget. Turns run one at a time per connection.
- Sends share one lock with a 30 s timeout. Token and tree queues each bound
  256 outstanding events, counting callbacks still pending on the loop;
  overflow closes 1013.
- A turn runs in a worker thread under
  `session.generation_state.cancellation_scope`. `n > 1` fans siblings
  serially under seeds from `loom.recipe.derive_seed_schedule`, holding the
  session lock for the whole fan; a stop cancels the current sibling and skips
  the rest, and an error aborts the rest.
- Disconnect cancels the turn and joins only this connection's worker. An
  unexpected exception sends an error frame and closes 1011.

## Deliberately absent

Refused designs; do not re-add them.

- A multi-session server: `{session_id}` exists for the URL shape only.
- `/packs` routes or an HTTP push; HF upload is CLI `pack push`.
- Per-family `/lens/*` or `/sae/*` groups; all of it is
  `/instruments/{family}`.
- One-shot text-scoring endpoints, which score text out of conversation
  context. Reads are live hooks or a `token-readout` replay.
- A traits SSE stream, or per-token aliases on the `token` frame; everything
  is inside `measurements`. Likewise no flat instrument keys on
  `session_info`.
- Tools, structured output, or embeddings on either compat protocol.

## Testing

All CPU-only, under `tests/server/`. `test_request_errors.py` pins the error
policy on every surface; `test_http_security.py` the trust boundary;
`test_server_operations.py` the ledger, runner, preparations, and shutdown;
`test_server_streaming.py` worker lifetimes; `test_webmcp_ws.py` WS receipts
and scoped stop; `test_server_manifold_eviction.py` runs a real tiny session
to prove folder changes over HTTP leave no stale steering. `test_server.py`
covers OpenAI/Ollama and `test_drowse_api.py` the native routes and WS frames.

`tests.support.server.mock_session(**overrides)` is the one route mock, a
`create_autospec(DrowseSession, instance=True)`, so a route calling a removed
or renamed session method fails. When a route test needs real behavior, use
`tiny_session` + `app_client` (`tests/conftest.py`); `StubSession` and
`script_generate` script WS generations.
