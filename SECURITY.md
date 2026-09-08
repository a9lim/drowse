# Security Policy

## Reporting a vulnerability

If you've found a security issue in drowse, please report it privately rather than filing a public issue.

- **GitHub:** use [private security advisories](https://github.com/a9lim/polythetic/security/advisories/new)
- **Email:** [contact@drowse.ai](mailto:contact@drowse.ai)

Please include a description, reproduction steps, affected version, model ID, and
whether the server was reachable beyond localhost. Do not include API keys,
private prompts, or credentials in the report.

## Supported versions

Only the latest release on PyPI receives security fixes. Upgrade before reporting
an issue that may already be fixed.

## Known dependency advisory

As of September 7, 2026, NLTK 3.10.3 has no published fix for
[CVE-2026-81726](https://github.com/advisories/GHSA-8mgp-746c-j5xp).
SAE Lens requires NLTK, so it remains in the Python dependency set. Drowse does
not call the affected NLTK model import/export APIs or use NLTK's `pathsec` as a
sandbox. This limits Drowse's exposure but does not fix the dependency itself.
Do not pass untrusted model-file paths to NLTK APIs in applications embedding
Drowse. Upgrade when a patched NLTK release becomes available.

## Threat model for `drowse serve`

The HTTP server (`drowse serve`) is designed for a single trusted user on a local
machine or trusted lab network. It is not a hardened multi-tenant service and
should not be exposed directly to the public internet.

What it does:

- CLI binding defaults to `127.0.0.1`. Non-loopback binds require bearer auth
  via `--api-key` or `$DROWSE_API_KEY`. Without a key, HTTP/WS also reject
  non-loopback TCP peers, including programmatic `create_app` deployments;
  changing the Host header cannot bypass this check.
- Browser API requests and WebSockets must have a valid same origin or an explicit `--cors`
  origin, even with a valid key. `--cors '*'` does not authorize arbitrary
  API or WebSocket origins. Clients without an Origin header still use normal auth.
  Without a key, API Host headers must identify loopback or the ASGI server
  address, preventing arbitrary DNS rebinding hosts from gaining access.
- API authentication runs before request-body parsing. HTTP bodies are limited
  to 64 MiB, counting received bytes even without Content-Length. Set
  `DROWSE_MAX_REQUEST_BYTES` or `create_app(max_request_bytes=...)` to change the
  limit; the explicit argument wins.
- The browser dashboard sends its WebSocket credential in the handshake's
  `Sec-WebSocket-Protocol` header and negotiates only `drowse.v1` back. Legacy
  `?token=` clients remain supported, with tokens removed from the ASGI query
  before Uvicorn logs the handshake. Proxies can still log a legacy client's
  original URL; use header authentication or redact those query parameters.
- API responses use `Cache-Control: no-store`. The Python dashboard has a
  Content Security Policy, blocks framing, omits referrers, disables MIME
  sniffing, and denies unused sensitive browser permissions.
- HTTP filesystem failures and wrapped Hugging Face transport errors keep
  paths and signed download URLs out of client responses. Full diagnostic
  exceptions remain available in local server logs.
- Native WebSocket inputs have a 1 MiB frame limit and a pending budget of 16
  messages / 4 MiB. Stop and disconnect controls do not consume that budget.
  Each outbound token/tree queue holds at most 256 events, including pending
  thread callbacks; slow consumers are disconnected. Sends time out after
  30 seconds. These limits do not cap model memory or individual output size.
- A bounded async session lock serializes generation-facing OpenAI, Ollama, and
  native requests before they enter the engine; the synchronous session also
  rejects generation re-entry. HTTP streams start their engine worker only after
  acquiring that lock. Blocking inference, token reads, and worker joins run off
  the ASGI event loop. Cancellation and send failures retain ownership until
  the worker exits; background artifact jobs follow the same rule.
- Native progress streams and JSON progress histories retain at most 256
  recent messages. Thread callbacks are coalesced and completion/error frames
  remain deliverable without waiting for a disconnected consumer.
- Pydantic validates protocol request bodies; native request models reject unknown
  fields.
- Installed manifold payloads and Drowse-owned fitted artifacts are checked
  against their declared SHA-256 digests before use; external J-lens/SAE sources
  are commit- or release-pinned through local bindings.

What it does not do:

- Rate limiting, quotas, per-user isolation, or audit logging
- Resource isolation from deliberately expensive prompts, sampling options,
  fitting jobs, or repeated downloads
- TLS; use a correctly configured reverse proxy if HTTPS is required
- Sandboxing for model code, tokenizers, checkpoints, or downloaded artifacts

If untrusted callers need access, add authentication, TLS, deployment-specific request limits,
rate limits, and process-level isolation outside Drowse. A reverse proxy alone is
not a complete isolation boundary.

## Model and checkpoint trust

Model loading disables repository Python by default, including configuration
and tokenizer modules declared through `auto_map`. Models requiring custom
Python must be explicitly trusted: pass `trust_remote_code=True` to
`DrowseSession.from_pretrained` / `load_model`, or set
`DROWSE_TRUST_REMOTE_CODE=1` for a CLI invocation. The environment also accepts
`true`, `yes`, and `on`; an explicit Python `False` overrides it. This opt-in
allows arbitrary Python execution with the process's permissions. Use only
publishers and revisions you trust. Native model implementations remain
preferred when Transformers supports the architecture.

Metadata-only lens fetches, model-shape/source checks, and offline cache checks
always use `trust_remote_code=False`, regardless of that environment opt-in. An unsupported custom configuration fails closed
on these paths; fetching metadata must not execute repository Python.

`drowse pack install <owner>/<name>` verifies files declared by the manifold's
`manifold.json` integrity map, but integrity is not authorship or safety. Manifold
metadata and corpora remain untrusted input; install only from publishers you
trust. Provider-owned J-lens and SAE payloads remain in their provider cache and
are pinned through local bindings.

## Local state and conversation imports

State written through Drowse's atomic JSON/byte helpers uses exclusive, unique
temporary files and publishes with owner-only permissions on POSIX (0600).
Payload data is synced before replacement, followed by a best-effort directory
sync. This prevents predictable staging-file symlink attacks and staging
collisions between writers. It does not encrypt the contents or change the
permissions of untouched older files, parent directories, or external backups.

`LoomTree.load` accepts at most 64 MiB of main JSON and 256 MiB of decompressed
token-sidecar JSON by default. Both limits apply before parsing; a positive
integer `max_bytes=` explicitly changes both for trusted large exports.
These limits constrain input expansion, not total process memory or compute.
Tree JSON and token sidecars are individually replaced but are not a single
crash-atomic transaction; interrupted overwrites can leave a mismatched pair.

## Browser data

Hosted conversations, fitted artifacts, and model downloads are stored locally
in browser storage. They are not encrypted by Drowse. The service worker caches
application assets; it does not cache conversation API responses. Published SAE
descriptions for the verified Gemma Scope 2 packs ship as application assets.
Missing descriptions fall back to Neuronpedia, sending only the public
dictionary/feature identity without cookies, referrers, activation values, or
conversation text.

"Clear all local data" stops persistence and clears the application's saved
sessions, artifacts, and preferences, including legacy Polythetic/Saklas
databases and files that migration could otherwise restore on reload. Legacy
directory handles remain usable after their contents are removed. A failed deletion is reported and can be
retried; it must not silently report success. Browser profiles, extensions, OS
accounts, and external backups remain outside this deletion boundary.
