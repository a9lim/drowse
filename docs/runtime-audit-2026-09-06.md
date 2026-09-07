# Runtime and lifecycle audit — 2026-09-06

Local audit of manifold fitting, steering, probes, replay, generation, Loom, persistence, and model-family integration. Existing worktree changes were preserved. No dependency upgrades, version bump, commit, push, or deployment was performed.

## Repairs

| Area | Reproduced problem and repair |
| --- | --- |
| Fitting capture cleanup | Affine-worker failure or cancellation could leave committed activation captures behind. Cleanup now covers that boundary, including cancellation after the last layer. Explicit capture retention also works when authored sigma fitting is disabled. |
| Preference subscriptions | Async bootstrap discarded the persistence effect's disposer. The mounted workbench now owns a single persistence subscription and clears its pending write on disposal. |
| Replay queue | Pending token readouts were unbounded even though settled results were capped. The queue now admits at most 96 pending jobs, retains deduplication and interactive priority, and rejects overflow with a retry message. Synchronous transport failures release the active slot; finished jobs clear progress listeners. |
| Probe refresh | Out-of-order list responses could overwrite newer state, erase attachments, or resurrect detached probes. Request and mutation revisions prevent stale commits. Failed refreshes preserve known probes; unchanged catalogs no longer invalidate readouts. Lens and SAE roster mutations now invalidate their own replay caches as geometry already did. |
| Re-fit experience | Duplicate clicks are ignored, typed cancellation is informational, and successful fits refresh the catalog, probe roster, vector metadata, and open diagnostics. Old inspector responses cannot overwrite new diagnostics. Late probe attachments cannot close a different drawer. |
| New-chat reset | An explicit reset previously removed incompatible autosaves but restored compatible ones. Reset now clears only the selected model's current autosave regardless of compatibility. Named saved chats use a separate store. |
| Loom identity | Every freshly loaded tree previously started with the literal root ID `root`. New chats therefore collided in the saved-conversation lookup, triggering stale-revision autosave failures. Fresh roots now have unique IDs. Restored tree identities are unchanged. |
| Build verification | Hosted-shell assertions still expected retired landing-page copy. Description validation now uses the canonical metadata source, and the local-compute assertion matches the current rendered promise. |

The probe refresh, retained-capture failure, compatible-session reset, and fresh-root collision regressions failed before their respective repairs and passed afterward.

## Automated verification

- Full Python suite: **3,789 passed, 20 skipped**, 11 warnings, 17m16s. No Python engine code was changed in this audit.
- Full browser runtime suite: passed after the final runtime changes. Includes authoritative Loom, generation, hybrid-cache contracts, steering compiler/kernel parity, instrument replay, storage, worker cancellation, fitting coordination, and artifact provenance tests.
- New lifecycle tests: 100 persistence mount/unmount cycles; single-owner replacement; pending-write cleanup; out-of-order probe responses; attach/detach races; unchanged-catalog cache reuse; 500 replay-queue overflow attempts; cancellation/invalidation; synchronous transport recovery; duplicate fits; re-fit diagnostics races.
- Rust fitting suite: **38 passed**, including golden fixtures, allocation ceilings, and adversarial topology/coordinate cases.
- Hosted fitting WASM reproducibility check: passed.
- Svelte checking: **0 errors, 0 warnings**. Theme, runtime-boundary, interface-policy, backup, metadata, and lifecycle checks passed.
- Native dashboard and hosted production builds: passed. Existing large-chunk warnings remain.
- Hosted build isolation and preview-shell verification: passed.
- Hosted release tooling/preflight tests and runtime/distribution lock validation: passed. Lock validation used the existing HEAD revision; it does not certify or deploy the uncommitted application changes.
- `npm audit`: **0 reported vulnerabilities**, including development dependencies.
- Python dependency consistency (`pip check`): passed.

## Live model checks

The browser checks used the locally built app and real installed model weights, not the fake runtime. Each of the four base-model families started a new conversation, generated, saved separately, and returned to the chat list after unloading.

| Model | Observed result |
| --- | --- |
| Pythia 70M Deduped Base / GPT-NeoX | 24-token completion; separate new-chat autosave with the older named chat still present. An additional run with `default/welcoming.detached` steering and a live probe produced finite per-layer readings and a bounded 60-reading history. |
| GPT-2 Base | 24-token completion, followed by a sibling generation in Loom: 3 turns, 1 fork, 48 generated tokens. |
| Qwen 3.5 2B Base / hybrid architecture | 24-token completions; an active longer generation stopped at 40 tokens; a subsequent 24-token generation completed successfully. |
| Gemma 3 1B PT | 24-token completion and autosave. Raw output included `<start_of_image>` markers. This is a runtime/integration pass, not a clean-text or semantic-quality pass; output was not silently filtered. |
| SmolLM2-360M-Instruct / native MPS | `scripts/check_manifold_model.py --device mps` passed six flat/controlled-curved injection checks at 0 and ±0.3. Outputs stayed finite; zero strength matched baseline within tolerance; removing each hook restored baseline exactly. |

The native curved fixture is constructed in a real activation frame. It verifies numerical integration, not discovery of a semantic manifold or calibrated steering quality.

The four browser runs remain in Your chats as `QA Sep 6 — Pythia`, `QA Sep 6 — GPT-2`, `QA Sep 6 — Qwen`, and `QA Sep 6 — Gemma`. The five pre-existing saved chats remained listed separately; the QA model was unloaded at the end.

## Boundaries

- This is not proof that the entire application is bug-free or leak-free. Resource cleanup was checked through ownership, queue, worker, storage, and repeated-lifecycle regressions; no long-duration GPU/JS heap-retention profile was performed.
- Live browser checks cover the four installed base-model families on this Mac/browser. Other model variants, platforms, mobile memory pressure, device loss, and all optional SAE/J-lens combinations were not exhaustively exercised with real weights in this pass.
- Full real-model browser authoring/re-fitting across every family was not run. Fitting was covered by coordinator/worker/storage regressions, WASM/Rust parity, and the native real-activation smoke test.
- Twenty Python tests were skipped; those are not counted as passes. Passing tests do not establish semantic manifold validity or output quality.
- Several pinned packages have newer registry releases (including Svelte, Vite, TypeScript, React, Playwright, and Three.js). No blanket upgrades were made: zero audit findings is not equivalent to using every latest version, and compiler/runtime changes require their own compatibility pass.
- No remote code-review service was available. Review was local, with the reported tests and live checks.
