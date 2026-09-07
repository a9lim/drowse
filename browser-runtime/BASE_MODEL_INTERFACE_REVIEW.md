# Base completion interface review

Reviewed 2026-09-05. Runtime repairs and interface improvements are implemented locally; this is **not a model-release approval**.

## Scope and Coverage

Full `better-interface` review with `make-interfaces-feel-better`, covering the base-model picker/session identity, completion editor, generation controls, token inspection, Loom edits, and backup confirmation. Svelte 5, existing CSS tokens, shared controls and motion conventions. Unrelated pages were not redesigned.

Browser interactions used the development-only `layoutFixture=base` runtime. Its deterministic output proves interface behavior, not Pythia inference. Desktop interactions and a 320px iframe layout were inspected; iframe keyboard interactions and physical mobile devices were not tested.

| Domain | Evidence inspected | Result |
| --- | --- | --- |
| Accessibility | Editor label and error association; token arrow/Enter navigation; drawer Escape/focus return; terminal announcements; counter DOM | Focus and status fixes; no assistive-technology certification |
| Layout | Empty, populated, editing, generating and stopped buffers; action grouping; narrow-width preview | Header/actions wrap; separate edit and generation groups |
| Writing | Editor hints, Continue/Save/Discard, inspection, completion identity, stopped/completed announcements | Completion-specific copy replaces misleading chat language |
| Typography | Editor/inspection text, hint wrapping, counters, narrow input size | Existing type system retained; mobile editor minimum 16px |
| Colors | Light/dark rendered states; shared neutral controls, accent and focus tokens; contrast script | Shared theme retained; checks pass |
| UI | Pending/committing controls, loading surface, draft retention, inspection and download interaction | Quiet card glow; no rotating decoration or new animation dependency |

## Findings

Implemented findings below are resolved unless explicitly marked outstanding. Locations refer to the resulting code.

| # | Severity | Domain | Location | Before | After | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | HIGH | UI | `webui/src/panels/RawBuffer.svelte:432` | A tinted mirror sat behind an independently scrolling textarea | One editable text surface; token coloring lives in Inspect tokens | Removes competing text layers and their scrolling/selection alignment risk |
| 2 | HIGH | Accessibility | `webui/src/hosted/runtime/browserLoom.ts:682`; `webui/src/panels/Chat.svelte:858` | An external-stop result could retain a successful finish reason | External stops store cancellation and announce Generation stopped | A partial completion must not be presented as finished; reproduced and retested in browser |
| 3 | MEDIUM | UI | `webui/src/panels/RawBuffer.svelte:197` | Boundary navigation failures escaped the edit error path; pending edits could overlap input | Busy guard, retained draft, persistent inline alert and boundary error handling | Keeps the recovery action next to the affected text; error branch checked in code, not browser-injected |
| 4 | MEDIUM | Accessibility | `webui/src/panels/RawBuffer.svelte:475` | Commit state did not lock the editing surface; token/editor focus lacked a dedicated keyboard outline | Read-only during commit/generation; explicit keyboard focus treatment | Prevents accidental overlapping edits and makes keyboard position visible |
| 5 | MEDIUM | Layout | `webui/src/panels/RawBuffer.svelte:484` | Editing and generation actions shared one undifferentiated row | Two wrapping groups; keyboard hint hidden on narrow screens | Preserves grouping without squeezing the input or primary action |
| 6 | MEDIUM | Writing | `webui/src/panels/Chat.svelte:858`; `webui/src/drawers/TokenDrilldownDrawer.svelte:783` | Response/chat language appeared within base completion tasks | Completion-specific announcements and token identity; Edit text / Inspect tokens hints | Clarifies that a base model continues a passage, rather than answering a chat message |
| 7 | MEDIUM | UI | `webui/src/panels/RawBuffer.svelte:450` | Assistant text without captured token rows could disappear in inspection | Preserve it as plain text, without inventing measurements | The inspection view must not silently omit saved content |
| 8 | LOW | Accessibility | `webui/src/lib/ui/RollingNumber.svelte:37` | Number accessibility relied on animation-library generated markup | Stable formatted text plus an aria-hidden visual animation subtree | Separates accessible content from animated digits; inspected DOM, not a screen-reader run |
| 9 | LOW | UI | `webui/src/panels/RawBuffer.svelte:432` | The completion input had no matching generating-state surface treatment | Existing loading-pulse surface style while generation is active | Gives a restrained activity cue consistent with the app; existing reduced-motion rule retained |
| 10 | LOW | Typography | `webui/src/panels/RawBuffer.svelte:631` | Input size followed only the global type scale | At least 16px for narrow input and inspection text | Keeps completion text readable at the tested narrow width |
| 11 | HIGH | UI | `browser-runtime/BASE_MODEL_SUPPORT_QA.md`, latest status section | Requested base families are not installable; public pinned artifact endpoints return 401 | **Outstanding:** exact runtime/pack validation and accessible published artifacts | Users still cannot complete the real download-to-generation task; do not enable a misleading Install action |

## Considered but Rejected

| Location | Candidate | Rejected because |
| --- | --- | --- |
| Completion editor | Keep animated token colors behind the textarea | One editable layer is more reliable; inspection already provides the token-color view |
| Completion editor | Add an animation library and a rotating generating icon | Shared surface motion already conveys activity with less distraction and no new runtime dependency |
| Completion input | Remove the keyboard outline to achieve an entirely soft glow | The soft focus shadow can coexist with a clear keyboard outline; visual restraint should not hide focus |
| Base model cards | Enable Install as soon as compiler code generation passes | Code generation is not successful checkpoint inference or a validated downloadable package |

## Verification

Passed checks:

- `cd webui && npm run check`: Svelte 0 errors/0 warnings, theme/contrast/boundary/interface policies, backup round trips, base-mode enforcement and instruct compatibility.
- `cd webui && npm run build:hosted`: build and hosted isolation pass; existing large-chunk advisory remains. The development preview is not a production build entry.
- `node scripts/qwen-hybrid-contract.test.mjs`: 53 checks pass against the **installed bundle**, using fake device objects. It is now part of `test:runtime`.
- `node scripts/browser-loom.test.mjs`: external-stop regression and authoritative Loom checks pass.
- `npm run test:runtime`: final sequential run passes all 44 scripts after isolating the two test harnesses from unnecessary dependency discovery. Log: `/tmp/polythetic-final-runtime-no-discovery.log`.
- `node scripts/worker-runtime.test.mjs`: 113 orchestration checks pass, including artifact routing after adding the missing fake-runtime method.
- `npm run test:fork-overlays` and `node scripts/check-runtime-lock.mjs`: regenerated overlay and packaged runtime identity checks pass. These are structural checks, not new GPU attestations.
- Browser: enter multiline text; Ctrl+Enter continues; Inspect tokens → ArrowRight → Enter opens token details; Escape closes and returns focus; edit/save creates a new Loom branch without deleting the original; saved model has `[BASE]`; download confirmation shows an editable filename and byte size; stop retains partial text and displays stopped status.
- Browser: Light/Dark switch, busy/read-only editor, disabled Continue/enabled Stop, and a rendered 320px iframe preview. DOM inspection confirms the counter's visual subtree is aria-hidden.

Verification limits:

- Fake-runtime browser output is **not** actual base-model generation, capture, replay or SAE execution.
- No VoiceOver, physical mobile keyboard, 200% browser zoom, or injected browser save-error run. Error retention is covered by the code path and underlying storage/backup tests, not a manually forced UI failure.
- Intermediate runtime runs hit native Node/Rolldown SIGSEGV/SIGBUS failures, including one after conversation-library assertions passed. After disabling unnecessary dependency discovery in the affected harnesses, five consecutive conversation-library runs and the complete 44-script chain exited successfully. The failed intermediate runs remain recorded in the QA log; they are not counted as passing runs.
- Release-mode lock checking requires a release revision; no release candidate, deployment or fresh GPU attestation was produced.
- All five anonymous pinned artifact-manifest requests returned HTTP 401. This does not establish whether the repositories are private, missing, or otherwise access-controlled.

## Verdict

Block
