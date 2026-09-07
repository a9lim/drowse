# Base completion interface polish

Reviewed 2026-09-05 with `better-interface` (full) and `make-interfaces-feel-better`. This follow-up records only the current interface work; the earlier support and release review remains separate.

## Scope and Coverage

Svelte 5, plain CSS, existing semantic colors, Wix Madefor interface type, Martian Mono data type, shared buttons and segmented controls. Scope: the base completion editor, saving/discarding, generation/stopping, sampling controls, token inspection and replacement branches, and missing optional J-lens states. The real model picker’s unavailable-download state was inspected, but base-model installation and inference are outside this UI approval.

Browser interactions used the development-only `layoutFixture=base` runtime. Its deterministic completion is not model inference. The desktop view and 320px/768px iframe previews were inspected. No deployment, model/compiler change, new comparison feature, dependency, or version bump was made.

| Domain | Evidence inspected | Result |
| --- | --- | --- |
| Accessibility | Editor/label/error association, save/discard focus, streaming locks, keyboard token navigation, drawer focus return, sampling inputs | 2 resolved findings; screen-reader certification not claimed |
| Layout | Empty and long editors; save, generating and finished states; compact status/control geometry | 1 resolved finding |
| Writing | Save versus generate status, base/instruct control labels, inspection colors, optional J-lens messages, branch copy | 3 resolved findings |
| Typography | Monospace editor, numeric metrics, line wrapping and narrow input size | Clear; existing type tokens and tabular figures retained |
| Colors | Light/dark rendered editor and controls, token/focus styles, theme and contrast checks | Clear; no palette changes |
| UI | Stream-follow behavior, scroll-back escape, inspection availability, shared press treatment and loading glow | 1 resolved finding; existing reduced-motion policy inspected in source |

## Findings

All findings below were implemented. Locations point to the resulting source.

| # | Severity | Domain | Location | Before | After | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | MEDIUM | UI | `webui/src/panels/RawBuffer.svelte:420` | Auto-scroll targeted the enclosing div, not the textarea. A long completion had textarea scrollTop=0 despite 898px of content in a 346px viewport; the empty outer surface also scrolled 5px. | Scroll the active text surface after rendering; respect manual scroll-back; offer Jump to latest text. A block textarea removes the inline baseline overflow. Verified textarea scrollTop=552, outer scrollTop=0, and zero remaining distance after Jump. | Keeps new output visible without taking away the ability to read earlier text. The jump control overlays the surface without resizing it. |
| 2 | MEDIUM | Accessibility | `webui/src/panels/RawBuffer.svelte:104`, `:228`, `:448` | Save/discard removed the focused action; browser focus fell to BODY. The visible title was not associated with the editor. | Return focus after completion when focus is still on the originating control or BODY; discard and Jump also return to the editor. Associate the visible label and expose an error’s invalid state. | Keyboard users can continue editing immediately; delayed saves do not steal focus from another selected control. Browser checks confirmed TEXTAREA focus after save and discard. |
| 3 | MEDIUM | Writing | `webui/src/panels/RawBuffer.svelte:510`; `webui/src/panels/StatusFooter.svelte:70`; `webui/src/panels/Chat.svelte:856` | Saving an authored edit displayed zero-token generation metrics and announced Completion finished. | Display/announce Edit saved. Distinguish a zero-token generated completion from an authored save. Unknown/error finishes say Ended, not Complete. | Saving text is not model generation. Tests cover save, cancellation, token limit, unknown finish and zero-token output. |
| 4 | MEDIUM | Writing | `webui/src/hosted/ui/JLensMissingState.svelte:9`; `webui/src/hosted/ui/JLensSourceSection.svelte:39` | Missing word insights told base-model users to download the model again. | Explain that J-lens is optional, this view needs a compatible pack, and text completion still works without it. Instruct-model wording is unchanged. | Avoids treating an intentionally omitted optional tool as a broken model download. The old and new messages were both observed in token inspection. |
| 5 | MEDIUM | Accessibility | `webui/src/panels/RawBuffer.svelte:322` | Inspect tokens could become enabled during generation even though it was intended for a settled buffer. | Include the active-generation guard and explain when inspection becomes available. Continue from inspection returns to the editable view. | Keeps availability consistent with the read-only generation state. Browser and rendered-component checks confirm the lock. |
| 6 | MEDIUM | Layout | `webui/src/panels/StatusFooter.svelte:94`, `:137` | Status used one nowrap, overflow-hidden row, with the finish reason after secondary metrics. Independent separators remained when metrics were hidden. | Put the terminal reason with the token count; wrap complete metric groups; remove loose separators. | Prioritizes the outcome over speed statistics. At 320px, the completed status measured 239px clientWidth and 239px scrollWidth; primary controls remained reachable. |
| 7 | LOW | Writing | `webui/src/App.svelte:485`; `webui/src/panels/ControlsPanel.svelte:17`; `webui/src/panels/InspectorPanel.svelte:66`; `webui/src/panels/SteeringRack.svelte:37`; `webui/src/panels/ProbeRack.svelte:38`; `webui/src/panels/Chat.svelte:1115`; `webui/src/drawers/TokenDrilldownDrawer.svelte:895` | Base controls called the output a response/reply; the color control did not identify the inspection view. | Completion-specific labels, help and branch copy; Inspection colors / Color inspected tokens by. Internal keys and instruct labels stay unchanged. | Matches a continuous-text workflow without renaming protocol fields or altering model behavior. |

## Considered but Rejected

| Location | Candidate | Rejected because |
| --- | --- | --- |
| `webui/src/lib/ui/Button.svelte`; `webui/src/lib/style/global.css` | Add a separate animation dependency, spinner, or token-by-token entrance effects | Shared 0.96 press feedback and the existing loading glow already cover discrete controls; streamed text should remain immediate. |
| `webui/src/lib/NumberInput.svelte` | Add separate keyboard stops for the mouse-only spinner pair | The native labeled spinbutton already provides arrow-key operation. Its duplicate steppers are aria-hidden and absent from tab order. Narrow layouts omit them. |
| `webui/src/panels/RawBuffer.svelte` | Give every inline token a 40px box | Inline token controls need to preserve text flow; the roving keyboard stop and shared drawer provide navigation. Main buttons already measured 40px on desktop and 44px in the responsive preview. |
| Loom map and generation recipe | Rename structural user/assistant roles or sampling/protocol keys to completion language | Those are provenance, not conversational instructions. Their identities must remain faithful to the saved tree. |

## Verification

Passed:

- `cd webui && npm run check`: final run exits 0; Svelte reports 0 errors/0 warnings. Theme, contrast, runtime boundary, entry, interface policy, backups and base-mode checks pass. Log: `/tmp/polythetic-base-interface-final-check.log`.
- `cd webui && node scripts/base-model-ui.test.mjs`: original base/instruct mode checks plus rendered-component regressions for readonly generation, inspection guards, visible label association, stop/token-limit/unknown outcomes, authored saves, zero-token model output, completion wording and optional J-lens copy. This script is already included in `npm run check`.
- `cd webui && node scripts/sampling-store.test.mjs && node scripts/loom-store.test.mjs && node scripts/browser-loom.test.mjs`: sampling/queued edits, tree invalidation/streaming, and authoritative branch tests pass.
- `cd webui && npm run build`: native build and isolation pass. Log: `/tmp/polythetic-base-interface-native-build.log`.
- `cd webui && npm run build:hosted`: hosted build and isolation pass. Log: `/tmp/polythetic-base-interface-hosted-build.log`. Both builds retain the existing large-chunk advisory.
- Browser: save and discard return focus to the editor; save displays and announces Edit saved; Ctrl+Enter continues the text; inspection stays disabled while streaming; Stop displays Stopped and unlocks the controls.
- Browser: a 40-line prefix now follows new output. Manual scroll-back exposes Jump to latest text, and Jump returns the textarea to the end. Empty editor no longer has outer baseline overflow.
- Browser: Inspect tokens → ArrowRight → Enter opens the next token; Escape returns focus to the originating token. Replacement text `was ` creates a distinct Loom sibling while the original `is ` path remains visible. The resulting branch retains the `I love marmots because` prefix.
- Browser: 320px and 768px completion previews have no editor horizontal overflow. The 320px sampling sliders and number input measured 44px tall. Light/dark states, invalid logit-bias field feedback, optional J-lens guidance and the actual unavailable-download setup state were inspected.
- Browser: the final workbench error/warning log query returned no entries.
- `git diff --check` passes for tracked edited UI files. Pre-existing untracked components/tests were also inspected directly; no claim is made that Git diff covers those files.

Not verified:

- VoiceOver, physical mobile keyboards, 200% browser zoom, and an actual reduced-motion browser session. Reduced-motion source rules were retained, not reimplemented.
- A browser-injected asynchronous save/storage failure. The retained-draft/error path and invalid-state association were inspected; the unknown/error terminal label has rendered-component regression coverage.
- Production base-model installation, model accuracy, GPU capture/replay or release readiness. The development fixture proves UI behavior only; no downloads were enabled or published.

The shared footer’s new `savedEdit` prop defaults to false; its chat-mode caller remains unchanged. The completion caller supplies the authored-save condition. Other changes are local focus/scroll state, presentation text, and CSS; no wire contract or generation algorithm changed.

## Verdict

Approve
