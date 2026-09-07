# Interface polish pass — 2026-09-07

Applied `make-interfaces-feel-better` across the public pages, setup, saved chats, workbench, shared controls, and drawer families. This is a UI assessment, not a certification of model behavior or every possible runtime state.

## Changes

| Area | Applied changes |
|---|---|
| Shared controls | Raised desktop workspace targets from 28–32px to at least 40px; retained larger mobile targets. Theme buttons now have separate, non-overlapping targets. Brand links, standalone footer/source links, manifold launchers, and the composer resize handle are included. |
| Mobile header | Preserved one row, as requested. Navigation scrolls horizontally, and keyboard focus reveals offscreen links. Theme controls and the wordmark stay outside that scroller. |
| Saved chats | Kept the recent flat materials, 8px card corners, padding, and inline rename behavior. The shuffle icon uses the requested scale/opacity/blur values. Color choices retain their checkmarks in the DOM so selection can crossfade in both directions. The color popup radius follows its actual inset. |
| Contextual icons | Added a shared, fixed-size icon stack for composer actions and device-check states. Variants remain mounted: scale 0.25–1, opacity 0–1, blur 4–0px, 300ms easing without a new animation dependency. |
| Help and token popovers | Help tips now retain their native top-layer surface during dismissal, support reversal, and keep Escape local to the tip. Token probability popovers fade through their parent conditional, disable outgoing controls, restore focus, and scroll within the space available above or below the anchor. |
| Page transitions | Headings, status, and primary content enter in separate groups with 100ms staggering. The first resolved page does not animate on initial load. Outgoing pages are inert; rapid reversal restores interaction. Reduced motion skips staged entrances. |
| Drawers and palette | Outgoing controls become inert before the visual exit finishes. Reopening restores interactivity. Existing focus traps and short exits are retained. |
| Workbench and Loom | Added reversible feedback to remaining sidebar controls; enlarged Loom action targets. The phone composer gives the primary label more space instead of splitting a word. Scientific values, token text, and branch semantics are unchanged. |
| Token details | Made the horizontally scrolling context ribbon keyboard-focusable. Arrow-key token navigation and automatic centering remain available from that single tab stop. |
| Pinned sidebars | Added a mirrored right-hand header toggle. Full token details occupy their own persistent sidebar, independent of modal tools; closing another tool, changing workspace views, or pressing Escape does not dismiss it. Token clicks update the inspector. Both sidebars slide with the same interruptible 300ms timing; reduced motion disables the slides. On phones the inspector remains a separate lower pane. Hidden inspectors do not start new replay requests. |

## Existing behavior retained

- Root font smoothing, balanced headings, paragraph wrapping, tabular dynamic values, and inset image outlines were already present.
- Existing shared controls already use explicit transition properties and an opt-out for press scaling. No animation package was added.
- Layered popup depth, forced-color boundaries, and reduced-motion behavior were retained. The deliberately flat workspace and saved-chat materials were not replaced with elevated cards.
- Removed the unconditional compositor hint from dropdowns; it was not supported by evidence of first-frame stutter.
- Inline token fragments and dense chart marks are not expanded into separate 40px boxes: that would alter prose and scientific geometry. Their containing controls and existing keyboard navigation remain the accessible interaction paths. Minimum-size assertions cover conventional controls, not every data mark.

## Verification coverage

- `npm run check`: zero Svelte errors or warnings; all policy, runtime-boundary, color, backup, and lifecycle checks passed.
- Default and hosted builds passed. The existing large-chunk advisory remains; this pass does not claim a bundle-size optimization.
- 50 distinct targeted browser cases passed across Chromium and iPhone-profile WebKit, including reruns after updating legacy layout assertions. One transient browser-evaluation failure passed on rerun without a production change.
- Public landing, Credits, 404, and setup headers at 320, 390, and 1440px, including actual center-point hit testing after scrolling focus.
- Conventional controls across the workbench and sixteen drawer entries at 320 and 1440px.
- Saved-chat layout and colors in both themes; card geometry remains unchanged when opening a popup.
- Popover enter/exit, rapid close/reopen, replacement with another token, focus restoration, nested Escape, and reduced motion.
- Real hosted-root navigation between setup and saved chats, including first-load suppression and interrupted transitions.
- Expanded interface audit in light and dark: empty/generated chat, all five Loom views, four instrument families, model controls, seventeen drawer entries, three authoring modes, and the template editor. Each surface receives layout, typography, contrast, and automated accessibility checks.
- Native authoring forms are rendered explicitly in the isolated fixture. This checks their UI, not fitting or storage-backed inference.
- Both sidebar directions, mid-transition reversal, empty selection, docking/undocking, token changes, simultaneous modal tools, focus restoration, reduced motion, and 320–1440px layouts.
- Additional Chromium/WebKit interaction checks confirmed that closing or undocking restores header focus and that sending a message at 320px keeps the inspector pinned. Long Loom cards grow and shrink with their full content; no internal-scroll cap was reintroduced.

Primary regressions: `e2e/interface-polish.spec.ts`, `e2e/workspace-consistency.spec.ts`, and the expanded-surface checks in `e2e/ui-skill-pass.spec.ts`.

## Boundaries

Tests use isolated local origins and deterministic runtime fixtures. The user's active model and saved chats are not reset or reloaded. No engine, numerical method, saved-data format, model capability policy, version, commit, push, or deployment is changed.

Headless iPhone-profile WebKit is not a physical iPhone or a VoiceOver session. GPU inference, native server authentication, real storage eviction, OS dialogs, and every possible measurement/error state are outside this UI verification.
