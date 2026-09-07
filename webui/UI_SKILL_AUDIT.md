# Drowse interface audit

Order for each surface: **better-ui → better-colors → better-layout → better-writing → better-typography → better-accessibility**.

This is an interface audit, not an inference-quality certification. Model kernels, sampling algorithms, measurement units, saved-chat formats, and artifact verification are unchanged. Existing unrelated work in this checkout is preserved.

## Review decisions

| Surface | UI | Colors | Layout | Writing | Typography | Accessibility |
|---|---|---|---|---|---|---|
| Landing and animated background | Added a visible pause control | Kept theme-specific washes and brand accent | Kept the existing responsive composition | Explicit animation-control label | Preserved the established heading and reading roles | Pause and live reduced-motion preference changes stop motion |
| Model selection, compatibility, downloads and storage | Preserved capability-based decisions and progressive detail | Checked existing status contrast in both themes | Included phone setup and model-control reflow checks | Preserved actionable errors and storage caveats | Kept one shared type scale | Existing labels, disclosures and status announcements retained |
| Saved chats, save/open drawers | Preserved named saves, avatars and explicit deletion | Existing neutral cards and semantic danger treatment retained | Included populated library and confirmation layouts | Full chat names remain available | Added bidi-aware names and full-name titles | Existing confirmation and focus behavior retained |
| Workbench shell and notifications | Update notices now reserve their own space | Opaque drawer and notification backgrounds; passive dismissal stays neutral | Shell height accounts for the notice; no composer overlap | Offline wording distinguishes the interface from installed models | Notifications use reading text rather than monospace | Dismissal tested; forced-color selections have a non-color cue |
| Chat, composer, roles and sampling | Preserved direct editing and one reply action row | Slider tracks use the contrast-tested data-track token | Existing compact/expanded roles, resizable composer and short-screen checks retained | Custom-role entries are described as values, not a separate action | Open options wrap instead of truncating | Dropdown Escape no longer closes its parent drawer |
| Loom: Weave, Map, Current path, Next options, Starred | Weave choices no longer all compete as primary actions | Preserved meaningful selection and branch states | Larger Current path token hit areas; phone and landscape tests | Help distinguishes Weave from Map | Weave body uses the reading size; path tokens use reading weight | Token targets enlarged; existing pinch/pan and keyboard controls retained |
| Response controls and instrument racks | Kept Shape and Watch separate | Preserved instrument-family colors and semantic data scales | Shared control fixes apply to every rack | Measurement provenance and units retained | Shared reading/structure/data roles retained | Existing control names and selected states retained |
| Token detail drawer: geometry, logits, SAE and lens | Preserved captured/replayed distinction | No artificial recoloring of scientific values | Evidence chips wrap; existing scrollable grids retained | Missing measurements remain missing, not zero | Evidence and numeric roles remain distinct | Existing token navigation and named reading controls retained |
| Help and technical references | Kept reference material behind disclosures | Help title is neutral, not action-colored | Long grammar remains independently scrollable | Updated Loom explanation for the current default view | Prose and code keep separate roles | Grammar region can receive keyboard focus |
| Manifold authoring: linear, template and custom | Kept the three modes and progressive fields | Kept semantic error states and primary build action | Fixed clipped names, squeezed axis values and overflowing node rows | Advanced disclosure says “Advanced options” | Input values remain readable without changing notation | Inputs and mode/node controls have usable targets; validation focus retained |
| Template lab: scoring, editor and catalog | Saved-template deletion now requires confirmation | Destructive action remains distinct | Catalog and confirmation wrap on phones | Confirmation names the template and says deletion cannot be undone | Long identifiers wrap | Cancel preserves the template; confirmed deletion is tested |
| Packs, merging and comparisons | Preserved import verification and local/external distinctions | Existing scientific comparison colors retained | Shared button wrapping applies to long actions | Replacement/deletion copy names the consequence and backup path | Existing data typography retained | Existing dialogs, input labels and cancellation retained |
| Native model health | Status is stated in words as well as color | No color-only pass/fail interpretation | Cards and long model names reflow | Expanded “ppl” to “Perplexity”; explicit loaded/unavailable/not-measured states | Existing numeric formatting retained | Textual statuses are available to assistive technology |
| Native API access | Explicit Apply key / Clear key / Refresh sessions controls | Neutral identifiers; semantic error color | Key controls and session headers wrap | Tab-only persistence explained accurately | Long session/model IDs wrap | Visible key label, form submission and persistent status feedback |
| Unavailable-runtime and 404 pages | Static headings do not look like controls | Existing theme tokens retained | Existing bounded layout retained | Plain “Page not found”; direct link to models/chats | Shared title and reading roles | Clear heading and destination label |
| Shared buttons, selects, comboboxes, help tips and tooltips | Popup and parent dismissal are separate | Opaque help surfaces; existing semantic tokens | Top-layer help; viewport-clamped popups; visible option wrapping | Full labels and contextual explanations remain available | Multiline labels have readable leading | Touch first-tap, hover persistence, Escape, listbox linkage and focus tested |
| Shared charts, marks, cards, radios, checkboxes and disclosures | Retained existing interaction conventions | Existing contrast-tested palettes retained | Checked consumers rather than changing scientific geometry globally | Kept real units and missing-data labels | Existing numeric roles retained | Numeric/text equivalents, native inputs and disclosure states retained |

## Evidence and limits

- Combined regression run: **75 passed, 2 explicitly skipped** across Chromium and iPhone-profile WebKit. This includes the interface, accessibility, phone/touch, Weave, authoring-validation and six-skill suites, plus the 404 and offline-notice regressions.
- After the final notice-height lifecycle adjustment, all **3 focused offline-notice tests passed again**, including a frame-by-frame check that dismissal does not push the workbench outside the viewport.
- `npm run check`: **0 Svelte errors, 0 warnings**; theme, runtime-boundary, entry, contrast and interface-policy checks passed. Native and hosted production builds passed; the existing large-chunk advisory remains.
- `npm run check` checks Svelte/TypeScript, declared theme tokens, runtime boundaries, entry behavior, semantic color contrast and interface policy across source files.
- `e2e/ui-skill-pass.spec.ts` runs six ordered checks per expanded surface in light and dark themes, at narrow and wide widths. It also checks touch help, complete dropdown labels, animation pause, reserved notice space and template deletion. Writing snapshots are attached to the test result; collecting copy is not a substitute for editorial review.
- `e2e/interface-audit.spec.ts` checks hierarchy, type roles, corners, RTL, primary flows, role states, drawer containment and 320px controls.
- `e2e/accessibility.spec.ts` covers the landing/setup/workbench, available command drawers, context drawers and populated saved chats with automated WCAG checks.
- `e2e/ios-layout.spec.ts` covers phone/landscape layout, keyboard-short viewports, Loom pinch/pan/cancel, geometry controls, instrument controls and saved-chat/token drawers.
- Authoring forms are explicitly rendered in the deterministic test fixture, which normally disables fitting. These are UI tests, not fitting-job tests.
- Chromium uses the storage-backed worker fixture. WebKit uses the in-memory workbench fixture for the expanded-surface audit because its temporary headless profile rejects OPFS. Storage-backed reload and template-deletion tests are explicitly skipped there, not reported as passes.
- The native API form is rendered through a test-only replacement of the hosted placeholder. This checks its layout, labels and contrast, not native server authentication.
- Automated WebKit with iPhone emulation is not a physical iPhone or VoiceOver test. Native server authentication, every possible real-model measurement, OS permission dialogs, actual storage eviction and screen-reader usability require separate live checks. No claim of universal bug-free behavior is made.

## Complete Svelte surface inventory

Inventory entries identify the scope, including shared components and runtime-specific replacements. Inclusion here means source coverage; it does **not** mean every possible data/error state was visually inspected. Rendered coverage is recorded by the tests above.

### Shell, hosted pages and runtime replacements

`App`; `HeroShader`; `HostedApp`; `HostedHome`; `HostedRoot`; `Landing`; `LandingRoot`; `ModelProviderLogo`; `NotFound`; `PwaUpdatePrompt`; hosted `JLensMissingState`, `JLensSourceSection`, `SaeSourceSection`.

### Drawers

`AdvancedSamplingDrawer`; `CastDrawer`; `CompareDrawer`; `CorrelationDrawer`; `HealthDrawer`; `HelpDrawer`; `LoadConversationDrawer`; `LocalRuntimeDrawer`; `ManifoldBuilderDrawer`; `ManifoldMergeDrawer`; `ManifoldPacksDrawer`; `NodeCompareDrawer`; `ProbeInspectorDrawer`; `RackDrawer`; `SaveConversationDrawer`; `SessionAdminDrawer`; `SystemPromptDrawer`; `TemplateLabDrawer`; `TokenDrilldownDrawer`; `TranscriptDrawer`; `UnavailableRuntimeDrawer`.

### Authoring and token-detail components

`AuthoredForm`; `DiscoverForm`; `DiscoverTuningFields`; `FitMethodPicker`; `TemplatedForm`; `DetailCardHeader`; `DetailSection`; `EmptyState`; `EvidenceChips`; `GeometryTab`; `InstrumentHeader`; token `JLensMissingState`; `LensTab`; `LogitsTab`; `PinnedReadings`; `SaeTab`; `TokenRibbon`.

### Shared primitives

`Checkbox`; `Combobox`; `Disclosure`; `NumberInput`; `Radio`; `Select`; `Slider`; `Toaster`; `AdvancedSection`; `ModeTabs`; `ValidationBlock`; `Bar`; `HeatmapCell`; `Sparkline`; `DiagnosticsPanel`; `Button`; `Chip`; `DrawerCloseButton`; `FluentIcon`; `InfoTip`; `SegmentedTabs`; `ThemeToggle`.

### Workbench and Loom panels

`Chat`; `CommandPalette`; `ControlsPanel`; `InspectorPanel`; `JLensPanel`; native `JLensSourceSection`; `PendingBubbles`; `ProbeRack`; `RawBuffer`; `RecipeBar`; `SaePanel`; native `SaeSourceSection`; `SamplingStrip`; `StatusFooter`; `SteeringRack`; `WorkbenchCard`; `LoomNode`; `LoomSidebar`; `LoomWeave`; `ManifoldMiniMap`; `XYPad`.

### Rack components

`AtomSteerCard`; `InstrumentSourceSection`; `JLensProbeCard`; `LayerStrip`; `ProbeCard`; `ProbeHighlightButton`; `ProbePinButton`; `ProbeReadingRow`; `RackCard`; `RackMarker`; `RackSectionHeader`; `SaeProbeCard`; `SteerCard`.
