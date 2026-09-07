<script lang="ts">
  import { tokenInspectorUi } from "../lib/stores/drawers.svelte";
  import FluentIcon from "../lib/ui/FluentIcon.svelte";
  import StateIcon from "../lib/ui/StateIcon.svelte";
  import { Blobatar } from "@blobatar/svelte";
  import { savedConversationState } from "../lib/stores/savedConversations.svelte";
  let { headersVisible = true }: { headersVisible?: boolean } = $props();
  // Chat panel: collapsible thinking per turn, per-token tinted spans,
  // optional compare-two stripe overlay, click-token drilldown, send /
  // stop, and an A/B split-view container.
  //
  // Single source of truth for state lives in ``lib/stores.svelte`` —
  // this file is presentation + local-only UI bits (textarea state,
  // scroll bookkeeping, per-turn collapse state).  The WS lifecycle and
  // gen-status accounting belongs to the store.
  //
  // Turn surface follows the cast model:
  // every speaker gets one neutral card, with the chat's blob avatar for
  // the model and an accent marker for the user. Role labels remain
  // arbitrary strings. Analysis badges appear only when
  // their backing artifacts exist. System turns render as stage directions (a
  // note about the scene, not a speaker). The ``speaking as`` chips in
  // the composer are the promoted SamplingStrip role boxes — same client
  // state, same wire block.
  //
  // Token rhythm: strip leading whitespace after </think>, with plain-text
  // fall-through when no probe is selected.

  import { onMount, tick, untrack } from "svelte";
  import { SvelteMap } from "svelte/reactivity";
  import { fly, slide } from "svelte/transition";
  import StatusFooter from "./StatusFooter.svelte";
  import PendingBubbles from "./PendingBubbles.svelte";
  import RawBuffer from "./RawBuffer.svelte";
  import TokenLogitsPopover from "../lib/ui/TokenLogitsPopover.svelte";
  import {
    collapseIn,
    collapseOut,
    contentIn,
    contentOut,
    motionDuration,
  } from "../lib/motion";
  import {
    autoRegenState,
    abState,
    chatLog,
    highlightState,
    loomTree,
    pinnedComparison,
    setHighlightTarget,
    setCompareTarget,
    toggleCompareTwo,
    setCompareTwo,
    unpinComparison,
    probeRack,
    sendSubmit,
    sendStop,
    genStatus,
    openDrawer,
    inputHistory,
    inputRestore,
    pushInputHistory,
    navigateInputHistory,
    cancelInputPull,
    consumePulledSlot,
    loomRegenerateNode,
    enqueuePending,
    pendingActions,
    cancelPendingAction,
    isPendingBusy,
    toggleAutoRegen,
    disableAutoRegen,
    setAutoRegenMode,
    setAutoRegenCustom,
    effectiveRawMode,
    genUiMode,
    roleDisplayLabel,
    castState,
    samplingState,
    sessionState,
    clearChat,
    beginTokenHover,
    endTokenHover,
  } from "../lib/stores.svelte";
  import type { AutoRegenMode } from "../lib/stores.svelte";
  import type { ChatRole, ChatTurn, TokenScore } from "../lib/types";
  import {
    ENTROPY_TARGET,
    SURPRISE_TARGET,
  } from "../lib/tokens";
  import {
    highlightStyleString,
  } from "../lib/highlight";
  import Select from "../lib/Select.svelte";
  import Combobox from "../lib/Combobox.svelte";
  import Checkbox from "../lib/Checkbox.svelte";
  import Button from "../lib/ui/Button.svelte";
  import InfoTip from "../lib/ui/InfoTip.svelte";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";

  function startsCompact(): boolean {
    return typeof window !== "undefined" && window.matchMedia(
      "(max-width: 720px), (max-height: 600px), (pointer: coarse) and (max-width: 1120px)",
    ).matches;
  }

  // --------------------------------------------------------------- input --

  let input = $state("");
  let chatRef: HTMLDivElement | null = $state(null);
  let textareaRef: HTMLTextAreaElement | null = $state(null);
  let clearConversationArmed = $state(false);
  let rolePlanOpen = $state(false);
  let rolePlanShell: HTMLDivElement | null = $state(null);
  let rolePlanAnimation: Animation | null = null;
  let composerHeight: number | null = $state(null);
  let composerMaxHeight = $state(420);
  let composerManuallySized = $state(false);
  let resizeStart = $state<{
    pointerId: number;
    y: number;
    height: number;
  } | null>(null);
  const COMPOSER_MIN_HEIGHT = 64;
  const COMPOSER_DEFAULT_HEIGHT = 80;
  const COMPACT_COMPOSER_MIN_HEIGHT = 64;
  const COMPACT_COMPOSER_DEFAULT_HEIGHT = 72;
  let composerMinHeight = $state(
    startsCompact() ? COMPACT_COMPOSER_MIN_HEIGHT : COMPOSER_MIN_HEIGHT,
  );
  let composerDefaultHeight = $state(
    startsCompact() ? COMPACT_COMPOSER_DEFAULT_HEIGHT : COMPOSER_DEFAULT_HEIGHT,
  );
  const composerControlHeight = $derived(Math.round(Math.max(
    composerMinHeight,
    Math.min(composerMaxHeight, composerHeight ?? composerDefaultHeight),
  )));
  const canClearConversation = $derived(
    loomTree.root_id !== null && loomTree.active_node_id !== loomTree.root_id,
  );

  function requestClearConversation(): void {
    if (!clearConversationArmed) {
      clearConversationArmed = true;
      return;
    }
    clearConversationArmed = false;
    clearChat();
  }

  /** Auto-grow the textarea from its comfortable default through 6 rows.
   *  With ``box-sizing: border-box`` set in CSS, ``el.scrollHeight``
   *  includes top/bottom padding — which is exactly what we want to write
   *  back into ``style.height``, so a one-line draft sits flush with no
   *  residual scrollbar.  The vertical scrollbar is suppressed unless the
   *  content actually overflows the 6-row cap. */
  function autosize(): void {
    const el = textareaRef;
    if (!el) return;
    if (composerManuallySized && composerHeight !== null) {
      el.style.height = `${composerHeight}px`;
      el.style.overflowY = el.scrollHeight > el.clientHeight ? "auto" : "hidden";
      return;
    }
    el.style.height = "auto";
    const rowHeight = 22; // mono line-height fudge — matches font-size-base
    const maxH = Math.min(rowHeight * 6, composerMaxHeight);
    const next = Math.min(Math.max(el.scrollHeight, composerDefaultHeight), maxH);
    el.style.height = `${next}px`;
    // Only show the scrollbar once we've actually hit the cap.  Without
    // this the browser's "always reserve a scrollbar gutter" heuristic
    // paints a 1-2px up/down nub on single-line input.
    el.style.overflowY = el.scrollHeight > maxH ? "auto" : "hidden";
  }

  function updateComposerBounds(): void {
    if (!chatRef) return;
    const compact = chatRef.clientHeight < 560 || chatRef.clientWidth <= 620 ||
      (window.visualViewport?.height ?? window.innerHeight) < 600;
    composerMinHeight = compact ? COMPACT_COMPOSER_MIN_HEIGHT : COMPOSER_MIN_HEIGHT;
    composerDefaultHeight = compact
      ? COMPACT_COMPOSER_DEFAULT_HEIGHT
      : COMPOSER_DEFAULT_HEIGHT;
    const reservedHeight = compact ? 190 : 300;
    composerMaxHeight = Math.max(
      composerMinHeight,
      Math.min(480, Math.floor(chatRef.clientHeight - reservedHeight)),
    );
    if (composerHeight !== null && composerHeight > composerMaxHeight) {
      composerHeight = composerMaxHeight;
    }
    queueMicrotask(autosize);
  }

  function clampComposerHeight(value: number): number {
    return Math.max(composerMinHeight, Math.min(composerMaxHeight, value));
  }

  function resizeComposerTo(value: number): void {
    composerManuallySized = true;
    composerHeight = clampComposerHeight(value);
    queueMicrotask(autosize);
  }

  function beginComposerResize(event: PointerEvent): void {
    if (event.button !== 0 || !textareaRef) return;
    const target = event.currentTarget as HTMLElement;
    updateComposerBounds();
    resizeStart = {
      pointerId: event.pointerId,
      y: event.clientY,
      height: textareaRef.getBoundingClientRect().height,
    };
    target.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function moveComposerResize(event: PointerEvent): void {
    if (!resizeStart || resizeStart.pointerId !== event.pointerId) return;
    resizeComposerTo(resizeStart.height + resizeStart.y - event.clientY);
  }

  function endComposerResize(event: PointerEvent): void {
    if (!resizeStart || resizeStart.pointerId !== event.pointerId) return;
    const target = event.currentTarget as HTMLElement;
    if (target.hasPointerCapture(event.pointerId)) {
      target.releasePointerCapture(event.pointerId);
    }
    resizeStart = null;
  }

  function resizeComposerWithKeyboard(event: KeyboardEvent): void {
    const current = textareaRef?.getBoundingClientRect().height
      ?? composerHeight
      ?? composerDefaultHeight;
    let next: number | null = null;
    if (event.key === "ArrowUp") next = current + 24;
    else if (event.key === "ArrowDown") next = current - 24;
    else if (event.key === "Home") next = composerMinHeight;
    else if (event.key === "End") next = composerMaxHeight;
    if (next === null) return;
    event.preventDefault();
    resizeComposerTo(next);
  }

  function resetComposerHeight(): void {
    composerManuallySized = false;
    composerHeight = null;
    queueMicrotask(autosize);
  }

  $effect(() => {
    // Run autosize whenever ``input`` changes — bind:value + an effect is
    // simpler than wiring an oninput handler that has to cooperate with
    // bind:.
    void input;
    autosize();
  });

  // --- Unified submission -----------------------------------------------
  // The composer exposes the native submit contract directly: one role for
  // an authored line, plus an independently selected continuation role (or
  // none). This keeps arbitrary scene sequences visible and makes same-role
  // prefill a single send: author a prefix and continue the same role.
  const rawMode = $derived.by(() => {
    void genUiMode.mode;
    return effectiveRawMode();
  });

  const activeNodeId = $derived(
    loomTree.loaded ? (loomTree.active_node_id ?? null) : null,
  );
  const sceneMode = $derived(sessionState.info?.scene_mode ?? false);
  type ContinuationRole = ChatRole | "none";
  const ROLE_SLUG_RE = /^[a-z0-9._-]+$/;

  let authoredRole = $state<ChatRole>("user");
  let continuationRole = $state<ContinuationRole>("assistant");
  $effect(() => {
    // Legacy renderers only support the ordinary user → assistant path. Keep
    // append-only available, but collapse unsupported role choices when a
    // model does not expose the scene stitcher.
    if (!sceneMode) {
      if (authoredRole !== "user") authoredRole = "user";
      if (continuationRole === "user") continuationRole = "assistant";
    }
  });
  const generatedRole = $derived<ChatRole | null>(
    continuationRole === "none" ? null : continuationRole,
  );
  const defaultUserLabel = $derived(
    sessionState.info?.default_user_role || "user",
  );
  const defaultAssistantLabel = $derived(
    sessionState.info?.default_assistant_role || "assistant",
  );
  const userLabel = $derived(
    samplingState.user_role.trim()
      || defaultUserLabel,
  );
  const assistantLabel = $derived(
    samplingState.assistant_role.trim()
      || defaultAssistantLabel,
  );
  const authoredLabel = $derived(authoredRole === "user" ? userLabel : assistantLabel);
  const authoredRoleValue = $derived(
    authoredRole === "user" ? samplingState.user_role : samplingState.assistant_role,
  );
  const continuationRoleValue = $derived(
    continuationRole === "user" ? samplingState.user_role : samplingState.assistant_role,
  );
  const userRoleSupported = $derived(
    sessionState.info?.is_base_model === false
      && sessionState.info?.user_role_supported === true,
  );
  const assistantRoleSupported = $derived(
    sessionState.info?.is_base_model === false
      && sessionState.info?.role_substitution_supported === true,
  );
  const customRoleLabels = $derived.by(() => {
    const labels = new Set(
      Object.keys(castState.roster).filter((key) => (
        key !== "user"
        && key !== "assistant"
        && key !== defaultUserLabel
        && key !== defaultAssistantLabel
      )),
    );
    if (userLabel !== defaultUserLabel && userLabel !== defaultAssistantLabel) {
      labels.add(userLabel);
    }
    if (
      assistantLabel !== defaultUserLabel
      && assistantLabel !== defaultAssistantLabel
    ) {
      labels.add(assistantLabel);
    }
    return [...labels].sort((a, b) => a.localeCompare(b));
  });

  const roleLabelOptions = $derived(
    [...new Set([defaultUserLabel, defaultAssistantLabel, ...customRoleLabels])]
      .map((value) => ({ value, label: value })),
  );

  function roleLabelValid(value: string): boolean {
    const label = value.trim();
    return label === "" || ROLE_SLUG_RE.test(label);
  }

  const authoredRoleValid = $derived(roleLabelValid(authoredRoleValue));
  const continuationRoleValid = $derived(
    continuationRole === "none" || roleLabelValid(continuationRoleValue),
  );
  const rolesValid = $derived(authoredRoleValid && continuationRoleValid);
  const rolePlanSummary = $derived(
    generatedRole === null
      ? `${authoredLabel} only`
      : `${authoredLabel} → ${generatedRole === "user" ? userLabel : assistantLabel}`,
  );

  $effect(() => {
    if (!rolesValid) rolePlanOpen = true;
  });

  async function setRolePlanOpen(open: boolean): Promise<void> {
    if (rolePlanOpen === open) return;
    const shell = rolePlanShell;
    const startHeight = shell?.getBoundingClientRect().height ?? 0;
    const restoreFocus = Boolean(shell?.contains(document.activeElement));
    rolePlanAnimation?.cancel();
    rolePlanAnimation = null;
    shell?.style.removeProperty("overflow");
    rolePlanOpen = open;
    await tick();
    if (!shell) return;
    if (restoreFocus) {
      shell.querySelector<HTMLElement>(open ? ".plan-minimize" : ".turn-plan-summary")
        ?.focus({ preventScroll: true });
    }
    const duration = motionDuration(220);
    if (duration === 0) return;
    const endHeight = shell.getBoundingClientRect().height;
    const easing = getComputedStyle(shell).getPropertyValue("--ease-move").trim()
      || "ease-in-out";
    shell.style.overflow = "hidden";
    const animation = shell.animate(
      [{ height: `${startHeight}px` }, { height: `${endHeight}px` }],
      { duration, easing },
    );
    rolePlanAnimation = animation;
    const finish = () => {
      if (rolePlanAnimation !== animation) return;
      rolePlanAnimation = null;
      shell.style.removeProperty("overflow");
    };
    animation.onfinish = finish;
    animation.oncancel = finish;
  }

  function roleSupportsCustomLabel(role: ChatRole): boolean {
    return role === "user" ? userRoleSupported : assistantRoleSupported;
  }
  const continuationRoleEditable = $derived(
    continuationRole !== "none" && roleSupportsCustomLabel(continuationRole),
  );

  function applyRoleChoice(role: ChatRole, label: string): void {
    if (role === "user") samplingState.user_role = label;
    else samplingState.assistant_role = label;
  }

  function selectAuthoredRole(label: string): void {
    applyRoleChoice(authoredRole, label);
  }

  function selectContinuationRole(label: string): void {
    if (continuationRole === "none") return;
    applyRoleChoice(continuationRole, label);
  }

  function setReplyEnabled(enabled: boolean): void {
    if (!enabled) {
      continuationRole = "none";
      return;
    }
    continuationRole = sceneMode && authoredRole === "assistant" ? "user" : "assistant";
  }
  const canSwapPlan = $derived(
    sceneMode && generatedRole !== null && generatedRole !== authoredRole,
  );

  function swapPlanRoles(): void {
    if (!canSwapPlan || generatedRole === null) return;
    const previousAuthored = authoredRole;
    authoredRole = generatedRole;
    continuationRole = previousAuthored;
  }

  // Authored-thinking input: a block the next authored line carries,
  // rendered through the family think delimiters.  Strip families keep
  // it for one turn only — the warning under the box says so before
  // submit (a9 convention 3).
  const thinkingInputSupported = $derived(
    sessionState.info?.thinking_input_supported ?? false,
  );
  const stripsHistoryThinking = $derived(
    sessionState.info?.strips_history_thinking ?? false,
  );
  let thinkingOpen = $state(false);
  let thinkingDraft = $state("");

  const hasText = $derived(input.trim() !== "");
  const appendSelected = $derived(generatedRole === null);
  const primaryDisabled = $derived(
    !loomTree.loaded || !rolesValid || (!hasText && generatedRole === null),
  );

  const inputPlaceholder = $derived(`Write as ${authoredLabel}…`);
  const sendLabel = $derived(
    appendSelected ? "Add message" : hasText ? "Send" : "Generate reply",
  );

  function doSend(): void {
    const text = hasText ? input : "";
    const replaceSlot = consumePulledSlot();
    if (!text) {
      if (replaceSlot !== null) {
        cancelPendingAction(pendingActions.queue[replaceSlot]?.id ?? "");
        return;
      }
      if (generatedRole === null) return;
    }
    const parent = isPendingBusy()
      ? ("active@drain" as const)
      : activeNodeId;
    const thinking = text && thinkingDraft.trim() !== "" ? thinkingDraft : null;
    if (thinking !== null) {
      thinkingDraft = "";
      thinkingOpen = false;
    }
    if (text) {
      pushInputHistory(text);
      input = "";
    }
    void sendSubmit(
      text || null,
      text ? authoredRole : null,
      generatedRole,
      {
        parent_node_id: parent,
        replaceSlot,
        raw: rawMode,
        authored_thinking: thinking,
      },
    );
    scrolledUp = false;
    queueScrollToBottom();
    queueMicrotask(autosize);
  }

  /** Edge-only multi-line policy: ↑ recalls history when the cursor
   *  sits on the first line of the draft; ↓ goes forward only on the
   *  last line.  In-between lines fall through to the textarea's
   *  native cursor nav so multi-line editing isn't hijacked. */
  function shouldRecallUp(ta: HTMLTextAreaElement): boolean {
    const value = ta.value;
    const cursor = ta.selectionStart ?? 0;
    const firstNL = value.indexOf("\n");
    return firstNL === -1 || cursor <= firstNL;
  }

  function shouldRecallDown(ta: HTMLTextAreaElement): boolean {
    const value = ta.value;
    const cursorEnd = ta.selectionEnd ?? value.length;
    const lastNL = value.lastIndexOf("\n");
    return lastNL === -1 || cursorEnd > lastNL;
  }

  function applyRecalled(text: string): void {
    input = text;
    // Defer cursor placement past the bind:value flush so the textarea
    // reflects the new value before we set the selection.
    queueMicrotask(() => {
      const el = textareaRef;
      if (el) {
        el.setSelectionRange(el.value.length, el.value.length);
        autosize();
      }
    });
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Enter") {
      // Shift-Enter is a newline; Enter executes the visible plan.  Append
      // is selected explicitly by choosing no continuation.
      if (ev.shiftKey) return;
      ev.preventDefault();
      doSend();
      return;
    }
    if (ev.key === "Escape") {
      // Esc is context-sensitive:
      //   1. Gen in flight → stop the gen.  Queue keeps its items.
      //   2. No gen, pull in flight → cancel the pull (restore the
      //      stash, leave the queued slot untouched).
      //   3. Otherwise → fall through to default Escape behavior.
      if (genStatus.active) {
        ev.preventDefault();
        sendStop();
        return;
      }
      if (inputHistory.pulledSlot !== null) {
        ev.preventDefault();
        const stash = cancelInputPull();
        if (stash !== null) {
          input = stash;
          queueMicrotask(autosize);
        }
        return;
      }
    }
    if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
      const ta = textareaRef;
      if (!ta) return;
      const goingUp = ev.key === "ArrowUp";
      if (goingUp ? !shouldRecallUp(ta) : !shouldRecallDown(ta)) return;
      // ↓ at the live slot (no recall in flight, no pending pull) is
      // a no-op — leave the keystroke for the textarea so it can move
      // within an empty last line or trigger the browser's native
      // end-of-input nudge.  Checking only ``index`` misses the
      // pulled-pending case (``pulledSlot`` is what tracks that),
      // which would otherwise strand the user inside an edited
      // pending item with no ↓ exit.
      if (
        !goingUp
        && inputHistory.index === null
        && inputHistory.pulledSlot === null
      ) return;
      const recalled = navigateInputHistory(goingUp ? -1 : +1, input);
      if (recalled === null) return;
      ev.preventDefault();
      applyRecalled(recalled);
    }
  }

  // ------------------------------------------------------------- highlight --

  /** All probe names available to the highlight dropdowns.  Sourced from
   * the live probe-rack — same source the ProbeRack panel uses. */
  const probeNames = $derived([...probeRack.active]);

  function onHighlightChange(value: string): void {
    setHighlightTarget(value === "" ? null : value);
  }

  function onCompareChange(value: string): void {
    setCompareTarget(value === "" ? null : value);
  }

  function onCompareToggle(): void {
    if (!highlightState.compareTwo) {
      const first = compareOptions[0]?.value ?? null;
      if (first === null) return;
      setCompareTarget(first);
    }
    toggleCompareTwo();
  }

  /** Highlight options for one probe.  A rank-1 flat probe (a 2-node concept
   *  axis) and every curved probe stay a single bare-name option — the bare
   *  channel is the pole coordinate / subspace fraction.  A multi-axis flat
   *  probe (the ``personas`` fan, a flat ``emotions``) fans out into one
   *  option per coordinate so a token can be tinted by each PC; axis 0 keeps
   *  the bare-name value (the channel that survives reload) while axis ``i``
   *  uses the ``name[i]`` form that lines up with the ``@when:name[i]`` gate. */
  function axisOptionsFor(name: string): { value: string; label: string }[] {
    const info = probeRack.entries.get(name)?.info;
    const dim = info?.intrinsic_dim ?? 0;
    const flat = info?.family === "geometry" ? info.is_affine : true;
    // Labels strip the namespace prefix (``default/emotions`` → ``emotions``)
    // to match the probe cards; the option value keeps the full registered
    // name so lookups stay unambiguous.
    const display = name.split("/").pop() ?? name;
    if (flat && dim > 1) {
      return Array.from({ length: dim }, (_, i) => ({
        value: i === 0 ? name : `${name}[${i}]`,
        label: `${display}[${i}]`,
      }));
    }
    return [{ value: name, label: display }];
  }

  /** Highlight-target picker options: built-in generation measurements plus
   *  live probe names, fanned out per coordinate axis for multi-axis probes. */
  const samplerEntropyAvailable = $derived(
    chatLog.turns.some((turn) => (
      [...(turn.thinkingTokens ?? []), ...(turn.tokens ?? [])]
        .some((token) => token.samplerEntropy != null && Number.isFinite(token.samplerEntropy))
    )),
  );
  const hasColorableTokens = $derived(
    chatLog.turns.some((turn) => (
      (turn.thinkingTokens?.length ?? 0) > 0 || (turn.tokens?.length ?? 0) > 0
    )),
  );

  const builtInHighlightOptions = $derived.by<{ value: string; label: string }[]>(() => [
    { value: SURPRISE_TARGET, label: "Token surprisal" },
    ...(samplerEntropyAvailable
      ? [{ value: ENTROPY_TARGET, label: "Sampler entropy" }]
      : []),
  ]);

  const highlightOptions = $derived.by<{ value: string; label: string }[]>(
    () => {
      const opts: { value: string; label: string }[] = [
        { value: "", label: "No color" },
        ...builtInHighlightOptions,
      ];
      for (const name of probeNames) opts.push(...axisOptionsFor(name));
      return opts;
    },
  );

  /** Compare-target picker — same shape but filtered so the A and B targets
   *  don't pick the same axis.  Distinct axes of one probe (PC0 vs PC1) are
   *  allowed — that's a useful two-stripe compare. */
  const compareOptions = $derived.by<{ value: string; label: string }[]>(() => {
    const opts = builtInHighlightOptions.filter(
      (option) => option.value !== highlightState.target,
    );
    for (const name of probeNames) {
      for (const opt of axisOptionsFor(name)) {
        if (opt.value !== highlightState.target) opts.push(opt);
      }
    }
    return opts;
  });

  const compareColorAvailable = $derived(
    hasColorableTokens && highlightState.target !== null && compareOptions.length > 0,
  );

  $effect(() => {
    if (!compareColorAvailable) {
      if (highlightState.compareTwo) setCompareTwo(false);
      if (highlightState.compareTarget !== null) setCompareTarget(null);
      return;
    }
    if (
      highlightState.compareTwo &&
      !compareOptions.some((option) => option.value === highlightState.compareTarget)
    ) {
      setCompareTarget(compareOptions[0]?.value ?? null);
    }
  });

  // -------------------------------------------------- conversation actions --
  //
  // clear / transcript / auto-regen used to live on the Topbar;
  // they act on the conversation, so they belong here.  The mutating
  // ones route through ``enqueuePending`` so clicking them mid-gen
  // queues rather than racing the WS.

  const AUTO_REGEN_MODES: { value: AutoRegenMode; label: string }[] = [
    { value: "unsteered", label: "Original behavior" },
    { value: "inverted", label: "Opposite guidance" },
    { value: "reseed", label: "New random seed" },
    { value: "cool", label: "More focused" },
    { value: "hot", label: "More varied" },
    { value: "custom", label: "Custom recipe…" },
  ];

  const generationAvailability = runtimeOperationAvailability("generation");
  const automaticComparisonAvailable = $derived(
    generationAvailability.available &&
    sessionState.info !== null &&
    sessionState.info.is_base_model !== true &&
    !rawMode,
  );

  $effect(() => {
    if (!automaticComparisonAvailable && autoRegenState.enabled) {
      disableAutoRegen();
    }
  });

  function regenMessage(turn: ChatTurn): void {
    const nodeId = turn.nodeId;
    if (!nodeId) return;
    if (isPendingBusy()) {
      enqueuePending({
        label: "regen",
        text: null,
        apply: () => void loomRegenerateNode(nodeId, 1),
        awaitsGen: true,
        rebuild: null,
        endsOnUserNode: turn.role === "user",
      });
    } else {
      void loomRegenerateNode(nodeId, 1);
    }
  }

  // Save / load act on the whole conversation tree; they live here at
  // the chat's edge rather than buried in a rail menu.  Regenerate-N and
  // fan-out used to sit here too — both were redundant (the loom right-
  // click menu carries "regenerate…" and "fan out…", and the experiment
  // lab is one click away in the analysis menu) so they were removed.
  function onAutoRegenModeChange(v: AutoRegenMode): void {
    setAutoRegenMode(v);
  }

  // ------------------------------------------------------------- A/B split --

  /** Do not open an empty comparison pane merely because the setting is on.
   * It appears when a shadow has started or a completed comparison exists. */
  const autoRegenActive = $derived(
    autoRegenState.enabled &&
    (abState.processingAb || chatLog.turns.some((turn) => turn.abPair !== undefined)),
  );

  /** A manually pinned Loom branch takes precedence over the automatic
   * comparison while it remains in the authoritative tree. */
  const pinnedActive = $derived(
    pinnedComparison.nodeId !== null &&
    loomTree.nodes.has(pinnedComparison.nodeId),
  );

  /** Render the conversation up to (and including) the pinned node by
   *  walking parent pointers from the pinned id back to root.  Skips
   *  the synthetic root.  Used by the right column when pinned. */
  const pinnedPath = $derived.by<ChatTurn[]>(() => {
    if (!pinnedActive || !pinnedComparison.nodeId) return [];
    const out: ChatTurn[] = [];
    let cursor: string | null = pinnedComparison.nodeId;
    const seen = new Set<string>();
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      const node = loomTree.nodes.get(cursor);
      if (!node) break;
      // Skip the synthetic root.
      if (!(node.parent_id === null && node.role === "system" && !node.text)) {
        out.push({
          role: node.role,
          text: node.text ?? "",
          roleLabel: node.role_label,
          nodeId: node.id,
          generated: node.recipe !== null,
          appliedSteering: node.applied_steering ?? null,
          aggregateReadings: node.aggregate_readings ?? undefined,
          finishReason: node.finish_reason ?? undefined,
        });
      }
      cursor = node.parent_id;
    }
    return out.reverse();
  });

  /** The right column is visible when EITHER auto-regen is on (which
   *  subsumes the v1.x A/B toggle) or a node is pinned for comparison. */
  const twoColumns = $derived(pinnedActive || autoRegenActive);

  // ----------------------------------------------------------- per-turn UI --

  /** Per-turn thinking-collapsed state.  Keyed by turn index so a re-
   * render of the chat log preserves user-explicit collapse choices.  We
   * default to "collapsed" on creation and auto-expand when the first
   * thinking token lands; on done the turn collapses again unless the
   * user manually expanded.
   *
   * SvelteMap (not plain Map) — Svelte 5's $state doesn't track plain
   * Map mutations, so a bare Map.set wouldn't re-render the toggle UI. */
  const collapsedThinking: SvelteMap<number, boolean> = $state(new SvelteMap());

  function turnCollapsed(turnIdx: number, turn: ChatTurn): boolean {
    const explicit = collapsedThinking.get(turnIdx);
    if (explicit !== undefined) return explicit;
    // Default: expanded while in-flight (so the user can watch it
    // generate), collapsed once the gen lands.
    const inFlight =
      chatLog.pendingIndex === turnIdx && (turn.thinkingTokens?.length ?? 0) > 0;
    return !inFlight;
  }

  function toggleThinking(turnIdx: number): void {
    const cur = collapsedThinking.get(turnIdx) ?? true;
    collapsedThinking.set(turnIdx, !cur);
  }

  // Keep the streamed transcript outside a live region: announcing every
  // token makes generation unusable with a screen reader. This stable,
  // atomic status reports only lifecycle transitions and completed
  // conversation updates.
  let streamAnnouncement = $state("");
  let announcementInitialized = false;
  let previousGenerationActive = false;
  let previousTurnCount = 0;
  $effect(() => {
    const active = genStatus.active;
    const finishReason = genStatus.finishReason;
    const pendingIndex = chatLog.pendingIndex;
    const turnCount = chatLog.turns.length;

    untrack(() => {
      if (!announcementInitialized) {
        announcementInitialized = true;
      } else if (active && !previousGenerationActive) {
        streamAnnouncement = rawMode ? "Updating completion." : "Generating response.";
      } else if (!active && previousGenerationActive) {
        streamAnnouncement = finishReason === "cancelled"
          ? "Generation stopped."
          : finishReason === null
            ? "Generation ended."
            : rawMode
              ? genStatus.tokensSoFar === 0 && chatLog.turns.at(-1)?.generated === false
                ? "Edit saved."
                : finishReason === "length" ? "Token limit reached." : "Completion finished."
              : "Response complete.";
      } else if (
        !active
        && pendingIndex === null
        && turnCount !== previousTurnCount
      ) {
        streamAnnouncement = rawMode
          ? "Completion updated."
          : `Conversation updated. ${turnCount} ${turnCount === 1 ? "message" : "messages"}.`;
      }

      previousGenerationActive = active;
      previousTurnCount = turnCount;
    });
  });

  // Cross-component input restore: when a queue drain pops the slot
  // the user was currently editing, ``drainNextPendingAction`` parks
  // the stash on ``inputRestore`` and bumps ``rev``.  This $effect
  // copies it back into the textarea on the next tick.
  let _restoreRev = $state(0);
  $effect(() => {
    if (inputRestore.rev !== _restoreRev) {
      _restoreRev = inputRestore.rev;
      input = inputRestore.text;
      queueMicrotask(autosize);
    }
  });

  // ------------------------------------------------------ scroll bookkeeping --

  let logRef: HTMLDivElement | null = $state(null);
  /** True iff the user has manually scrolled up — freezes auto-scroll
   * until they hit the bottom again. */
  let scrolledUp = $state(false);

  function onScroll(ev: Event): void {
    const el = ev.currentTarget as HTMLElement;
    // 8px slop so a scrollbar that doesn't quite hit the floor still
    // counts as "at bottom".
    const atBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 8;
    scrolledUp = !atBottom;
  }

  function scrollToBottom(): void {
    const el = logRef;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }

  let _scrollScheduled = false;
  function queueScrollToBottom(): void {
    if (_scrollScheduled) return;
    _scrollScheduled = true;
    queueMicrotask(() => {
      _scrollScheduled = false;
      if (!scrolledUp) scrollToBottom();
    });
  }

  // Auto-scroll on new turns or token deltas.  Reads (not writes) the
  // length-aggregates that drive the chat — Svelte 5 tracks these via
  // the runes graph.
  $effect(() => {
    // Touch the things we care about so the effect re-runs on changes.
    void chatLog.turns.length;
    const lastTurn = chatLog.turns[chatLog.turns.length - 1];
    void lastTurn?.tokens?.length;
    void lastTurn?.thinkingTokens?.length;
    void lastTurn?.text;
    untrack(() => queueScrollToBottom());
  });

  onMount(() => {
    autosize();
    scrollToBottom();
    if (!window.matchMedia("(pointer: coarse)").matches) textareaRef?.focus();
    updateComposerBounds();
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(updateComposerBounds);
    if (chatRef) observer?.observe(chatRef);
    window.visualViewport?.addEventListener("resize", updateComposerBounds);
    return () => {
      observer?.disconnect();
      window.visualViewport?.removeEventListener("resize", updateComposerBounds);
      rolePlanAnimation?.cancel();
    };
  });

  // ----------------------------------------------------------- token render --

  /** Drop whitespace-only tokens from the head of the response so the gap below ``</think>``
   * goes away in plain-text mode too.  Returns the surviving slice
   * starting at the first non-whitespace token. */
  interface VisibleToken {
    tok: TokenScore;
    originalIdx: number;
  }

  function visibleResponseTokens(tokens: TokenScore[]): VisibleToken[] {
    let i = 0;
    while (i < tokens.length && !tokens[i].text.trim()) i++;
    return tokens.slice(i).map((tok, offset) => ({
      tok,
      originalIdx: i + offset,
    }));
  }

  let tokenPopup = $state<{
    token: TokenScore; anchor: HTMLElement; turnIdx: number; tokenIdx: number; isThinking: boolean;
  } | null>(null);
  const tokenFocusIndices = new SvelteMap<string, number>();

  function tokenClicked(
    turnIdx: number,
    tokenIdx: number,
    ev: MouseEvent | KeyboardEvent,
    isThinking: boolean = false,
  ): void {
    ev.stopPropagation();
    const turn = chatLog.turns[turnIdx];
    const token = (isThinking ? turn?.thinkingTokens : turn?.tokens)?.[tokenIdx];
    if (!token) return;
    if (tokenInspectorUi.docked) {
      tokenPopup = null;
      openDrawer("token_drilldown", { turnIdx, tokenIdx, isThinking });
      return;
    }
    tokenPopup = { token, anchor: ev.currentTarget as HTMLElement, turnIdx, tokenIdx, isThinking };
  }

  function tokenKeydown(ev: KeyboardEvent, turnIdx: number, tokenIdx: number, isThinking = false): void {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      tokenClicked(turnIdx, tokenIdx, ev, isThinking);
      return;
    }
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(ev.key)) return;
    const target = ev.currentTarget as HTMLElement;
    const tokens = [...target.parentElement!.querySelectorAll<HTMLElement>(".tok")];
    const current = tokens.indexOf(target);
    const next = ev.key === "Home" ? 0 : ev.key === "End" ? tokens.length - 1
      : (current + (ev.key === "ArrowRight" ? 1 : -1) + tokens.length) % tokens.length;
    ev.preventDefault();
    tokens[next].focus();
  }

  /** One keyboard stop per message opens the drawer at a real token; the
   * drawer's arrow-key controls then traverse every token and segment. */
  function inspectTurnTokens(turn: ChatTurn, turnIdx: number): void {
    const firstResponse = visibleResponseTokens(turn.tokens ?? [])[0];
    if (firstResponse) {
      openDrawer("token_drilldown", {
        turnIdx,
        tokenIdx: firstResponse.originalIdx,
      });
      return;
    }
    if ((turn.thinkingTokens?.length ?? 0) > 0) {
      openDrawer("token_drilldown", {
        turnIdx,
        tokenIdx: 0,
        isThinking: true,
      });
      return;
    }
    if ((turn.tokens?.length ?? 0) > 0) {
      openDrawer("token_drilldown", { turnIdx, tokenIdx: 0 });
    }
  }

  // The bare-text form for plain (no-highlight) rendering still strips
  // leading whitespace when no probe is selected.
  function plainResponseText(turn: ChatTurn): string {
    if (!turn.tokens || turn.tokens.length === 0) {
      // Fall back to the accumulated text if the per-token list is
      // unset (extremely early in a stream, or for non-streamed loads).
      return (turn.text ?? "").replace(/^\s+/, "");
    }
    return visibleResponseTokens(turn.tokens)
      .map(({ tok }) => tok.text)
      .join("");
  }
</script>

<div class="chat-panel">
  {#if headersVisible}
  <header class="chat-header" id="chat-tools-header" inert={!headersVisible}
    in:slide={collapseIn()} out:slide={collapseOut()}>
    <div class="reading-tools">
      <div class="color-control">
        <label class="ctl">
          <span class="ctl-label">Color words by</span>
          <!-- Logit-pass: ``surprise`` tints tokens by ``-logprob /
               (1 - logprob)`` per Decision 4.  Sentinel value sits next to
               real probe names in the same picker so a single dropdown
               covers both axes. -->
          <span class="ctl-select">
            <Select
              value={highlightState.target ?? ""}
              options={highlightOptions}
              onchange={onHighlightChange}
              ariaLabel={rawMode ? "Color completion tokens by" : "Color generated words by"}
            />
          </span>
        </label>

        <InfoTip
          label="About word colors"
          text={rawMode
            ? "Colors show recorded readings in both editing and inspection views, and pause for unsaved edits. Open Inspect tokens for exact values. Surprisal and entropy use the sampling distribution; greedy decoding can record zero for both."
            : "Color words by token surprisal, sampler entropy, or an attached model reading. Select a word for its exact value."}
        />
      </div>

    {#if compareColorAvailable}
      <button
        type="button"
        class="compare-color"
        class:active={highlightState.compareTwo}
        aria-pressed={highlightState.compareTwo}
        onclick={onCompareToggle}
      >{highlightState.compareTwo ? "Use one color" : "Compare colors"}</button>
      {/if}

    {#if highlightState.compareTwo}
      <label class="ctl">
        <span class="ctl-label">With</span>
        <!-- Allow surprise as the B-stripe target too — "probe X vs.
             surprise" is a useful axis ("does probe X light up at the
             surprising tokens?"). -->
        <span class="ctl-select">
          <Select
            value={highlightState.compareTarget ?? ""}
            options={compareOptions}
            onchange={onCompareChange}
            disabled={!highlightState.compareTwo}
            ariaLabel="Second word color"
          />
        </span>
      </label>
      {/if}

    </div>

    <!-- Conversation actions. -->
    <div class="header-actions">
      {#if automaticComparisonAvailable}
        <div class="comparison-control">
          <span class="ctl ctl-inline">
            <Checkbox
              checked={autoRegenState.enabled}
              onchange={toggleAutoRegen}
              label="Compare replies"
            />
          </span>
          <InfoTip
            label="About automatic comparison"
            text="Create a second version after every reply using the comparison settings."
          />
        </div>
        {#if autoRegenState.enabled}
          <span class="ctl-select">
            <Select
              value={autoRegenState.mode}
              options={AUTO_REGEN_MODES}
              onchange={onAutoRegenModeChange}
              disabled={abState.processingAb}
              ariaLabel="Automatic comparison style"
            />
          </span>
          {#if autoRegenState.mode === "custom"}
            <input
              type="text"
              class="ctl-input"
              value={autoRegenState.custom}
              disabled={abState.processingAb}
              oninput={(ev) =>
                setAutoRegenCustom(
                  (ev.currentTarget as HTMLInputElement).value,
                )}
              placeholder="Example: seed=42, temperature=1.5"
              aria-label="Custom automatic comparison recipe"
            />
          {/if}
        {/if}
      {/if}
    </div>
  </header>
  {/if}

<div class="chat" aria-label={rawMode ? "Text completion" : "Chat"} bind:this={chatRef}>
  <div
    class="sr-only"
    role="status"
    aria-live="polite"
    aria-atomic="true"
  >{streamAnnouncement}</div>

  {#if rawMode}
    <RawBuffer />
  {:else}
  <div
    class="log"
    class:ab={twoColumns}
    bind:this={logRef}
    onscroll={onScroll}
    role="region"
    aria-label="Conversation"
  >
    {#if chatLog.turns.length === 0}
      <div
        class="conversation-empty"
        in:fly={contentIn(8)}
        out:fly={contentOut()}
      >
        <p>Type a prompt to get started</p>
      </div>
    {:else if twoColumns}
      <!-- Two-column split. The right column is a pinned branch path or a
           stateless automatic comparison. Comparisons never alter the loom. -->
      <div class="ab-grid">
        <div class="ab-col ab-primary">
          {#each chatLog.turns as turn, turnIdx (turnIdx)}
            {@render bubble(turn, turnIdx, false)}
          {/each}
        </div>
        <div class="ab-col ab-shadow">
          {#if pinnedActive}
            <header class="pin-header">
              <span class="pin-tag">pinned</span>
              <code class="pin-id">{pinnedComparison.nodeId?.slice(0, 12)}</code>
              <button
                type="button"
                class="pin-unpin"
                onclick={unpinComparison}
                title="Unpin"
              >unpin</button>
            </header>
            {#each pinnedPath as turn, idx (idx)}
              {@render bubble(turn, idx, true)}
            {/each}
          {:else}
            {#each chatLog.turns as turn, turnIdx (turnIdx)}
              {#if !turn.generated || turn.role === "system"}
                {@render bubble(turn, turnIdx, false)}
              {:else if turn.abPair}
                {@render bubble(turn.abPair, turnIdx, true)}
              {:else}
                <div class="msg placeholder" aria-hidden="true">
                  <div class="who">
                    {@render speaker(turn, false)}
                    <span class="who-meta">(alt)</span>
                  </div>
                  <span class="placeholder-text">
                    {abState.pendingTurnIdx === turnIdx ? "Generating comparison…" : "Not compared"}
                  </span>
                </div>
              {/if}
            {/each}
          {/if}
        </div>
      </div>
    {:else}
      {#each chatLog.turns as turn, turnIdx (turnIdx)}
        {@render bubble(turn, turnIdx, false)}
      {/each}
    {/if}
  </div>

  <div
    class="composer-resizer-shell"
    class:dragging={resizeStart !== null}
  >
    <input
      type="range"
      class="composer-resizer"
      min={composerMinHeight}
      max={composerMaxHeight}
      value={composerControlHeight}
      aria-label="Resize writing area"
      aria-orientation="vertical"
      aria-controls="conversation-composer"
      aria-valuetext={`${composerControlHeight} pixel writing area`}
      aria-describedby="composer-resize-help"
      onpointerdown={beginComposerResize}
      onpointermove={moveComposerResize}
      onpointerup={endComposerResize}
      onpointercancel={endComposerResize}
      onkeydown={resizeComposerWithKeyboard}
      ondblclick={resetComposerHeight}
    />
    <span aria-hidden="true"></span>
    <p class="sr-only" id="composer-resize-help">Use arrow keys to resize. Double-click to reset.</p>
  </div>

  <StatusFooter />

  <PendingBubbles />

  <div class="turn-plan-shell" bind:this={rolePlanShell}>
  {#if rolePlanOpen}
    <div
      id="turn-role-controls"
      class="turn-plan"
      aria-label="Next turn"
      in:fly={contentIn(2)}
    >
    <div class="plan-card">
      <span class="plan-actor">You</span>
      <Combobox
        value={authoredRoleValue}
        options={roleLabelOptions}
        onchange={selectAuthoredRole}
        disabled={!roleSupportsCustomLabel(authoredRole)}
        invalid={!authoredRoleValid}
        placeholder={authoredRole === "user" ? defaultUserLabel : defaultAssistantLabel}
        ariaLabel="You write as"
        ariaDescribedby={!authoredRoleValid ? "role-label-error" : undefined}
      />
    </div>

    <div class="plan-card">
      <span class="plan-actor">Model</span>
      <Combobox
        value={continuationRole === "none" ? "" : continuationRoleValue}
        options={roleLabelOptions}
        onchange={selectContinuationRole}
        disabled={!continuationRoleEditable}
        invalid={!continuationRoleValid}
        placeholder={continuationRole === "user" ? defaultUserLabel : defaultAssistantLabel}
        ariaLabel="Model writes as"
        ariaDescribedby={!continuationRoleValid ? "role-label-error" : undefined}
      />
    </div>

    <div class="plan-actions">
      <Button
        variant="flat"
        size="sm"
        disabled={!canSwapPlan}
        onclick={swapPlanRoles}
        ariaLabel="Swap writer roles"
      >Swap</Button>
      <label class="reply-toggle">
        <Checkbox
          checked={generatedRole !== null}
          onchange={setReplyEnabled}
          ariaLabel="Generate a model reply"
        />
        <span>Reply</span>
      </label>
      <Button
        size="sm"
        variant="flat"
        disabled={!rolesValid}
        onclick={() => openDrawer("cast")}
      >Role settings</Button>
      <button
        type="button"
        class="plan-minimize"
        aria-expanded="true"
        aria-controls="turn-role-controls"
        onclick={() => setRolePlanOpen(false)}
      >Collapse roles</button>
    </div>
    </div>
  {:else}
    <button
      type="button"
      class="turn-plan-summary"
      aria-expanded="false"
      aria-controls="turn-role-controls"
      onclick={() => setRolePlanOpen(true)}
      in:fly={contentIn(2)}
    >
      <span class="summary-label">Roles</span>
      <span class="summary-value">{rolePlanSummary}</span>
      <span class="summary-action">Edit roles</span>
    </button>
  {/if}
  </div>

  {#if !rolesValid}
    <p id="role-label-error" class="role-error" role="alert">
      Use lowercase letters, numbers, periods, underscores, or hyphens for role names.
    </p>
  {/if}

  {#if thinkingInputSupported}
    <div class="thinking-row">
      <div class="thinking-control">
        <Button
          variant="flat"
          size="sm"
          accent={thinkingDraft.trim() !== "" ? "var(--pillar-manifold)" : undefined}
          onclick={() => (thinkingOpen = !thinkingOpen)}
          title="Add private reasoning text to the next authored line"
        >{thinkingOpen ? "Hide reasoning" : "Add reasoning"}</Button>
      </div>
      {#if thinkingOpen}
        <div class="thinking-box" in:slide={collapseIn()} out:slide={collapseOut()}>
          <textarea
            class="thinking-input field-focus"
            bind:value={thinkingDraft}
            dir="auto"
            placeholder="Optional reasoning for this line…"
            rows="2"
            spellcheck="false"
            aria-label="Reasoning for the next authored line"
          ></textarea>
          {#if stripsHistoryThinking}
            <p class="thinking-warn" role="note">This model uses it for one turn only.</p>
          {/if}
        </div>
      {/if}
    </div>
  {/if}

  <form
    id="conversation-composer"
    class="input-row"
    onsubmit={(ev) => { ev.preventDefault(); doSend(); }}
  >
    <textarea
      class="input field-focus"
      bind:this={textareaRef}
      bind:value={input}
      dir="auto"
      onkeydown={onKeydown}
      placeholder={inputPlaceholder}
      rows="3"
      aria-label={`Compose as ${authoredLabel}`}
    ></textarea>
    <div class="input-actions" class:has-clear={canClearConversation}>
      <Button
        type="submit"
        variant="solid"
        disabled={primaryDisabled}
        title={appendSelected ? "Enter · add your message only" : "Enter · send and generate a reply"}
      ><StateIcon icons={["add", "send", "conversation"]} name={appendSelected ? "add" : hasText ? "send" : "conversation"} />{sendLabel}</Button>
      <Button
        variant="danger"
        onclick={sendStop}
        disabled={!genStatus.active}
        title="Escape · stop the current reply"
      ><FluentIcon name="stop" />Stop</Button>
      {#if canClearConversation}
        <button
          type="button"
          class="clear-conversation"
          class:confirm-clear={clearConversationArmed}
          onclick={requestClearConversation}
          onblur={() => (clearConversationArmed = false)}
          onkeydown={(event) => {
            if (event.key === "Escape") clearConversationArmed = false;
          }}
          title={clearConversationArmed
            ? "Clear the current view and start a new path"
            : "Start a blank conversation; existing branches remain available"}
          aria-label={clearConversationArmed ? "Confirm clear conversation" : "Clear conversation"}
        >
          {clearConversationArmed ? "Confirm clear" : "Clear conversation"}
        </button>
      {/if}
    </div>
  </form>
  {/if}
</div>
  {#if sessionState.info?.model_id}
    <div class="active-model" aria-label="Active model" title={sessionState.info.model_id}>
      <span>Model</span>
      <span class="active-model-name">{sessionState.info.model_id.split("/").at(-1)}</span>
    </div>
  {/if}

{#if tokenPopup}
  {@const popup = tokenPopup}
  {#key popup.anchor}
    <TokenLogitsPopover token={popup.token} anchor={popup.anchor}
      source={chatLog.turns[popup.turnIdx]?.generated ? "Model token" : "User text · recorded"}
      onclose={() => tokenPopup = null}
      ondetails={() => openDrawer("token_drilldown", { turnIdx: popup.turnIdx, tokenIdx: popup.tokenIdx, isThinking: popup.isThinking, initialTab: "logits" })} />
  {/key}
{/if}
</div>

{#snippet speaker(turn: ChatTurn, interactive = true)}
  <span class="role-chip">
    {#if turn.role === "assistant"}
      {#if interactive}
        <button type="button" class="model-avatar" aria-label="Open model settings" title="Model settings"
          onclick={() => window.dispatchEvent(new CustomEvent("drowse:workspace", { detail: { view: "controls", section: "model" } }))}>
          <Blobatar name={savedConversationState.avatarSeed ?? sessionState.info?.model_id ?? "drowse"} size={32} background="circle" alt="" />
        </button>
      {:else}
        <span class="speaker-marker" aria-hidden="true"><Blobatar name={savedConversationState.avatarSeed ?? sessionState.info?.model_id ?? "drowse"} size={32} background="circle" alt="" /></span>
      {/if}
    {:else}
      <span class="speaker-marker" aria-hidden="true"><span class="user-avatar"></span></span>
    {/if}
    <span class="role-label">{roleDisplayLabel(turn.role, turn.roleLabel)}</span>
  </span>
{/snippet}

{#snippet bubble(turn: ChatTurn, turnIdx: number, isShadow: boolean)}
  {#if turn.role === "system"}
    <!-- Stage direction — a note about the scene, not a speaker. -->
    <div
      class="stage"
      class:shadow={isShadow}
      dir="auto"
      title="system prompt"
      in:fly={contentIn()}
    >{turn.text}</div>
  {:else}
  <div
    class="msg"
    class:shadow={isShadow}
    class:generation-active={genStatus.active && (isShadow ? abState.pendingTurnIdx === turnIdx : !abState.processingAb && chatLog.pendingIndex === turnIdx)}
    in:fly={contentIn()}
  >
    <div class="who">
      {@render speaker(turn)}
      {#if turn.nodeId}
        <Button
          size="sm"
          variant="flat"
          onclick={() => regenMessage(turn)}
          title="reroll this message"
          ariaLabel={`Reroll ${roleDisplayLabel(turn.role, turn.roleLabel)} message`}
        ><FluentIcon name="refresh" /></Button>
      {/if}
      {#if (turn.tokens?.length ?? 0) > 0 || (turn.thinkingTokens?.length ?? 0) > 0}
        <Button
          size="sm"
          variant="flat"
          onclick={() => inspectTurnTokens(turn, turnIdx)}
          ariaLabel={`Inspect tokens in ${roleDisplayLabel(turn.role, turn.roleLabel)} message`}
        >inspect tokens</Button>
      {/if}
      {#if isShadow && !pinnedActive}<span class="who-meta">(unsteered)</span>{/if}
      {#if genStatus.active && (isShadow ? abState.pendingTurnIdx === turnIdx : !abState.processingAb && chatLog.pendingIndex === turnIdx)}
        <span class="who-meta generation-label" role="status">{(turn.tokens?.length ?? 0) > 0 ? "Writing…" : (turn.thinkingTokens?.length ?? 0) > 0 ? "Thinking…" : "Preparing reply…"}</span>
      {/if}
      {#if turn.meanLogprob != null && Number.isFinite(turn.meanLogprob)}
        <span
          class="prov"
          title="sequence perplexity"
        >seq ppl {Math.exp(-turn.meanLogprob).toFixed(1)}</span>
      {/if}
    </div>

    {#if (turn.thinkingTokens?.length ?? 0) > 0 || turn.thinking}
      <div class="thinking-block" class:collapsed={turnCollapsed(turnIdx, turn)}>
        <button
          type="button"
          class="thinking-toggle"
          onclick={() => toggleThinking(turnIdx)}
          aria-expanded={!turnCollapsed(turnIdx, turn)}
        >
          <span class="caret" class:collapsed={turnCollapsed(turnIdx, turn)}><FluentIcon name="down" /></span>
          <span>thinking{turnCollapsed(turnIdx, turn) ? "…" : ""}</span>
        </button>
        {#if !turnCollapsed(turnIdx, turn)}
          <div
            class="thinking-body"
            dir="auto"
            in:slide={collapseIn()}
            out:slide={collapseOut()}
          >
            {#each turn.thinkingTokens ?? [] as tok, tokenIdx (tokenIdx)}
              <span
                class="tok"
                data-cursor="inspect"
                class:tinted={highlightState.target !== null}
                style={highlightStyleString(tok)}
                onpointerenter={() => beginTokenHover(tok, turn.nodeId)}
                onpointerleave={endTokenHover}
                onfocus={() => { tokenFocusIndices.set(`${turn.nodeId ?? turnIdx}:thinking`, tokenIdx); beginTokenHover(tok, turn.nodeId); }}
                onblur={endTokenHover}
                onclick={(ev) => tokenClicked(turnIdx, tokenIdx, ev, true)}
                onkeydown={(ev) => tokenKeydown(ev, turnIdx, tokenIdx, true)}
                role="button"
                tabindex={tokenIdx === (tokenFocusIndices.get(`${turn.nodeId ?? turnIdx}:thinking`) ?? 0) ? 0 : -1}
                aria-haspopup="dialog"
                aria-label={`Inspect token ${tok.text.trim() || "whitespace"}`}
              >{tok.text}</span>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    <div class="response-body" dir="auto">
      {#if (turn.tokens?.length ?? 0) > 0}
        {#each visibleResponseTokens(turn.tokens ?? []) as { tok, originalIdx }, visibleIdx (originalIdx)}
          <span
            class="tok"
            data-cursor="inspect"
            class:tinted={highlightState.target !== null}
            style={highlightStyleString(tok)}
            onpointerenter={() => beginTokenHover(tok, turn.nodeId)}
            onpointerleave={endTokenHover}
            onfocus={() => { tokenFocusIndices.set(`${turn.nodeId ?? turnIdx}:response`, originalIdx); beginTokenHover(tok, turn.nodeId); }}
            onblur={endTokenHover}
            onclick={(ev) => tokenClicked(turnIdx, originalIdx, ev, false)}
            onkeydown={(ev) => tokenKeydown(ev, turnIdx, originalIdx)}
            role="button"
            tabindex={(tokenFocusIndices.has(`${turn.nodeId ?? turnIdx}:response`) ? tokenFocusIndices.get(`${turn.nodeId ?? turnIdx}:response`) === originalIdx : visibleIdx === 0) ? 0 : -1}
            aria-haspopup="dialog"
            aria-label={`Inspect token ${tok.text.trim() || "whitespace"}`}
          >{tok.text}</span>
        {/each}
      {:else}
        <span class="plain">{plainResponseText(turn)}</span>
      {/if}
    </div>
  </div>
  {/if}
{/snippet}

<style>
  .chat-panel {
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    gap: var(--space-2);
    height: 100%;
    min-width: 0;
    min-height: 0;
  }
  .active-model {
    display: flex;
    flex: 0 0 auto;
    align-self: flex-end;
    align-items: baseline;
    justify-content: flex-end;
    gap: var(--space-2);
    max-width: 100%;
    color: var(--fg-muted);
    font-family: var(--font-ui);
    font-size: var(--text-sm);
    line-height: 1.5;
  }
  .active-model-name { min-width: 0; overflow-wrap: anywhere; text-align: end; }
  .chat {
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    overflow-y: auto;
    overscroll-behavior: contain;
    scrollbar-gutter: stable both-edges;
    gap: var(--space-2);
    font-family: var(--font-reading);
    font-size: var(--text);
    color: var(--fg);
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--workspace-panel-bg);
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .chat-header {
    display: grid;
    flex: 0 0 auto;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--group-gap);
    padding-bottom: var(--space-2);
    color: var(--fg-dim);
    font-size: var(--text-sm);
  }
  .reading-tools {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--cluster-gap);
    min-width: 0;
  }
  .color-control,
  .comparison-control {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .ctl {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .ctl-inline {
    cursor: pointer;
    user-select: none;
  }
  .ctl-label {
    color: var(--fg-dim);
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
    letter-spacing: 0;
  }
  /* Layout host for the themed Select — Select owns its own theme. */
  .ctl-select {
    display: inline-flex;
    min-width: 9em;
  }

  .compare-color {
    min-height: var(--control-target);
    padding: var(--space-1) var(--space-3);
    border: 0;
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg-dim);
    font: inherit;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    cursor: pointer;
    transition:
      background var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .compare-color:hover,
  .compare-color.active {
    background: var(--glass);
    color: var(--fg);
  }
  .compare-color:active { transform: scale(var(--press-scale)); }

  /* Conversation-actions strip — inline, pushed to the right edge of
   * the header.  Wraps onto a second row on narrow layouts. */
  .header-actions {
    display: flex;
    align-items: center;
    gap: var(--cluster-gap);
    flex-wrap: nowrap;
    justify-content: flex-end;
  }
  .ctl-input {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    min-width: 14em;
  }
  .ctl-input:focus {
    outline: none;
    border-color: var(--accent);
  }

  .log {
    flex: 1 1 auto;
    overflow-y: auto;
    overflow-x: hidden;
    display: flex;
    flex-direction: column;
    gap: var(--space-md);
    min-height: 0;
    padding-inline-end: var(--space-2);
  }

  .conversation-empty {
    display: flex;
    flex: 1 1 auto;
    align-items: center;
    justify-content: center;
    margin: auto;
    padding: var(--space-8) var(--space-5);
    text-align: center;
  }

  .conversation-empty p {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text);
  }

  .log.ab {
    /* Container itself stays vertical scroll; the inner ab-grid handles
     * the two-column rendering so each column shares the same scroll
     * surface. */
    display: block;
  }
  .ab-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-4);
  }
  .ab-col {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
  }
  .pin-header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    background: color-mix(in srgb, var(--pillar-manifold) 10%, transparent);
    border-radius: var(--radius);
    color: var(--pillar-manifold);
    font-size: var(--text-xs);
    margin-bottom: var(--space-2);
  }
  .pin-tag {
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .pin-id {
    color: var(--accent-yellow);
    flex: 1 1 auto;
  }
  .pin-unpin {
    background: var(--glass);
    color: var(--fg-dim);
    border: 0;
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-2);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
    min-height: var(--control-target);
  }
  .pin-unpin:hover {
    color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 12%, transparent);
  }
  /* The neutral fill groups each turn; the border marks A/B comparisons. */
  .msg {
    border: 1px solid transparent;
    border-radius: var(--radius-lg);
    background: var(--workspace-message-bg, var(--surface-card));
    box-shadow: var(--shadow-rack);
    padding: var(--surface-padding);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
    word-break: break-word;
  }
  .msg.shadow {
    border-color: color-mix(in srgb, var(--pillar-manifold) 26%, transparent);
  }
  .msg.generation-active {
    box-shadow: var(--shadow-rack);
  }
  .generation-label { display: inline-flex; align-items: center; gap: var(--space-2); }

  .who {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .role-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    letter-spacing: 0;
    text-transform: none;
    color: var(--fg-dim);
    padding: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    max-width: 40%;
    white-space: nowrap;
  }
  .role-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .model-avatar, .speaker-marker { display: inline-grid; place-items: center; width: var(--control-target); height: var(--control-target); flex: none; }
  .model-avatar { padding: 0; border: 0; border-radius: var(--radius-pill); background: transparent; cursor: pointer; }
  .model-avatar:hover { background: var(--bg-hover); }
  .model-avatar:active { transform: scale(0.96); }
  .model-avatar:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .model-avatar :global(img), .speaker-marker :global(img) { display: block; width: 32px; height: 32px; border-radius: var(--radius-pill); outline: 1px solid oklch(1 0 0 / 0.1); outline-offset: -1px; }
  :global(:root[data-theme="light"]) .model-avatar :global(img), :global(:root[data-theme="light"]) .speaker-marker :global(img) { outline-color: oklch(0 0 0 / 0.1); }
  .user-avatar { width: 24px; height: 24px; border-radius: var(--radius-pill); background: var(--accent); }
  @media (forced-colors: active) { .msg { border-color: CanvasText; } .user-avatar { background: Highlight; forced-color-adjust: none; } }
  .who-meta {
    color: var(--fg-muted);
    font-family: var(--font-data);
    font-size: var(--text-xs);
  }
  .prov {
    margin-inline-start: auto;
    color: var(--fg-muted);
    font-family: var(--font-data);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    flex: none;
  }

  /* Stage direction — the system prompt as a note about the scene. */
  .stage {
    font-style: italic;
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    line-height: 1.5;
    padding: var(--space-1) var(--space-5);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .stage.shadow {
    opacity: 0.7;
  }

  .msg.placeholder {
    color: var(--fg-muted);
    font-style: italic;
    opacity: 0.6;
  }
  .placeholder-text {
    font-size: var(--text-sm);
  }

  .response-body {
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--fg-strong);
    font-size: var(--workspace-message-size, var(--text-md));
    line-height: 1.55;
  }
  .plain {
    white-space: pre-wrap;
  }

  /* Thinking-collapsible block — visible-only header when collapsed,
   * with the body indented when expanded.  Borderless: the caret + the
   * italic dim body delimit it; no rules. */
  .thinking-block {
    margin-bottom: var(--space-1);
  }
  .thinking-toggle {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    font: inherit;
    font-family: var(--font-mono);
    padding: var(--space-1) 0;
    cursor: pointer;
    text-align: start;
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    min-height: var(--control-target);
  }
  .thinking-toggle:hover {
    color: var(--fg-strong);
  }
  .thinking-toggle .caret {
    color: var(--fg-muted);
    display: inline-flex;
    transition: transform var(--dur) var(--ease-out);
  }
  .thinking-toggle .caret.collapsed { transform: rotate(-90deg); }
  .thinking-body {
    /* The inline-start pad is a hanging indent tuned to the caret width. */
    padding-block: var(--space-1) var(--space-2);
    padding-inline: var(--space-md) 0;
    color: var(--fg-dim);
    font-style: italic;
    white-space: pre-wrap;
    line-height: 1.4;
  }
  .thinking-body .tok {
    font-style: italic;
  }

  /* Tokens — minimal padding so the tinted span hugs the glyph.  The
   * click handler attaches regardless of highlight state; ``.tinted``
   * marks rows whose background is being painted by the score so the
   * untinted hover outline only fires when there's no other visual.
   * Hover outline gives the click affordance even when highlighting is
   * off (matches the user-visible click contract). */
  .tok {
    cursor: pointer;
    border-radius: var(--radius);
  }
  .tok:hover {
    outline: 1px solid var(--fg-muted);
  }

  /* The visible next-turn plan names roles; each option carries its structural
   * seat internally so the composer never makes the user manage both. */
  .turn-plan-shell {
    min-width: 0;
  }
  .turn-plan {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-group);
    background: transparent;
    --control-field: var(--control-target);
    --control-compact: var(--control-target);
  }
  .plan-card {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .plan-actor {
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    letter-spacing: 0;
    text-transform: none;
    color: var(--fg-muted);
  }
  .reply-toggle {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-muted);
    font-family: var(--font-structure);
    font-size: var(--text-xs);
    cursor: pointer;
  }
  .plan-actions {
    align-self: center;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: var(--control-target);
  }
  .plan-minimize,
  .turn-plan-summary {
    border: 0;
    font: inherit;
    font-family: var(--font-structure);
    cursor: pointer;
    transition:
      background var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .plan-minimize {
    min-height: var(--control-target);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius);
    background: var(--glass);
    box-shadow: var(--shadow-control);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    font-weight: var(--weight-structure);
    white-space: nowrap;
  }
  .plan-minimize:hover {
    background: var(--glass-strong);
    box-shadow: var(--shadow-control-hover);
    color: var(--fg);
  }
  .turn-plan-summary {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-3);
    width: 100%;
    min-height: var(--control-target);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-lg);
    background: var(--workspace-panel-bg);
    color: var(--fg-dim);
    text-align: start;
  }
  .turn-plan-summary:hover {
    background: var(--glass);
    color: var(--fg);
  }
  .plan-minimize:active,
  .turn-plan-summary:active {
    transform: scale(var(--press-scale));
  }
  .plan-minimize:focus-visible,
  .turn-plan-summary:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  .summary-label,
  .summary-action {
    font-size: var(--text-xs);
    font-weight: var(--weight-structure);
  }
  .summary-label {
    color: var(--fg-muted);
    text-transform: none;
    letter-spacing: 0;
  }
  .summary-value {
    min-width: 0;
    overflow: hidden;
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .summary-action {
    color: var(--accent);
  }
  .role-error {
    margin: calc(-1 * var(--space-2)) var(--space-6) 0;
    color: var(--accent-red);
    font-size: var(--text-sm);
    line-height: 1.45;
  }

  @media (max-width: 720px) {
    .turn-plan {
      grid-template-columns: minmax(0, 1fr);
      align-items: stretch;
      padding: var(--space-4);
    }
    .plan-actions {
      grid-column: 1;
      justify-self: start;
      flex-wrap: wrap;
    }
  }

  @media (max-width: 560px) {
    .ab-grid {
      grid-template-columns: minmax(0, 1fr);
    }
  }

  @media (min-width: 381px) and (max-width: 560px) {
    .turn-plan {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      grid-template-rows: auto auto;
      gap: var(--space-2);
      padding: var(--space-3);
    }
    .plan-card {
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-2);
    }
    .plan-card:first-child {
      grid-column: 1;
      grid-row: 1;
    }
    .plan-card:nth-child(2) {
      grid-column: 2;
      grid-row: 1;
    }
    .plan-actions {
      grid-column: 1 / -1;
      grid-row: 2;
      justify-self: stretch;
      justify-content: space-between;
    }
  }

  .thinking-row {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: 0 var(--space-1);
  }
  .thinking-control {
    align-self: flex-start;
  }
  .thinking-box {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .thinking-input {
    background: var(--workspace-field-bg);
    color: var(--fg-dim);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-family: var(--font-reading);
    font-size: var(--text-sm);
    padding: var(--surface-padding);
    resize: vertical;
    min-height: 44px;
    transition: box-shadow var(--dur-fast) var(--ease-out);
  }
  .thinking-input:focus-visible {
    outline: none;
    color: var(--fg);
  }
  .thinking-warn {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-style: italic;
  }

  .composer-resizer-shell {
    position: relative;
    display: grid;
    flex: 0 0 auto;
    place-items: center;
    width: 100%;
    min-height: var(--control-target);
    margin-block: 0;
    border-radius: var(--radius-sm);
    pointer-events: none;
  }
  .composer-resizer {
    position: absolute;
    inset-inline: 0;
    inset-block: 0;
    z-index: 1;
    width: 5.5rem;
    height: 100%;
    margin: 0 auto;
    opacity: 0;
    cursor: row-resize;
    touch-action: none;
    pointer-events: auto;
  }
  .composer-resizer-shell span {
    width: 3.5rem;
    height: 3px;
    border-radius: var(--radius-pill);
    background: var(--glass-strong);
    transition:
      width var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .composer-resizer:hover + span,
  .composer-resizer-shell:focus-within span,
  .composer-resizer-shell.dragging span {
    width: 4.5rem;
    background: var(--accent);
  }
  .composer-resizer-shell:focus-within span {
    box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px var(--focus-ring);
  }

  .input-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--composer-space, var(--space-6));
    padding: var(--surface-padding);
    border-radius: var(--popup-radius);
    background: var(--workspace-field-bg);
    -webkit-backdrop-filter: blur(1px);
    backdrop-filter: blur(1px);
  }
  .input {
    --shadow-field-focus: none;
    --shadow-field-keyboard: none;
    width: 100%;
    background: transparent;
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-xs);
    font: inherit;
    font-family: var(--font-reading);
    resize: none;
    /* border-box lets autosize() write ``scrollHeight`` straight into
     * ``style.height`` without a padding/border double-count — without
     * this the one-line draft height was off by ~6px and the textarea's
     * vertical scrollbar leaked through as a tiny up/down nub. */
    box-sizing: border-box;
    overflow-y: hidden;
    min-height: 64px;
    max-height: 480px;
    line-height: 1.45;
    transition: box-shadow var(--dur-fast) var(--ease-out);
  }
  .input:focus {
    outline: none;
  }
  .input-actions {
    display: flex;
    gap: var(--composer-space, var(--space-6));
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
  }
  .clear-conversation {
    min-height: var(--control-target);
    padding: var(--space-2) var(--space-4);
    border: 0;
    border-radius: var(--radius);
    background: var(--danger-bg);
    color: var(--accent-red);
    font: inherit;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    white-space: nowrap;
    cursor: pointer;
    transition:
      background var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .clear-conversation:hover:not(:disabled),
  .clear-conversation.confirm-clear {
    background: var(--danger-hover);
  }
  .clear-conversation:active:not(:disabled) {
    transform: scale(var(--press-scale));
  }
  .clear-conversation:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  .clear-conversation:disabled {
    cursor: not-allowed;
    opacity: 0.45;
  }

  @media (prefers-contrast: more), (forced-colors: active), (prefers-reduced-transparency: reduce) {
    .input-row { -webkit-backdrop-filter: none; backdrop-filter: none; }
  }

  @media (max-width: 860px) {
    .chat-header {
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-6);
    }
    .header-actions {
      justify-content: flex-start;
      flex-wrap: wrap;
    }
  }

  @media (max-width: 620px) {
    .reading-tools {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      align-items: stretch;
      gap: var(--space-3);
    }
    .color-control {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      width: 100%;
    }
    .color-control .ctl {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      width: 100%;
    }
    .ctl-select {
      flex: 1 1 auto;
      min-width: 0;
      width: 100%;
    }
    .header-actions {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      align-items: center;
      gap: var(--space-3);
    }
    .header-actions > .ctl-select,
    .header-actions > .ctl-input {
      grid-column: 1 / -1;
    }
    .msg {
      padding: var(--surface-padding);
    }
    .response-body {
      font-size: var(--text);
    }
  }

  @media (max-width: 420px) {
    .input-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
    }
    .input {
      width: 100%;
      min-height: 92px;
    }
    .input-actions {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content;
      align-items: stretch;
    }
    .input-actions.has-clear { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .input-actions.has-clear :global(button:first-child) {
      grid-column: 1 / -1;
    }
    .input-actions :global(button) {
      width: 100%;
      min-height: var(--control-target);
    }
    .clear-conversation {
      padding-inline: var(--space-2);
      white-space: normal;
    }
  }

  /* Wrap compact controls so scrolling does not clip their focus halos. */
  @media (max-width: 620px), (max-height: 600px) {
    .chat-header {
      display: flex;
      flex: 0 0 auto;
      flex-wrap: wrap;
      align-items: center;
      gap: var(--space-2);
      min-height: var(--control-target);
      padding-bottom: 0;
      overflow: visible;
    }

    .reading-tools,
    .header-actions {
      display: flex;
      flex: 0 1 auto;
      align-items: center;
      flex-wrap: wrap;
      gap: var(--space-2);
    }

    .color-control,
    .comparison-control,
    .color-control .ctl {
      display: flex;
      flex: 0 0 auto;
      align-items: center;
      width: auto;
    }

    .ctl-label {
      display: none;
    }

    .ctl-select {
      flex: 0 0 auto;
      width: 8.75rem;
      min-width: 8.75rem;
    }

    .header-actions > .ctl-select,
    .header-actions > .ctl-input {
      grid-column: auto;
    }

    .input {
      min-height: 64px;
    }
  }

  @media (max-width: 620px) {
    .chat {
      overflow-y: auto;
      overscroll-behavior: contain;
      scroll-padding-block-end: var(--surface-padding);
      scrollbar-gutter: auto;
      padding: var(--space-2);
    }

    .input-row { padding: var(--space-2); }

    .chat > :global(*) {
      flex-shrink: 0;
    }

    .chat > .log {
      flex: 1 1 0;
      min-height: var(--control-target);
    }

    .chat-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      width: 100%;
      overflow: visible;
    }

    .reading-tools,
    .header-actions {
      width: 100%;
      min-width: 0;
      flex-wrap: wrap;
      justify-content: space-between;
      overflow: visible;
    }

    .comparison-control {
      flex: 1 1 auto;
      min-width: 0;
      justify-content: space-between;
    }
  }

  @media (max-height: 600px) {
    .chat {
      gap: var(--space-1);
      overflow-y: auto;
      overscroll-behavior: contain;
      scroll-padding-block-end: var(--control-target);
    }

  }

  @media (min-width: 621px) and (max-height: 600px) {
    .chat:not(:has(.chat-header)) {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 5.5rem;
      grid-template-rows: minmax(24px, 1fr);
      grid-auto-rows: max-content;
      padding: 0;
      scrollbar-gutter: auto;
    }
    .chat:not(:has(.chat-header)) > :global(*) { grid-column: 1 / -1; }
    .chat:not(:has(.chat-header)) > .log { grid-row: 1; min-height: 24px; }
    .chat:not(:has(.chat-header)) > .composer-resizer-shell { grid-column: 2; grid-row: 2; }
    .chat:not(:has(.chat-header)) > :global(.status-footer) { grid-column: 1; grid-row: 2; }
    .input-row {
      grid-template-columns: minmax(0, 1fr) max-content;
      align-items: end;
      padding: var(--space-1);
    }
  }

  @media (max-width: 760px), (max-height: 600px) {
    .input,
    .ctl-input,
    .thinking-input {
      font-size: var(--text-input-touch);
    }

    .composer-resizer-shell {
      min-height: var(--control-target);
      margin-block: 0;
    }

    .composer-resizer {
      inset-block: 0;
      height: 100%;
    }
  }
</style>
